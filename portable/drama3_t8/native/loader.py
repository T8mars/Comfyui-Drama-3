"""Load the converted Drama 3 files (`tools/convert.py`) as ComfyUI-managed models."""
import gc
import json
import logging
import os
import contextlib
import inspect
import sys

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
from .codec import Drama3Codec
from .config import parse_config
from .model import Drama3Model, SecondGPU
from .offload import ExpertStore

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish", "tokenizer.json")
# What each card keeps besides its share of the layers: the first card holds the embedding table, the second the
# head, the tower and the caches.
FIRST_RESERVE = 3 * 1024 ** 3
SECOND_RESERVE = 4 * 1024 ** 3


class _OpsConfig:
    """`pick_operations` reads `.quant_config` off a model config; quantized files select `mixed_precision_ops`,
    which also carries the `MoEExperts` bank."""

    def __init__(self, quant_config):
        self.quant_config = quant_config


def load_checkpoint_storage(model, state):
    """Assign checkpoint tensors without allocating a second model or losing quantized storage.

    Comfy's mixed-precision modules load their weights and quantization parameters
    themselves. Unlike ordinary ``assign=True`` parameters, those custom loaders
    use ``factory_kwargs['device']`` and would move the checkpoint *onto meta* if
    the model was constructed there. Point only meta factories at the checkpoint
    device before loading; the model patcher still controls subsequent transfers.
    """
    for name, module in model.named_modules():
        factory = getattr(module, "factory_kwargs", None)
        if not isinstance(factory, dict):
            continue
        device = factory.get("device")
        if device is None or torch.device(device).type != "meta":
            continue
        key = f"{name}.weight" if name else "weight"
        weight = state.get(key)
        if weight is None and callable(getattr(module, "reset_parameters", None)) and hasattr(module, "_orig_shape"):
            # Some host quantized loaders merely warn and return when a weight
            # is absent, so load_state_dict would otherwise miss this failure.
            raise ValueError(f"Drama 3 checkpoint is missing weight: {key}")
        module.factory_kwargs = {**factory, "device": weight.device if weight is not None else torch.device("cpu")}

    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f"Drama 3 checkpoint keys: unexpected {result.unexpected_keys[:6]}, missing {result.missing_keys[:6]}")
    # A QuantizedTensor's public device is not sufficient: inspect its actual
    # packed data and every tensor-valued scale/codebook as well.
    from .offload import tensor_fields
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        fields = {"data": value}
        if hasattr(value, "_qdata"):
            fields = {"data": value._qdata, **tensor_fields(value)}
        for field, tensor in fields.items():
            if tensor.is_meta:
                raise ValueError(f"Drama 3 checkpoint left unmaterialized storage: {name}.{field}")
    return result


class Drama3:
    """A loaded Drama 3: its patchers, where each part runs, and the tokenizer."""

    def __init__(self, model, patcher, second=None, second_patcher=None, experts=None, checkpoint_reader=None):
        from tokenizers import Tokenizer
        self.model = model
        self.patcher = patcher
        self.second = second
        self.second_patcher = second_patcher
        self.experts = experts
        self.checkpoint_reader = dict(checkpoint_reader or {})
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    @property
    def config(self):
        return self.model.config

    def patchers(self):
        return [p for p in (self.patcher, self.second_patcher) if p is not None]

    def load_resident(self, extra_patchers=(), memory_required=0):
        """Load and verify full single-GPU weights, including optional scene heads."""
        from .resident import ResidentExperts
        if not isinstance(self.experts, ResidentExperts):
            raise ValueError('load_resident requires full VRAM expert placement')
        self.experts.load(self.patcher.load_device, [*self.patchers(), *extra_patchers], memory_required)
        return self

    def expert_stats(self):
        from .resident import ResidentExperts
        if isinstance(self.experts, ResidentExperts):
            return self.experts.stats()
        store = self.experts
        return dict(placement='ram', requested_backend=getattr(store, 'requested_backend', 'torch'),
                    actual_backend=getattr(store, 'last_backend', 'torch'),
                    fallback_reason=getattr(store, 'fallback_reason', None),
                    backend_calls=dict(getattr(store, 'backend_calls', {})),
                    cpu_experts_run=getattr(store, 'cpu_experts_run', 0),
                    cpu_decode_calls=getattr(store, 'cpu_decode_calls', 0),
                    sync_free=getattr(store, 'sync_free', False))

    def session(self, cpu_decode=False, extra_patchers=()):
        """Keep the expert cache across related takes; inner generations reuse it.

        CPU decode is opt-in because the caller must service CPU work between
        layers/steps. The reference Torch store does not implement this option.
        """
        if self.experts is None:
            return contextlib.nullcontext()
        options = {}
        if "cpu_decode" in inspect.signature(self.experts.session).parameters:
            options["cpu_decode"] = cpu_decode
        return self.experts.session(self.patcher.load_device, [*self.patchers(), *extra_patchers], **options)

    def parts(self):
        """(layers, layer devices, final norm, audio head, tower): the head and tower run on the last device."""
        backbone = self.model.model
        first = self.patcher.load_device
        if self.second is None:
            layers = list(backbone.layers)
            return layers, [first] * len(layers), backbone.norm, self.model.audio_head, self.model.tower
        second = self.second_patcher.load_device
        layers = list(backbone.layers) + list(self.second.layers)
        devices = [first] * len(backbone.layers) + [second] * len(self.second.layers)
        return layers, devices, self.second.norm, self.second.audio_head, self.second.tower


def layer_bytes(layer):
    return sum(t.numel() * t.element_size() for t in layer.state_dict().values())


def auto_split(model, first_device, second_device):
    """The first layer index the second card runs: as many layers on the first card as its free memory allows,
    the rest on the second, which also holds the head and the tower."""
    sizes = [layer_bytes(layer) for layer in model.model.layers]
    tail = sum(t.numel() * t.element_size() for m in (model.audio_head, model.tower, model.model.norm) for t in m.state_dict().values())
    budget = comfy.model_management.get_total_memory(first_device) - FIRST_RESERVE - layer_bytes(model.model.embed_tokens)
    second_budget = comfy.model_management.get_total_memory(second_device) - SECOND_RESERVE - tail
    # balance the two cards' headroom
    best = None
    for first in range(1, len(sizes)):
        left = budget - sum(sizes[:first])
        right = second_budget - sum(sizes[first:])
        score = min(left, right)
        if best is None or score > best[0]:
            best = (score, first)
    if best[0] < 0:
        logging.warning("Drama 3: the model does not fit on both cards; ComfyUI will stream the rest of it")
    return best[1]


def _clear_failed_load_frames(error, owner_frame):
    """Release this load's internal copies without clearing user tracebacks.

    An exception may refer to a prior, unrelated cause or an external callback.
    Only frames within this exact active load and an owned implementation root
    can contain temporary copies that this loader is entitled to release.
    """
    modules = [torch, comfy.ops, sys.modules.get('comfy_kitchen')]
    roots = [os.path.normcase(os.path.abspath(os.path.dirname(__file__)))]
    for implementation in modules:
        filename = getattr(implementation, '__file__', None)
        if filename:
            roots.append(os.path.normcase(os.path.abspath(os.path.dirname(filename))))
    pending, seen, frames, frame_ids = [error], set(), [], set()
    current = tb = frame = ancestor = None
    try:
        # Collect first: clearing an inner frame must not change the ancestry
        # checks for another exception or linked cause.
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            tb = current.__traceback__
            while tb is not None:
                frame = tb.tb_frame
                filename = os.path.normcase(os.path.abspath(frame.f_code.co_filename))
                owned = False
                for root in roots:
                    try:
                        if os.path.commonpath((filename, root)) == root:
                            owned = True
                            break
                    except ValueError:
                        pass  # Different Windows volumes cannot share a root.
                if owned and frame is not owner_frame and id(frame) not in frame_ids:
                    ancestor = frame.f_back
                    while ancestor is not None and ancestor is not owner_frame:
                        ancestor = ancestor.f_back
                    if ancestor is owner_frame:
                        frames.append(frame)
                        frame_ids.add(id(frame))
                tb = tb.tb_next
            for linked in (current.__cause__, current.__context__):
                if linked is not None:
                    pending.append(linked)
        for frame in frames:
            try:
                frame.clear()
            except RuntimeError:
                pass  # Python does not allow clearing an executing frame.
    finally:
        pending.clear()
        frames.clear()
        current = tb = frame = ancestor = owner_frame = None


def _failed_load_cleanup(operation, description, *args):
    """Attempt owned rollback without replacing the original load exception."""
    try:
        operation(*args)
        return True
    except BaseException as cleanup_error:
        try:
            logging.warning('Drama 3 failed-load %s: %s: %s', description,
                            type(cleanup_error).__name__, str(cleanup_error))
        except BaseException:
            pass  # Diagnostics must not hide the original cancellation/error.
        return False
    finally:
        operation = None
        args = ()


def _clear_unpublished_model(model):
    for module in model.modules():
        module._parameters.clear()
        module._buffers.clear()


def _release_failed_load_pins():
    from .acceleration import release_unused_pinned_memory
    release_unused_pinned_memory()


def _diagnostic_text(value,limit=512):
    return (value if len(value)<=limit else value[:limit-3]+'...') if type(value) is str else None


def _diagnostic_numbers(value,keys):
    if type(value) is not dict:return {}
    return {key:value[key] for key in keys if type(value.get(key)) in (int,float) and 0<=value[key]<2**64}


def checkpoint_load_summary(reader,experts=None,incremental=None,requested=None,*,outcome='loaded',proof_present=None):
    """Fixed-field JSON facts only; never copy state/tensors or check arrays."""
    reader=reader if type(reader) is dict else {}
    owner=reader.get('expert_ownership');owner=owner if type(owner) is dict else {}
    admission=getattr(experts,'admission',None);admission=admission if type(admission) is dict else {}
    host=reader.get('host_context');host=host if type(host) is dict else {}
    checks=reader.get('memory_checks');checks=checks if type(checks) is list else []
    numeric=('available_bytes','required_bytes','reserve_bytes','maximum_remaining_bank_bytes','layer_destination_bytes')
    result={'version':1,'outcome':_diagnostic_text(outcome,32),'requested_backend':_diagnostic_text(requested,32),
        'actual_reader':_diagnostic_text(reader.get('actual_reader'),64),
        'reader_fallback_reason':_diagnostic_text(reader.get('fallback_reason')),
        **_diagnostic_numbers(reader,('source_cpu_bytes','temporary_margin_bytes','header_bytes')),
        'source_header_sha256':_diagnostic_text(reader.get('header_sha256'),64),
        'host_context':{key:host[key] for key in ('aimdo_enabled','disable_mmap') if type(host.get(key)) is bool},
        'memory_check_count':len(checks),'ownership_proof_present':proof_present if type(proof_present) is bool else None,
        'ownership_incremental':incremental if type(incremental) is bool else None,
        'ownership':{'policy':_diagnostic_text(owner.get('policy'),128),
            'fallback_reason':_diagnostic_text(owner.get('fallback_reason')),
            **{key:owner[key] for key in ('read_certificate_issued','proof_issued') if type(owner.get(key)) is bool},
            **_diagnostic_numbers(owner,('completed_layers','release_count'))},
        'admission':{'policy':_diagnostic_text(admission.get('policy'),128),
            'ownership_refusal':_diagnostic_text(admission.get('ownership_refusal')),
            **{key:admission[key] for key in ('selected_accelerated_store',) if type(admission.get(key)) is bool},
            **_diagnostic_numbers(admission,numeric)},
        'selected_store':_diagnostic_text(type(experts).__module__+'.'+type(experts).__qualname__,192) if experts is not None else None}
    if checks:
        result['memory_check_edges']={'first':_diagnostic_numbers(checks[0],numeric),'last':_diagnostic_numbers(checks[-1],numeric)}
    bank=getattr(experts,'pin_ram_admission',None)
    if type(bank) is dict:
        result['bank_refusal']={'bank':_diagnostic_text(bank.get('bank'),128),
            **_diagnostic_numbers(bank,('physical_bytes','available_bytes','required_bytes','reserve_bytes','total_bytes'))}
    return result


def load_drama3(path, second_device=None, split=0, experts_in_ram=False, expert_cache_gb=0.0,moe_backend="torch"):
    """Auto may retry exactly once after a verified pin-RAM refusal/rollback."""
    from .checkpoint_ownership import IncrementalPinRAMAdmissionError
    refusal=None
    try:
        return _load_drama3_once(path,second_device,split,experts_in_ram,expert_cache_gb,moe_backend)
    except IncrementalPinRAMAdmissionError as error:
        if type(error) is not IncrementalPinRAMAdmissionError or moe_backend!='auto' or not experts_in_ram or not error.partial_cleanup_complete:
            raise
        reason=(_diagnostic_text(error.args[0],1024) if len(error.args)==1 else None) or type(error).__name__
        refusal={'requested_backend':'auto','actual_backend':'torch','attempts':2,
                 'scope':error.admission.get('scope','verified_incremental_pin_ram_admission'),'partial_cleanup_complete':True,
                 'reason':reason,'admission':dict(error.admission),
                 'first_attempt':getattr(error,'load_diagnostics',None)}
    # The first except scope and all partial-load cleanup have finished. No
    # model/exception frame is used as the input to this independent read.
    comfy.model_management.throw_exception_if_processing_interrupted()
    logging.warning('Drama 3 auto backend: reloading once with Torch after complete pin rollback: %s',refusal['reason'])
    result=_load_drama3_once(path,second_device,split,experts_in_ram,expert_cache_gb,'torch',_disable_expert_pinning=True)
    result.experts.requested_backend='auto'
    result.experts.fallback_reason='Automatic pin-RAM recovery: '+refusal['reason']
    result.checkpoint_reader['automatic_backend_fallback']=refusal
    summary=result.checkpoint_reader.get('load_summary')
    if type(summary) is dict:
        result.checkpoint_reader['load_summary']={**summary,'requested_backend':'auto',
            'recovery':{key:refusal[key] for key in ('actual_backend','attempts','scope','partial_cleanup_complete','reason','first_attempt')}}
    return result


def _load_drama3_once(path, second_device=None, split=0, experts_in_ram=False, expert_cache_gb=0.0,moe_backend="torch",*,_disable_expert_pinning=False):
    if moe_backend not in {'auto', 'torch', 'cuda'}:
        raise ValueError('MoE backend must be auto, torch or cuda')
    if not experts_in_ram and second_device is not None:
        raise ValueError('Full VRAM expert placement supports one CUDA device')
    load_device = comfy.model_management.get_torch_device()
    if not experts_in_ram and load_device.type != 'cuda':
        from .resident import ResidentMemoryError
        raise ResidentMemoryError('Full VRAM placement requires one CUDA GPU; select RAM placement on this host')
    if not experts_in_ram:
        from .resident import preflight_resident_checkpoint
        preflight_resident_checkpoint(path, load_device)
    state = model = patcher = experts = second = second_patcher = layer = None
    ownership_module = ownership = assignment = None
    reader = {}
    load_failed = False
    cleanup_complete = True
    incremental = None
    try:
        if experts_in_ram:
            from .checkpoint_reader import load_torch_file
            state, metadata = load_torch_file(path, return_metadata=True, diagnostics=reader,
                                             _with_ownership=moe_backend != 'torch')
        else:
            state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
            reader.update(actual_reader='host', fallback_reason='VRAM placement retains the host checkpoint loader')
        checks = reader.get('memory_checks') or []
        logging.info('Drama 3 checkpoint reader: actual_reader=%s fallback_reason=%r '
                     'source_bytes=%s ram_checks=%d first_available_ram=%s last_available_ram=%s',
                     reader.get('actual_reader'), reader.get('fallback_reason'), reader.get('source_cpu_bytes'),
                     len(checks), checks[0].get('available_bytes') if checks else None,
                     checks[-1].get('available_bytes') if checks else None)
        if not metadata or "drama3_config" not in metadata:
            raise ValueError(f"{os.path.basename(path)} is not a converted Drama 3 file (tools/convert.py model)")
        config = parse_config(json.loads(metadata["drama3_config"]))
        config.fp8_experts = metadata.get("drama3_format") == "fp8"
        quant_config = comfy.utils.detect_layer_quantization(state, "")
        dtype = torch.bfloat16
        offload_device = comfy.model_management.unet_offload_device()
        operations = comfy.ops.pick_operations(dtype, dtype, load_device=load_device, model_config=_OpsConfig(quant_config))
        model = Drama3Model(config, device=torch.device("meta"), dtype=dtype, operations=operations)
        patcher_class = comfy.model_patcher.CoreModelPatcher if experts_in_ram else comfy.model_patcher.ModelPatcher
        patcher = patcher_class(model, load_device=load_device, offload_device=offload_device)
        if experts_in_ram:
            from . import checkpoint_ownership as ownership_module
            assignment = ownership_module.assign_owned(model, state, diagnostics=reader)
        else:
            load_checkpoint_storage(model, state)
        state.clear()
        state = None
        if ownership_module is not None:
            ownership = ownership_module.finish_assignment(assignment, model, diagnostics=reader)
            assignment = None
        comfy.model_management.archive_model_dtypes(model)
        if experts_in_ram:
            from .acceleration import expert_store
            experts = expert_store(model,int(expert_cache_gb * 1024 ** 3),moe_backend,ownership=ownership)
            if _disable_expert_pinning:
                experts.pin_memory=False
                prior=dict(getattr(experts,'admission',{}).get('legacy_pin_admission',{}))
                experts.admission['legacy_pin_admission']={**prior,'pin_memory':False,
                    'original_pin_memory':bool(prior.get('pin_memory',False)),
                    'fallback_reason':'Automatic RAM recovery disables additional pinned copies'}
            incremental = ownership_module.is_incremental(ownership, model, experts)
            admission = getattr(experts, 'admission', None)
            if admission is not None:
                reader.setdefault('expert_ownership', {}).update(policy=admission['policy'], admission=admission)
                if admission.get('ownership_refusal') is not None:
                    reader['expert_ownership']['fallback_reason']=admission['ownership_refusal']
            if incremental:
                reader['expert_ownership'].update(layer_releases=[], layer_admissions=[], release_count=0)
            for index, layer in enumerate(model.model.layers):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if incremental:
                    reader['expert_ownership']['layer_admissions'].append(ownership_module.begin_layer(ownership, model, index))
                layer.mlp.experts = experts.add(layer.mlp.experts)
                comfy.model_management.throw_exception_if_processing_interrupted()
                if getattr(experts, 'pin_requested', False) and getattr(experts, 'pin_fallback_reason', None):
                    from .checkpoint_reader import ReaderResourceError
                    from .offload_accelerated import pin_ram_refusal
                    admission_refusal=pin_ram_refusal(experts)
                    if admission_refusal is not None:
                        raise ownership_module.IncrementalPinRAMAdmissionError(
                            f'Drama 3 expert pinning stopped at layer {index}: {experts.pin_fallback_reason}. '
                            'Loading aborted before publishing a partially pinned model.',
                            {**admission_refusal,'scope':'accelerated_bank_pin_ram_admission','incremental':incremental})
                    raise ReaderResourceError(
                        f'Drama 3 expert pinning stopped at layer {index}: {experts.pin_fallback_reason}. '
                        'Loading aborted before publishing a partially pinned model.')
                if incremental:
                    released = ownership_module.complete_layer(ownership, model, index)
                    reader['expert_ownership']['layer_releases'].append(released)
                    reader['expert_ownership'].update(completed_layers=released['completed_layers'],
                                                      release_count=released['completed_layers'])
        else:
            from .resident import ResidentExperts
            experts = ResidentExperts(model, moe_backend)
        if second_device is None:
            reader['load_summary']=checkpoint_load_summary(reader,experts,incremental,moe_backend,proof_present=ownership is not None)
            return Drama3(model, patcher, experts=experts, checkpoint_reader=reader)
        first = split or auto_split(model, load_device, second_device)
        second = SecondGPU(model, first)
        second_patcher = comfy.model_patcher.CoreModelPatcher(second, load_device=second_device, offload_device=offload_device)
        logging.info("Drama 3: layers 0-%d on %s, layers %d-%d, head and tower on %s", first - 1, load_device, first,
                     config.text.num_hidden_layers - 1, second_device)
        reader['load_summary']=checkpoint_load_summary(reader,experts,incremental,moe_backend,proof_present=ownership is not None)
        return Drama3(model, patcher, second, second_patcher, experts, checkpoint_reader=reader)
    except BaseException as error:
        load_failed = True
        try:
            error.load_diagnostics=checkpoint_load_summary(reader,experts,incremental,moe_backend,
                outcome='failed',proof_present=ownership is not None)
            error.load_diagnostics['error_type']=_diagnostic_text(type(error).__name__,128)
        except Exception:
            pass  # Observations cannot mask the original load failure.
        if state is not None:
            cleanup_complete = _failed_load_cleanup(state.clear, 'checkpoint state cleanup') and cleanup_complete
        close = getattr(experts, 'close', None)
        if close is not None:
            cleanup_complete = _failed_load_cleanup(close, 'expert cleanup') and cleanup_complete
        # These models were never published or registered as loaded. Empty their
        # owned weights even when a caller retains the exception traceback.
        for partial in (model, second):
            if partial is not None:
                cleanup_complete = _failed_load_cleanup(_clear_unpublished_model, 'model storage cleanup', partial) and cleanup_complete
        cleanup_complete = _failed_load_cleanup(_clear_failed_load_frames, 'retained frame cleanup', error, inspect.currentframe()) and cleanup_complete
        state = model = patcher = experts = second = second_patcher = layer = None
        raise
    finally:
        primary_error = sys.exc_info()[1] if load_failed else None
        try:
            if ownership_module is not None:
                for proof in (assignment, ownership):
                    if primary_error is None:
                        ownership_module.close(proof)
                    else:
                        cleanup_complete = _failed_load_cleanup(ownership_module.close, 'ownership cleanup', proof) and cleanup_complete
        finally:
            assignment = ownership = proof = None
            if load_failed:
                cleanup_complete = _failed_load_cleanup(gc.collect, 'garbage collection') and cleanup_complete
                cleanup_complete = _failed_load_cleanup(_release_failed_load_pins, 'unused pinned allocator cleanup') and cleanup_complete
                if ownership_module is not None and isinstance(primary_error,ownership_module.IncrementalPinRAMAdmissionError):
                    primary_error.partial_cleanup_complete=cleanup_complete
                if type(getattr(primary_error,'load_diagnostics',None)) is dict:
                    primary_error.load_diagnostics['partial_cleanup_complete']=cleanup_complete
            primary_error = None


class Codec:
    def __init__(self, model, patcher):
        self.model = model
        self.patcher = patcher


def load_codec(path):
    state = comfy.utils.load_torch_file(path)
    load_device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.vae_offload_device()
    model = Drama3Codec(dtype=torch.float32, device=offload_device, operations=comfy.ops.manual_cast)
    comfy.model_management.archive_model_dtypes(model)
    model.load_state_dict(state, strict=True)
    return Codec(model, comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device))
