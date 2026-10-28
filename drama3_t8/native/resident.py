"""Single-CUDA-device expert execution with verified full weight residency.

There is no host expert store, slot cache, pinned copy, or CPU expert worker.
The controller retains weak module references and reads the checkpoint's existing
parameters after ComfyUI moves them.  Ordinary Torch execution remains available
when the selected CUDA operator cannot implement a layout or host weight hook.
"""
import contextlib
import logging
import json
import os
import struct
import weakref

import torch

import comfy.model_management as mm
from .acceleration import physical_bytes, physical_tensors
from .offload import FP8Experts as ReferenceFP8Experts


RESIDENT_RESERVE = 2 * 1024 ** 3


class ResidentMemoryError(RuntimeError):
    """The requested full-device placement cannot be fulfilled without streaming."""


def preflight_resident_checkpoint(path, device):
    """Reject an impossible safetensors placement before reading model tensors.

    The converted checkpoint records the physical packed bank fields directly.
    This coarse capacity gate precedes the detailed live-module/head budget; it
    does not certify runtime residency or generation. No checkpoint-wide mmap.
    """
    with open(path, 'rb') as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError('Resident placement requires a converted Drama safetensors checkpoint')
        header_size = struct.unpack('<Q', prefix)[0]
        size = os.fstat(source.fileno()).st_size
        if not 0 < header_size <= min(64 * 1024**2, size - 8):
            raise ValueError('Invalid resident safetensors header size')
        header = json.loads(source.read(header_size))
    if 'drama3_config' not in header.get('__metadata__', {}):
        raise ValueError('Resident placement requires a converted Drama safetensors checkpoint')
    weight_bytes = 0
    for name, value in header.items():
        if name == '__metadata__':
            continue
        first, last = value['data_offsets']
        if not 0 <= first <= last <= size - 8 - header_size:
            raise ValueError('Resident safetensors field exceeds the checkpoint size')
        if not name.endswith('.comfy_quant'):
            weight_bytes += last - first
    total = mm.get_total_memory(device)
    if weight_bytes + RESIDENT_RESERVE > total:
        raise ResidentMemoryError(f'Single-GPU full VRAM placement needs at least {weight_bytes / 1024**3:.2f} GiB '
            f'checkpoint weights + {RESIDENT_RESERVE / 1024**3:.2f} GiB runtime reserve; this GPU has '
            f'{total / 1024**3:.2f} GiB. Select RAM placement or a GPU with sufficient VRAM; '
            'no model tensors were loaded and streaming is not used for this mode.')
    return dict(checkpoint_weight_bytes=weight_bytes, runtime_reserve_bytes=RESIDENT_RESERVE, total_device_bytes=total)


def _on_device(module, device):
    return all(not tensor.is_meta and tensor.device == device for _, tensor in physical_tensors(module))


def _bank_description(module):
    from .offload_accelerated import kernel_bank
    if isinstance(module, ReferenceFP8Experts):
        banks = [{"weight": getattr(module, f'{name}_weight').view(torch.uint8).flatten(0, 1),
                  "scale": getattr(module, f'{name}_scale').flatten(0, 1)} for name in ('gate_up', 'down')]
        return 'fp8', banks, [module.gate_up_weight.shape[1], module.down_weight.shape[1]], (None, None)
    weights = [module.gate_up_proj.weight, module.down_proj.weight]
    found = [kernel_bank(weight) for weight in weights]
    if any(item is None for item in found) or found[0][0] != found[1][0]:
        return None
    kind = found[0][0]
    banks = [item[1] for item in found]
    rows = [bank['_qdata'].shape[0] // module.num_experts for bank in banks]
    if any(row % 16 for row in rows):
        return None
    books = [weight._params.codebook for weight in weights] if kind == 'w4' else (None, None)
    return kind, banks, rows, books


def _host_hook_reason(module):
    for child in module.modules():
        if any(getattr(child, name, None) for name in ('weight_function', 'bias_function',
                                                      'weight_lowvram_function', 'bias_lowvram_function')):
            return 'Host weight hooks require the Torch expert path'
        if getattr(child, '_full_precision_mm', False):
            return 'Host full-precision expert matmul requires the Torch expert path'
    return None


def _rollback_owned_patchers(patchers):
    """Undo a partial resident load without unloading unrelated host models."""
    owned = set(patchers)
    for loaded in list(mm.current_loaded_models):
        if loaded.model not in owned:
            continue
        try:
            loaded.model_unload()
        except BaseException as exc:
            logging.warning('Drama resident load rollback failed: %s', exc)
        finally:
            if loaded in mm.current_loaded_models:
                mm.current_loaded_models.remove(loaded)
    for patcher in patchers:
        # A loader can fail before adding its LoadedModel entry.
        try:
            patcher.detach()
        except BaseException as exc:
            logging.warning('Drama resident patcher detach failed: %s', exc)


class ResidentExperts:
    placement = 'vram'
    cpu_side = None
    cpu_experts_run = 0
    cpu_decode_calls = 0

    def __init__(self, model, requested_backend='auto'):
        if requested_backend not in {'auto', 'torch', 'cuda'}:
            raise ValueError('MoE backend must be auto, torch or cuda')
        self._model = weakref.ref(model)
        self._members = []
        self.requested_backend = requested_backend
        self.backend_calls = {'torch': 0, 'cuda': 0}
        self.last_backend = 'not_run'
        self.selected_backend = 'torch'
        self.fallback_reason = None
        self.sync_free = False
        self.device = None
        self.active = False
        self.closed = False
        self.memory = None
        for layer in model.model.layers:
            module = layer.mlp.experts
            if not getattr(type(module), '_supports_resident_policy', False):
                raise TypeError('Resident placement requires native expert modules with a resident execution policy')
            module._resident_controller = self
            self._members.append(weakref.ref(module))

    def modules(self):
        members = [member() for member in self._members]
        if not members or any(member is None for member in members):
            raise RuntimeError('Resident expert modules are no longer available')
        return members

    def _validate_residency(self, patchers, device):
        for patcher in patchers:
            if getattr(patcher, 'is_dynamic', lambda: False)():
                raise ResidentMemoryError('Full VRAM placement requires a non-dynamic model patcher')
            missing = [name for name, tensor in physical_tensors(patcher.model)
                       if tensor.is_meta or tensor.device != device]
            if missing or getattr(patcher.model, 'model_lowvram', False):
                detail = ', '.join(missing[:4]) or 'host low-VRAM streaming is active'
                raise ResidentMemoryError(f'Full VRAM placement was not achieved: {detail}')

    def _select_backend(self, device):
        reason = None
        if self.requested_backend != 'torch':
            from . import kernels
            for module in self.modules():
                reason = _host_hook_reason(module)
                if reason:
                    break
                if _bank_description(module) is None:
                    reason = 'Resident CUDA expert kernels do not support this weight layout'
                    break
            if reason is None and not kernels.available(device):
                reason = 'NVRTC resident expert kernels are unavailable on this host'
        self.selected_backend = 'cuda' if self.requested_backend != 'torch' and reason is None else 'torch'
        self.fallback_reason = reason

    def load(self, device, patchers, memory_required=0):
        if self.closed:
            raise RuntimeError('Resident expert controller was closed')
        device = torch.device(device)
        if device.type != 'cuda':
            raise ResidentMemoryError('Full VRAM placement requires one CUDA GPU; select RAM placement on this host')
        patchers = list(dict.fromkeys(patchers))
        if not patchers or any(patcher.load_device != device for patcher in patchers):
            raise ResidentMemoryError('Resident model and auxiliary heads must use the same CUDA device')
        if any(getattr(patcher, 'is_dynamic', lambda: False)() for patcher in patchers):
            raise ResidentMemoryError('Full VRAM placement requires non-dynamic model patchers')
        if type(memory_required) not in (int, float) or not 0 <= memory_required < float('inf'):
            raise ValueError('Resident inference memory requirement must be finite and nonnegative')
        reserve = max(RESIDENT_RESERVE, int(memory_required))
        modules = [patcher.model for patcher in patchers]
        weight_bytes = physical_bytes(modules)
        total = mm.get_total_memory(device)
        self.memory = dict(weight_bytes=weight_bytes, runtime_reserve_bytes=reserve, total_device_bytes=total)
        if weight_bytes + reserve > total:
            raise ResidentMemoryError(f'Single-GPU full VRAM placement needs {weight_bytes / 1024**3:.2f} GiB weights '
                f'+ {reserve / 1024**3:.2f} GiB runtime reserve; this GPU has {total / 1024**3:.2f} GiB. '
                'Select RAM placement or a GPU with sufficient VRAM; streaming is not used for this mode.')
        mm.throw_exception_if_processing_interrupted()
        pending = physical_bytes(modules, outside_device=device)
        keep = [loaded for loaded in mm.current_loaded_models if loaded.model in patchers]
        mm.free_memory(pending + reserve, device, keep)
        free = mm.get_free_memory(device)
        self.memory.update(pending_weight_bytes=pending, free_device_bytes=free)
        if pending + reserve > free:
            raise ResidentMemoryError(f'Full VRAM placement needs {pending / 1024**3:.2f} GiB more weights '
                f'+ {reserve / 1024**3:.2f} GiB runtime reserve; only {free / 1024**3:.2f} GiB is free. '
                'Select RAM placement or release GPU memory; the model was not silently streamed.')
        try:
            mm.load_models_gpu(patchers, memory_required=reserve, minimum_memory_required=reserve, force_full_load=True)
            self._validate_residency(patchers, device)
            mm.throw_exception_if_processing_interrupted()
            self.device = device
            self._select_backend(device)
        except BaseException as exc:
            self.sync_free = False
            self.device = None
            _rollback_owned_patchers(patchers)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                raise ResidentMemoryError('Full VRAM allocation failed despite the resident budget check. '
                    'No streaming fallback was used; select RAM placement or release more GPU memory.') from exc
            raise

    @contextlib.contextmanager
    def session(self, device, patchers, cpu_decode=False):
        if self.active:
            if torch.device(device) != self.device:
                raise ResidentMemoryError('An active resident session cannot change CUDA device')
            yield
            return
        self.sync_free = False
        try:
            self.load(device, patchers)
            self.active = True
            self.sync_free = self.selected_backend == 'cuda'
            yield
        finally:
            self.active = False
            self.sync_free = False

    def forward(self, module, x, ids, weights, torch_operation):
        if self.closed or not self.active:
            raise RuntimeError('Resident experts require an active verified generation session')
        if x.device != self.device or not _on_device(module, self.device):
            self.sync_free = False
            raise ResidentMemoryError('Resident expert storage left its verified CUDA device; streaming is not permitted')
        backend = self.selected_backend
        if backend == 'cuda':
            from .offload_accelerated import resident_decode, resident_prefill
            kind, banks, rows, books = _bank_description(module)
            if x.shape[0] == 1:
                out = resident_decode(x, ids[0], weights[0], banks, rows, kind, books)
            else:
                out = resident_prefill(x, ids, weights, banks, rows, kind, books, module.num_experts)
        else:
            out = torch_operation(x, ids, weights)
        self.backend_calls[backend] += 1
        self.last_backend = backend
        return out

    def stats(self):
        return dict(placement='vram', requested_backend=self.requested_backend, actual_backend=self.last_backend,
                    selected_backend=self.selected_backend, fallback_reason=self.fallback_reason,
                    backend_calls=dict(self.backend_calls), cpu_experts_run=0, cpu_decode_calls=0,
                    sync_free=self.sync_free, resident_memory=self.memory)

    def close(self):
        if self.active:
            raise RuntimeError('Cannot close an active resident generation')
        self.sync_free = False
        self.closed = True
        self._members.clear()


class ResidentFP8Experts(ReferenceFP8Experts):
    """Same checkpoint keys/storage; optional resident execution policy."""
    _resident_controller = None
    _supports_resident_policy = True

    def sync_free(self, device):
        control = self._resident_controller
        return control is not None and control.sync_free and control.device == device and _on_device(self, device)

    def forward(self, x, ids, weights):
        if self._resident_controller is None:
            return super().forward(x, ids, weights)
        return self._resident_controller.forward(self, x, ids, weights, super().forward)
