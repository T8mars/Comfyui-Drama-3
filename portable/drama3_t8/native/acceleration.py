"""Choose an expert backend from actual host capabilities and memory headroom."""
import torch
import psutil
import comfy.model_management as mm
from comfy.cli_args import args


def release_unused_pinned_memory():
    """Release only unoccupied host allocator blocks after an engine unload.

    CUDA empty_cache clears device memory; Torch keeps a separate pinned-host
    pool. On 64 GB machines its old expert-sized blocks otherwise prevent the
    next quantization from loading. Feature-detect the public or older binding.
    """
    if not torch.cuda.is_initialized():return False
    accelerator=getattr(torch,'accelerator',None)
    release=getattr(accelerator,'empty_host_cache',None)
    if release is None:release=getattr(torch._C,'_host_emptyCache',None)
    if release is None:release=getattr(torch._C,'_accelerator_emptyHostCache',None)
    if release is not None:
        release()
        return True
    return False


def resident_bytes(module):
    from .offload import tensor_fields
    total=0
    for p in module.parameters():
        if hasattr(p,'_qdata'):
            tensors=[p._qdata,*tensor_fields(p).values()]
            total+=sum(t.numel()*t.element_size() for t in tensors)
        else:total+=p.numel()*p.element_size()
    return total


def physical_tensors(module):
    """Actual packed storage and scale/codebook fields, including buffers.

    A QuantizedTensor's public dtype/shape describes its logical BF16 matrix,
    not the physical W4/W6/INT8 bytes that must fit on the resident device.
    """
    from .offload import tensor_fields
    for name, value in [*module.named_parameters(), *module.named_buffers()]:
        if hasattr(value, '_qdata'):
            yield name + '._qdata', value._qdata
            for field, tensor in tensor_fields(value).items():
                yield name + '.' + field, tensor
        else:
            yield name, value


def physical_bytes(modules, *, outside_device=None):
    """Count each physical tensor once; optional device filters pending copies."""
    seen = set()
    total = 0
    for module in modules:
        for _, tensor in physical_tensors(module):
            if outside_device is not None and tensor.device == outside_device:
                continue
            identity = (str(tensor.device), tensor.data_ptr() if not tensor.is_meta else id(tensor),
                        tensor.numel(), tensor.element_size())
            if identity not in seen:
                seen.add(identity)
                total += tensor.numel() * tensor.element_size()
    return total


def _legacy_pin_admission(model):
    """Legacy copies whole banks; incremental CUDA ownership cannot budget these copies."""
    if mm.DISABLE_PINNED_MEMORY:
        return dict(pin_memory=False, fallback_reason='The ComfyUI host disabled pinned memory')
    size=sum(resident_bytes(layer.mlp.experts) for layer in model.model.layers)
    required=size*1.02+1024**3
    try:
        available=int(psutil.virtual_memory().available)
        if available<0:raise ValueError('Negative available RAM')
    except Exception as exc:
        return dict(pin_memory=False,physical_bytes=size,required_bytes=required,
                    fallback_reason=f'Cannot verify legacy pinned memory headroom: {type(exc).__name__}: {exc}')
    accepted=available>=required
    return dict(pin_memory=accepted,physical_bytes=size,available_bytes=available,required_bytes=required,
                fallback_reason=None if accepted else 'Insufficient RAM for a complete legacy pinned expert copy')


def expert_store(model,cache_bytes,requested,*,ownership=None):
    if requested not in {'auto','torch','cuda'}:
        raise ValueError('MoE backend must be auto, torch or cuda')
    reason=None
    admission=dict(policy='legacy_full_expert')
    if requested!='torch':
        device=mm.get_torch_device()
        size=sum(resident_bytes(layer.mlp.experts) for layer in model.model.layers)
        available=psutil.virtual_memory().available
        from . import checkpoint_ownership as owned
        verified=owned.initial_admission(ownership,model)
        required=size*1.02+1024**3
        if verified is not None:
            admission=verified
            available=verified['available_bytes']
            required=verified['required_bytes']
        else:
            admission.update(available_bytes=int(available),required_bytes=required)
            refusal=owned.selection_refusal(ownership)
            if refusal is not None:admission['ownership_refusal']=refusal
        if device.type!='cuda':reason='CUDA GPU is unavailable'
        elif mm.DISABLE_PINNED_MEMORY and not getattr(args,'drama_standalone_engine',False):
            reason='The ComfyUI host disabled pinned memory'
        elif available<required:
            reason=f'Pinned expert memory policy {admission["policy"]} needs {required/1024**3:.2f} GiB; available RAM is {available/1024**3:.2f} GiB'
        else:
            from . import kernels
            if kernels.available(device):
                from .offload_accelerated import ExpertStore
                store=ExpertStore(cache_bytes,pin_memory=True)
                store.requested_backend=requested
                admission['selected_accelerated_store']=True
                store.admission=admission
                if verified is not None:
                    try:owned.bind_store(ownership,model,store)
                    except owned.OwnershipUnavailable:
                        store.close()
                        raise
                return store
            reason='NVRTC expert kernels are unavailable on this host'
    from .offload import ExpertStore
    legacy_pin=_legacy_pin_admission(model)
    store=ExpertStore(cache_bytes,pin_memory=legacy_pin['pin_memory'])
    store.requested_backend=requested
    store.fallback_reason=reason
    store.admission=dict(admission,selected_accelerated_store=False,legacy_pin_admission=legacy_pin)
    if admission['policy']=='verified_pread_incremental':
        store.admission.update(policy='verified_pread_not_selected',candidate_policy=admission['policy'])
    return store
