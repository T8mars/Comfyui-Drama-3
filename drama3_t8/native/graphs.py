"""Optional static CUDA graph replay, with caches advanced outside capture."""
import logging
import torch

_capture_failure=None


class CudaGraphRuntimeError(RuntimeError):
    pass


def ensure_healthy():
    if _capture_failure is not None:
        raise CudaGraphRuntimeError('CUDA graph capture invalidated this process. Restart ComfyUI or the local studio, '
            'then disable CUDA Graph for this configuration. '+_capture_failure)


def _clone(value):
    if isinstance(value,torch.Tensor):return value.clone()
    if isinstance(value,tuple):return tuple(_clone(v) for v in value)
    if isinstance(value,list):return [_clone(v) for v in value]
    if value is None or isinstance(value,(int,float,bool)):return value
    raise TypeError(f'Unsupported CUDA graph input: {type(value).__name__}')


def _copy(buffer,value):
    if isinstance(buffer,torch.Tensor):
        if buffer.shape!=value.shape or buffer.dtype!=value.dtype or buffer.device!=value.device:
            raise ValueError('CUDA graph input tensor contract changed')
        buffer.copy_(value)
    elif isinstance(buffer,(tuple,list)):
        if len(buffer)!=len(value):raise ValueError('CUDA graph input structure changed')
        for a,b in zip(buffer,value):_copy(a,b)
    elif buffer!=value:raise ValueError('CUDA graph constant input changed')


class StaticReplay:
    def __init__(self,device,*,generator=None,on_replay=None):
        self.device=device
        self.generator=generator
        self.on_replay=on_replay
        self.calls=0
        self.graph=None
        self.capture_stream=None
        self.inputs=None
        self.output=None
        self.captures=0
        self.replays=0
        self.fallback_reason=None

    def __call__(self,operation,*inputs):
        global _capture_failure
        ensure_healthy()
        self.calls+=1
        if self.device.type!='cuda' or self.calls<=2 or self.fallback_reason:
            return operation(*inputs)
        if self.graph is not None:
            for buffer,value in zip(self.inputs,inputs):_copy(buffer,value)
            self.graph.replay()
            self.replays+=1
            if self.on_replay:self.on_replay()
            return self.output
        self.inputs=tuple(_clone(value) for value in inputs)
        graph=torch.cuda.CUDAGraph()
        if self.generator is not None:graph.register_generator_state(self.generator)
        state=self.generator.get_state() if self.generator is not None else None
        original_stream=torch.cuda.current_stream(self.device)
        self.capture_stream=torch.cuda.Stream(device=self.device)
        try:
            try:
                with torch.cuda.graph(graph,stream=self.capture_stream):
                    self.output=operation(*self.inputs)
            finally:
                # torch.cuda.graph.__exit__ can raise from capture_end before
                # restoring its stream context. Eager fallback must not keep
                # submitting work to that invalidated capture stream.
                torch.cuda.set_stream(original_stream)
        except BaseException as exc:
            text=str(exc).lower()
            try:graph.pool();completed=True
            except RuntimeError:completed=False
            if not completed:
                # PyTorch capture_end can leave allocator and RNG capture state
                # invalidated. reset()/set_state() cannot safely recover it;
                # never continue and present a corrupt eager result as fallback.
                _capture_failure=str(exc).splitlines()[0]
                self.inputs=None;self.output=None
                raise CudaGraphRuntimeError('CUDA graph capture could not finish safely. Restart ComfyUI or the local '
                    'studio and disable CUDA Graph for this configuration. '+_capture_failure) from exc
            graph.reset()
            if state is not None:self.generator.set_state(state)
            torch.cuda.synchronize(self.device)
            self.inputs=None
            self.output=None
            if (not isinstance(exc,RuntimeError) or isinstance(exc,torch.cuda.OutOfMemoryError)
                    or not any(s in text for s in ('capture','cuda graph','cudagraph'))):
                raise
            self.fallback_reason=str(exc)
            logging.warning('DRAMA CUDA graph unavailable; running this operation eagerly: %s',exc)
            return operation(*inputs)
        self.graph=graph
        self.captures+=1
        # capture_begin queues RNG seed/offset initialization on the capture
        # stream before recording. Order those writes before first replay's
        # RNG offset updates on the caller stream; otherwise offset=0 can race
        # with replay's offset and change a fixed seed's first captured sample.
        original_stream.wait_stream(self.capture_stream)
        graph.replay()
        return self.output


class BackboneGraphs:
    def __init__(self,experts):
        self.runners={}
        self.experts=experts

    def run(self,layer,cache,source,freqs,attention):
        key=id(layer)
        if key not in self.runners:
            def counted():
                self.experts.backend_calls['cuda']+=1
                if layer.layer_type=='linear_attention':
                    delta=layer.linear_attn
                    delta.backend_calls[delta.backend]+=1
            self.runners[key]=StaticReplay(source.device,on_replay=counted)
        def operation(x,rotary):
            return layer(x,attention_mask=None,freqs_cis=rotary,
                optimized_attention=attention,past_key_value=cache)[0]
        return self.runners[key](operation,source,freqs)

    def report(self):
        return dict(captures=sum(r.captures for r in self.runners.values()),
            replays=sum(r.replays for r in self.runners.values()),
            fallback_reasons=list(dict.fromkeys(r.fallback_reason for r in self.runners.values() if r.fallback_reason)))


class TowerGraph:
    def __init__(self,tower,caches,sample,finite_heads,generator,device):
        self.tower=tower
        self.caches=caches
        self.sample=sample
        self.finite_heads=finite_heads
        self.runner=StaticReplay(device,generator=generator)

    def step(self,main,hidden,frame,context):
        for cache in self.caches:cache.prepare(1)
        freqs,mask,attention=context
        if mask is not None:raise ValueError('Tower decode graph requires one unmasked row')
        def operation(token,embedding,previous,rotary):
            self.finite_heads.clear()
            result=self.tower.step(token,embedding,previous,(rotary,None,attention),self.caches,self.sample)
            finite=torch.stack(self.finite_heads).all() if self.finite_heads else torch.ones((),dtype=torch.bool,device=hidden.device)
            return result,finite
        self.tower.external_cache_control=True
        try:
            result,finite=self.runner(operation,main,hidden,frame,freqs)
        finally:
            self.tower.external_cache_control=False
        for cache in self.caches:cache.advance(1)
        return result.clone(),finite
