"""Bounded unload observations; no model loads, heap scans or CUDA init."""
import json
import itertools
import logging
import sys
import time
import weakref

import psutil

_FIELDS=('drama','codec','asr','text_head','text_head_patcher','line_head','line_head_patcher')
_HOST_KEYS=('allocated.current','allocated_bytes.current','reserved_bytes.current','segment.current',
            'allocated_bytes.peak','reserved_bytes.peak','num_host_alloc','num_host_free')


def _text(value,limit=1024):
    try:text=str(value)
    except Exception:text='<unprintable>'
    return text if len(text)<=limit else text[:limit-3]+'...'


def _kind(value):
    kind=type(value)
    return _text(kind.__module__+'.'+kind.__qualname__,256)


def _error(error):
    return _text(_kind(error)+': '+_text(error))


def owned_weakrefs(engine):
    """Only weak ownership probes; never keep a retired model alive."""
    result={}
    for field in _FIELDS:
        value=getattr(engine,field,None)
        if value is None:continue
        for suffix,entry in [('',value),*((name,getattr(value,name,None)) for name in
                ('model','patcher','second','second_patcher','experts'))]:
            if entry is None:continue
            name=field+('.'+suffix if suffix else '')
            try:reference=weakref.ref(entry)
            except TypeError:reference=None
            result[name]=(reference,_kind(entry))
    return result


def snapshot(engine, references=()):
    report={'at':time.time(),'engine_fields':{name:getattr(engine,name,None) is not None for name in _FIELDS}}
    try:
        memory=psutil.Process().memory_full_info()
        report['process']={name:int(getattr(memory,name)) for name in ('rss','uss','private','vms') if hasattr(memory,name)}
        physical=psutil.virtual_memory()
        report['system']={'total_bytes':int(physical.total),'available_bytes':int(physical.available)}
    except Exception as error:
        report['process_error']=_error(error)
    torch=sys.modules.get('torch')
    try:
        initialized=bool(torch is not None and torch.cuda.is_initialized())
        report['cuda_initialized']=initialized
        if initialized:
            read=getattr(getattr(torch.cuda,'memory',None),'host_memory_stats',None)
            if read is None:
                report['host_allocator']={'status':'unavailable'}
            else:
                stats=read()
                report['host_allocator']={'status':'measured',**{name:int(stats[name]) for name in _HOST_KEYS
                    if name in stats and isinstance(stats[name],int) and not isinstance(stats[name],bool)}}
        else:
            report['host_allocator']={'status':'CUDA not initialized; not queried'}
    except Exception as error:
        report['host_allocator']={'status':'unavailable','error':_error(error)}
    mm=sys.modules.get('comfy.model_management')
    try:
        managed=getattr(mm,'current_loaded_models',())
        report['managed_models']={'registered_count':len(managed),'samples':[],'sample_limit':16}
        for item in managed[:16]:
            patcher=item.model
            report['managed_models']['samples'].append({'patcher_alive':patcher is not None,
                'patcher_type':_kind(patcher) if patcher is not None else None})
    except Exception as error:
        report['managed_models']={'error':_error(error)}
    probes={}
    entries=references.items() if hasattr(references,'items') else references
    for name,(reference,kind) in itertools.islice(entries,42):
        probes[_text(name,128)]={'type':_text(kind,256),'weakref_supported':reference is not None,
                      'alive':reference() is not None if reference is not None else None}
    report['owned_objects']=probes
    report['owned_probe_limit']=42
    report['owned_probe_truncated']=len(references)>42 if hasattr(references,'__len__') else None
    return report


def begin_unload(engine):
    references=owned_weakrefs(engine)
    return {'version':1,'scope':'unload_observation','before':snapshot(engine,references)},references


def finish_unload(engine,report,references,*,status='ok',error=None):
    report.update(status=status,after=snapshot(engine,references))
    if error is not None:report['error']={'type':_kind(error),'message':_text(error)}
    # Pure JSON only: no strong tensors, models, weakref objects or tracebacks.
    frozen=json.loads(json.dumps(report,allow_nan=False))
    engine.last_unload_diagnostics=frozen
    try:logging.info('Drama 3 unload memory: %s',json.dumps(frozen,ensure_ascii=False,allow_nan=False))
    except Exception:pass
    return frozen
