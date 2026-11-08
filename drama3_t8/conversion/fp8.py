"""Lossless native FP8 conversion with one destination bank in memory."""
import json
import os
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer
from safetensors_stream import TensorSource, write_tensor_sources


def convert_fp8_streamed(args,checkpoint=None):
    from convert_model import Checkpoint,LAYER_PREFIX,TOWER_PREFIXES,TOWER_RENAMES,AUDIO_HEAD_KEY,source_revision
    checkpoint=checkpoint or Checkpoint(args.src)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(args.src)/'config.json').read_text('utf-8'))
    tokenizer=Tokenizer.from_file(str(Path(args.src)/'tokenizer.json'))
    controls=[tokenizer.token_to_id(value) for value in ('<|audio_separator|>','<|endofturn|>')]
    if any(value is None for value in controls):raise ValueError('Missing scene/stream control tokens')
    ids=[config['eos_token_id'],*range(config['semantic_start_token_id'],config['semantic_end_token_id']+1),*controls]
    config['drama3_output_token_ids']=ids
    sources={}
    def add(key,names,axis=None,rows=None):
        specs=[checkpoint.describe(name) for name in names]
        shape,dtype=specs[0]
        if any(spec[1]!=dtype for spec in specs):raise ValueError('Mixed source dtypes')
        if axis is not None:
            shape=list(shape);shape[axis]=sum(spec[0][axis] for spec in specs);shape=tuple(shape)
        if rows is not None:shape=(len(rows),*shape[1:])
        def load():
            checkpoint.begin_tensor()
            values=[checkpoint.get(name) for name in names]
            value=torch.cat(values,dim=axis) if axis is not None else values[0]
            if rows is not None:value=value.index_select(0,torch.tensor(rows))
            return value.contiguous()
        sources[key]=TensorSource(shape,dtype,load)
    for name in checkpoint.index:
        if name.startswith(LAYER_PREFIX) and '.mlp.experts.' in name:continue
        if name=='lm_head.weight':add(AUDIO_HEAD_KEY,[name],rows=ids);continue
        if name.startswith('model.language_model.'):
            key='model.'+name[len('model.language_model.'):]
            if '.shared_expert.up_proj.' in key or '.shared_expert.down_proj.' in key:continue
            if '.shared_expert.gate_proj.' in key:
                prefix=name.rsplit('.gate_proj.',1)[0];target=key.rsplit('.gate_proj.',1)[0]
                add(target+'.gate_up_proj.weight',[prefix+'.gate_proj.weight',prefix+'.up_proj.weight'],axis=0)
                add(target+'.down_proj.weight',[prefix+'.down_proj.weight'])
            else:add(key,[name])
            continue
        if name=='audio_decoder.codebook_embeddings.weight':add('tower.codebook_embeddings.weight',[name]);continue
        source,target=next((s,t) for s,t in TOWER_PREFIXES if name.startswith(s))
        key=target+name[len(source):]
        for old,new in TOWER_RENAMES:key=key.replace(old,new)
        if '.feed_forward.w3.' in key:continue
        if '.feed_forward.w1.' in key:
            add(key.replace('.w1.','.w13.'),[name,name.replace('.w1.','.w3.')],axis=0)
        else:add(key,[name])
    for layer in range(config['text_config']['num_hidden_layers']):
        for bank,projections in [('gate_up',['gate_proj','up_proj']),('down',['down_proj'])]:
            for field,suffix in [('weight','weight'),('scale','weight_scale_inv')]:
                names=[[f'{LAYER_PREFIX}{layer}.mlp.experts.{e}.{p}.{suffix}' for p in projections]
                    for e in range(config['text_config']['num_experts'])]
                base,dtype=checkpoint.describe(names[0][0]);shape=(len(names),base[0]*len(projections),*base[1:])
                def load(names=names,bank=bank,layer=layer,field=field):
                    checkpoint.begin_tensor()
                    value=torch.stack([torch.cat([checkpoint.get(name) for name in group],dim=0) for group in names])
                    print(f'FP8_BANK layer={layer} bank={bank} field={field}',flush=True)
                    return value
                sources[f'model.layers.{layer}.mlp.experts.{bank}_{field}']=TensorSource(shape,dtype,load)
    metadata={'drama3_format':'fp8','drama3_dense':'bf16','drama3_config':json.dumps(config),
        'drama3_t8_format_version':'2','source_revision':source_revision(args)}
    started=time.time()
    write_tensor_sources(sources,str(out)+'.part',metadata)
    os.replace(str(out)+'.part',out)
    # Both heads must come from the same revision as the backbone.
    for filename,rows in [('drama3-text-head.safetensors',None),('drama3-line-head.safetensors',[tokenizer.encode('\n').ids[0],controls[0]])]:
        checkpoint.begin_tensor()
        head=checkpoint.get('lm_head.weight')
        if rows is not None:head=head.index_select(0,torch.tensor(rows))
        target=out.parent/filename
        source=TensorSource(tuple(head.shape),head.dtype,lambda head=head:head.contiguous())
        head_sources={'weight':source}
        if rows is not None:
            row_ids=torch.tensor(rows,dtype=torch.int64)
            head_sources['ids']=TensorSource(tuple(row_ids.shape),row_ids.dtype,lambda row_ids=row_ids:row_ids)
        write_tensor_sources(head_sources,str(target)+'.part',{'source_revision':metadata['source_revision'],
            **({'drama3_full_text_head':'1'} if rows is None else {'drama3_control_rows':json.dumps(rows)})})
        os.replace(str(target)+'.part',target)
        del source,head
    report={'source_revision':metadata['source_revision'],'bytes':out.stat().st_size,'tensors':len(sources),
        'elapsed_seconds':time.time()-started,'format':'fp8','lossless_expert_bytes':True}
    out.with_suffix('.conversion.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FP8_CONVERSION_COMPLETE',json.dumps(report),flush=True)
