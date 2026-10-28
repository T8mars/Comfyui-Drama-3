"""Explicit, reversible audio preparation and pause cleanup policies."""
import math

import torch
import torch.nn.functional as F


def prepare_reference(wave, sample_rate, *, trim_silence=False, normalize=False):
    """Mono [1,samples], preserving speech and recording every static operation."""
    first,last=0,wave.shape[-1]
    if trim_silence:
        frame=max(1,round(sample_rate*.025))
        hop=max(1,round(sample_rate*.00625))
        padded=F.pad(wave,(frame//2,frame-frame//2))
        rms=padded.unfold(-1,frame,hop).square().mean(-1).sqrt()[0]
        threshold=max(float(rms.max())*.01,1e-6)
        active=(rms>threshold).nonzero().flatten()
        if not len(active):
            raise ValueError("参考录音没有可检测到的人声，请选择包含清晰语音的片段")
        pad=round(sample_rate*.08)
        first=max(0,int(active[0])*hop-pad)
        last=min(last,int(active[-1])*hop+frame+pad)
    result=wave[:,first:last]
    peak=float(result.abs().max())
    gain=min(10.,.9/peak) if normalize and peak>0 else 1.
    result=result*gain if gain!=1. else result
    return result,dict(trim_silence=bool(trim_silence),normalize=bool(normalize),
        trim_start_seconds=first/sample_rate,trim_end_seconds=last/sample_rate,
        original_seconds=wave.shape[-1]/sample_rate,gain=gain)


def shorten_pauses(audio, sample_rate, *, maximum=.75, tail=.15, threshold_db=-50.):
    """Shorten only detected silence, returning a source-to-output edit map."""
    if not all(math.isfinite(v) for v in (maximum,tail,threshold_db)) or maximum<0 or tail<0:
        raise ValueError("Invalid pause cleanup settings")
    if audio.ndim!=2 or not audio.numel() or not torch.isfinite(audio).all():
        raise ValueError("Pause cleanup needs finite [channels,samples] audio")
    window=max(1,round(sample_rate*.01))
    rms=torch.stack([p.mean() for p in audio.square().mean(0).split(window)]).sqrt()
    quiet=(rms<10**(threshold_db/20)).tolist()
    if all(quiet):
        return audio,[]
    last=max(i for i,silent in enumerate(quiet) if not silent)
    limit=min(audio.shape[-1],(last+1)*window+round(tail*sample_rate))
    pieces=[];held=0;started=False
    for index,silent in enumerate(quiet):
        start=index*window
        if start>=limit:break
        if not silent:
            started=True;held=0;end=min(start+window,limit)
        elif started:
            keep=min(window,max(0,round(maximum*sample_rate)-held))
            held+=keep;end=min(start+keep,limit)
        else:continue
        if end>start:
            if pieces and pieces[-1][1]==start:pieces[-1][1]=end
            else:pieces.append([start,end])
    result=torch.cat([audio[:,a:b] for a,b in pieces],dim=-1)
    cursor=0;mapping=[]
    for first,last in pieces:
        mapping.append(dict(source_start=first/sample_rate,source_end=last/sample_rate,
            output_start=cursor/sample_rate,output_end=(cursor+last-first)/sample_rate))
        cursor+=last-first
    return result,mapping
