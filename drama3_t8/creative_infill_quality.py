"""Conservative inserted-region signal guard; no Torch, models or I/O.

Caller passes the source/target Takes' already-spoken transcripts, normalized
without named-Scene parsing. Literal colons are words, not character prefixes.
Tags and director notes must not be treated as newly inserted spoken words.
"""
import math
import unicodedata
import numpy as np

BLOCK_SAMPLES=65536


def _alnum(text):
    if not isinstance(text,str):raise ValueError("Spoken text must be a string")
    return "".join(ch for ch in unicodedata.normalize("NFKC",text).casefold() if ch.isalnum())


def needs_new_spoken_alnum(source_spoken,target_spoken):
    """True only when target alnum needs insertion/replacement, not deletion.

    A target subsequence can be obtained using deletions only. This linear-time
    exact test avoids quadratic diff memory/time and conservatively permits
    all delete-only, punctuation, spacing and case-only changes.
    """
    source,target=_alnum(source_spoken),_alnum(target_spoken)
    if not target:return False
    cursor=0
    for ch in source:
        if ch==target[cursor]:
            cursor+=1
            if cursor==len(target):return False
    return True


def analyze_infill_insert(source_spoken,target_spoken,waveform,start_sample,inserted_samples,sample_rate=44100):
    wave=np.asarray(waveform)
    if wave.ndim!=2 or wave.shape[0]!=1:raise ValueError("Expected mono channel-by-samples actual PCM")
    if type(start_sample) is not int or type(inserted_samples) is not int or not 0<=start_sample<=wave.shape[-1] or not 0<=inserted_samples<=wave.shape[-1]-start_sample:
        raise ValueError("Invalid actual inserted PCM range")
    if type(sample_rate) is not int or sample_rate<=0:raise ValueError("Invalid sample rate")
    needs=needs_new_spoken_alnum(source_spoken,target_spoken)
    result={"scope":"actual_inserted_PCM_only","requires_new_spoken_alnum":needs,"inserted_samples":inserted_samples,
        "duration_seconds":inserted_samples/sample_rate,"rms_dbfs":None,"peak_dbfs":None,"peak":None,
        "rms_threshold_dbfs":-80.,"peak_threshold_dbfs":-60.,"rejected":False,"reason":"empty_infill_not_globally_prohibited"}
    if not inserted_samples:
        if needs:result.update(rejected=True,reason="new_spoken_material_but_empty_insert")
        return result
    squares=0.;peak=0.
    for first in range(start_sample,start_sample+inserted_samples,BLOCK_SAMPLES):
        block=wave[:,first:min(first+BLOCK_SAMPLES,start_sample+inserted_samples)]
        if not np.isfinite(block).all():raise ValueError("Inserted PCM is not finite")
        squares+=float(np.sum(np.square(block,dtype=np.float64),dtype=np.float64))
        peak=max(peak,float(np.max(np.abs(block))))
    rms=math.sqrt(squares/inserted_samples)
    rms_db=20*math.log10(max(rms,1e-12));peak_db=20*math.log10(max(peak,1e-12))
    rejected=needs and rms_db < -80. and peak_db < -60.
    result.update(rms_dbfs=rms_db,peak_dbfs=peak_db,peak=peak,rejected=rejected,
        reason="new_spoken_material_but_near_silent_insert" if rejected else "not_rejected_by_conservative_signal_guard")
    return result
