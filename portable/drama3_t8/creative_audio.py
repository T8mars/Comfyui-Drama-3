"""CPU finishing and source-preserving delivery for the local creative desk."""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import io
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


BLOCK_SAMPLES = 65536
DISK_RESERVE_BYTES = 64 * 1024 * 1024
_RIFF_LIMIT = 0xFFFFFFFF


@lru_cache(maxsize=16)
def _probe_ffmpeg(path, signature):
    """Probe one immutable file identity without running a shell or downloading."""
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run([path, "-version"], stdout=output, stderr=subprocess.STDOUT,
                timeout=5, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            output.seek(0)
            first_line = output.read(4096).splitlines()[0:1]
        return result.returncode == 0 and bool(first_line) and first_line[0].startswith(b"ffmpeg version ")
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def _ffmpeg_file(value):
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        return None
    try:
        path = Path(os.path.expandvars(str(value).strip().strip('"'))).expanduser().resolve()
        info = path.stat()
        if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
            return None
        signature = (info.st_size, info.st_mtime_ns, info.st_ino)
        return str(path) if _probe_ffmpeg(str(path), signature) else None
    except (OSError, ValueError):
        return None


def ffmpeg_executable():
    """Find a validated executable: explicit path, portable wheel, then PATH.

    No dependency installation or binary download occurs here. An explicitly
    configured invalid DRAMA_FFMPEG_PATH is an error instead of silently using
    a different FFmpeg. Automatic candidates may fall back to the next source.
    """
    configured = os.environ.get("DRAMA_FFMPEG_PATH")
    if configured is not None:
        selected = _ffmpeg_file(configured)
        if selected is None:
            raise ValueError("DRAMA_FFMPEG_PATH 必须指向可运行的 FFmpeg 文件")
        return selected
    try:
        import imageio_ffmpeg
        selected = _ffmpeg_file(imageio_ffmpeg.get_ffmpeg_exe())
        if selected:
            return selected
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError):
        pass
    return _ffmpeg_file(shutil.which("ffmpeg"))


def _sample_rate(value):
    if type(value) is not int or not 1000 <= value <= 384000:
        raise ValueError("无效采样率")
    return value


def _finite(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(name + "必须为有限数值")
    result = float(value)
    if (minimum is not None and result < minimum) or (maximum is not None and result > maximum):
        raise ValueError(name + "超出有效范围")
    return result


def audio_array(waveform):
    wave = np.asarray(waveform, dtype=np.float32)
    if wave.ndim == 1:
        wave = wave[None, :]
    if wave.ndim != 2 or not 1 <= wave.shape[0] <= 8 or not wave.shape[-1]:
        raise ValueError("音频必须是非空的声道×采样数组")
    if not np.isfinite(wave).all():
        raise ValueError("音频包含非有限采样，不能导出")
    return wave


def read_audio_file(path):
    wave, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio_array(wave.T), int(rate)


def analyze_audio(waveform, sample_rate):
    wave = audio_array(waveform)
    _sample_rate(sample_rate)
    rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
    return {"sample_rate": sample_rate, "channels": int(wave.shape[0]),
            "samples": int(wave.shape[-1]), "duration_seconds": wave.shape[-1] / sample_rate,
            "peak": float(np.max(np.abs(wave))), "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
            "clipped_fraction": float(np.mean(np.abs(wave) >= 1)),
            "silent_fraction": float(np.mean(np.abs(wave) < 0.001))}


def resample_audio(waveform, source_rate, target_rate):
    wave = audio_array(waveform)
    for value in (source_rate, target_rate):
        _sample_rate(value)
    if source_rate == target_rate:
        return wave.copy()
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(wave, target_rate // divisor, source_rate // divisor, axis=-1)
    return result.astype(np.float32)


def _duration_plan(samples, sample_rate, target_seconds, mode, min_rate=.8, max_rate=1.25):
    _sample_rate(sample_rate)
    if mode not in {"report", "pad", "stretch"}:
        raise ValueError("时长方式应为 report、pad 或 stretch")
    min_rate = _finite(min_rate,"最小变速")
    max_rate = _finite(max_rate,"最大变速")
    if not 0 < min_rate <= 1 <= max_rate:
        raise ValueError("无效变速边界")
    source_seconds = samples / sample_rate
    report = {"mode": mode, "source_seconds": source_seconds, "actual_seconds": source_seconds,
              "target_seconds": target_seconds, "applied": False, "time_scale": 1.0,
              "overflow": False, "warning": ""}
    if target_seconds is None:
        return samples,"copy",report
    target_seconds = _finite(target_seconds,"目标时长",minimum=0,maximum=86400)
    if target_seconds <= 0:
        raise ValueError("目标时长必须是有效的正数")
    report.update(target_seconds=target_seconds, overflow=source_seconds > target_seconds + 1 / sample_rate)
    if mode == "report":
        return samples,"copy",report
    count = max(1, round(target_seconds * sample_rate))
    if mode == "pad":
        if count >= samples:
            report.update(applied=count != samples, actual_seconds=count / sample_rate)
            return count,"pad",report
        report["warning"] = "台词超出目标时长，保留完整音频；可改稿、重做或启用变速"
        return samples,"copy",report
    rate = source_seconds / target_seconds
    report["rate"] = rate
    if not min_rate <= rate <= max_rate:
        report["warning"] = f"所需变速 {rate:.3f} 超出允许范围 {min_rate:g}–{max_rate:g}，保留完整原音"
        return samples,"copy",report
    report.update(applied=True,actual_seconds=count/sample_rate,time_scale=count/samples)
    return count,"stretch",report


def fit_duration(waveform, sample_rate, target_seconds=None, *, mode="report", min_rate=0.8, max_rate=1.25):
    """Never chop speech to fit. Stretch is an explicit, bounded post-process."""
    wave = audio_array(waveform)
    count,action,report = _duration_plan(wave.shape[-1],sample_rate,target_seconds,mode,min_rate,max_rate)
    if action == "copy":return wave.copy(),report
    if action == "pad":return np.pad(wave,((0,0),(0,count-wave.shape[-1]))),report
    import librosa
    result = librosa.effects.time_stretch(wave, rate=report["rate"])
    result = librosa.util.fix_length(result, size=count, axis=-1).astype(np.float32)
    report["overflow"] = False
    return result, report


def _container(frames, channels, subtype="PCM_24"):
    """Conservative RIFF bound, including header/metadata space."""
    width = {"PCM_24":3,"FLOAT":4}[subtype]
    return "RF64" if frames*channels*width+4096 > _RIFF_LIMIT else "WAV"


def _loudness_process(command, deadline, check_cancel):
    """Only the owned child is stopped; logs never accumulate in a RAM pipe."""
    with tempfile.TemporaryFile() as errors:
        check_cancel()
        process = subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=errors,
            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            while True:
                check_cancel()
                try:code=process.wait(timeout=.1);break
                except subprocess.TimeoutExpired:
                    if time.monotonic()>deadline:raise TimeoutError("FFmpeg 响度处理超时")
        except BaseException:
            try:process.terminate()
            except OSError:pass  # The child can finish between cancellation and termination.
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:process.kill()
                except OSError:pass
                process.wait(timeout=2)
            raise
        check_cancel()
        errors.seek(0,os.SEEK_END);size=errors.tell();errors.seek(max(0,size-65536))
        stderr = errors.read(65536).decode("utf-8","replace")
    if code:
        raise RuntimeError("FFmpeg 响度处理失败："+stderr[-1500:])
    try:
        result = json.loads(stderr[stderr.rfind("{"):stderr.rfind("}")+1])
    except (ValueError,json.JSONDecodeError) as exc:
        raise RuntimeError("FFmpeg 响度处理缺少有效测量报告："+stderr[-1500:]) from exc
    if not isinstance(result,dict):
        raise RuntimeError("FFmpeg 响度报告格式无效")
    for key in ("input_i","input_lra","input_tp","input_thresh","target_offset"):
        _loudness_number(result.get(key),key,allow_infinite=True)
    return result


def _loudness_number(value, name, *, allow_infinite=False):
    if isinstance(value,bool) or not isinstance(value,(str,numbers.Real)):
        raise RuntimeError("FFmpeg 响度报告字段无效："+name)
    try:result=float(value)
    except (ValueError,TypeError,OverflowError) as exc:
        raise RuntimeError("FFmpeg 响度报告字段无效："+name) from exc
    if math.isnan(result) or (not math.isfinite(result) and not allow_infinite):
        raise RuntimeError("FFmpeg 响度报告字段非有限："+name)
    return result if math.isfinite(result) else None


def _gain_loudness_file(path, gain_db, check_cancel):
    """Scale an owned private FLOAT through a separate bounded stream.

    libsndfile read/write handles cannot safely rewrite every FFmpeg WAV header;
    never mutate either an input source or a currently open reader in place.
    """
    gain=10**(gain_db/20)
    path=Path(path);temporary=path.with_name("."+uuid.uuid4().hex+".gain.wav")
    with sf.SoundFile(str(path)) as reader,sf.SoundFile(str(temporary),"w",samplerate=reader.samplerate,
            channels=reader.channels,subtype="FLOAT",format=_container(len(reader),reader.channels,"FLOAT")) as writer:
        while True:
            check_cancel();block=reader.read(_frames_per_block(reader.channels),dtype="float32",always_2d=True)
            if not len(block):break
            if not np.isfinite(block).all():raise RuntimeError("响度结果包含非有限采样")
            block*=gain
            if not np.isfinite(block).all():raise RuntimeError("响度校正产生非有限采样")
            writer.write(block)
        writer.flush()
    check_cancel();temporary.replace(path)


def _run_loudnorm_file(source, target, sample_rate, target_db, peak_limit, duration, check_cancel):
    """Two-pass loudnorm plus measured final-rate verification and bounded repair.

    Dynamic loudnorm meters before the final resample. Its output_i is not used
    as proof about the saved file. At most three renders are attempted; each
    receives actual first-pass input statistics. The best measured result is
    returned even if the requested loudness is physically peak-constrained.
    """
    binary=ffmpeg_executable()
    if not binary:
        raise ValueError("LUFS 处理需要 FFmpeg；请安装整合包依赖 imageio-ffmpeg 或设置 DRAMA_FFMPEG_PATH")
    source,target=Path(source),Path(target)
    if target.exists():raise FileExistsError(target)
    sample_rate=_sample_rate(sample_rate)
    target_db=_finite(target_db,"LUFS目标",minimum=-60,maximum=-1)
    peak_limit=_finite(peak_limit,"LUFS峰值上限",minimum=1e-12,maximum=1)
    source_info=sf.info(str(source))
    if source_info.samplerate!=sample_rate or not source_info.frames:
        raise ValueError("LUFS 源应为非空的目标采样率音频")
    frames,channels=source_info.frames,source_info.channels
    source_signature=(source.stat().st_size,source.stat().st_mtime_ns,source.stat().st_ino)
    # At peak, raw plus three private renders can coexist; no hidden unbudgeted
    # full-timeline array/file allocation is needed for retries.
    _ensure_disk(target.parent,frames*channels*4*4+16384,"LUFS两遍与有限校正")
    container=_container(frames,channels,"FLOAT")
    desired_tp=20*math.log10(peak_limit)
    render_tp=max(-9.,min(-.5,desired_tp-.3))
    sample_bound=peak_limit*(1-1e-6)  # PCM24 quantization and float safety margin.
    deadline=time.monotonic()+max(60.,frames/sample_rate*12)
    prefix=[binary,"-hide_banner","-nostdin","-nostats","-loglevel","info"]
    def measure(file, goal):
        return _loudness_process(prefix+["-i",str(file),"-af",
            f"loudnorm=I={goal:.10g}:TP={render_tp:.10g}:LRA=7:print_format=json","-f","null","-"],deadline,check_cancel)
    def validate(file):
        info=sf.info(str(file))
        if (info.samplerate,info.frames,info.channels)!=(sample_rate,frames,channels):
            raise RuntimeError("响度处理改变了音频时长、声道或采样率")
        return _file_stats(file,check_cancel)
    report={"schema":"drama-loudness-v2","method":"two-pass-final-rate-verified","target_i":target_db,
        "target_true_peak_dbfs":desired_tp,"render_true_peak_dbfs":render_tp,"tolerance_lu":.5,
        "verification_scope":"Actual saved FLOAT at delivery sample rate after sample/true-peak gain; final PCM24 differs only by quantization",
        "max_render_attempts":3,"attempts":[],"target_achieved":False,"warning":""}
    effective=max(-60.,min(-5.,target_db))
    best=None;best_error=math.inf;best_measure=None;best_stats=None;best_render=None;best_attempt=None
    with tempfile.TemporaryDirectory(prefix=".drama-loudness-",dir=str(target.parent)) as private:
        for attempt in range(3):
            first=measure(source,effective)
            if attempt==0:report["analysis"]=first
            input_i=_loudness_number(first["input_i"],"input_i",allow_infinite=True)
            measured_tp=_loudness_number(first["input_tp"],"input_tp",allow_infinite=True)
            current=Path(private)/f"attempt-{attempt}.wav"
            if input_i is None or measured_tp is None:
                # Silence or a sub-window clip has no defined integrated LUFS.
                # Preserve the real samples, limiting only an excessive peak.
                with sf.SoundFile(str(source)) as reader,sf.SoundFile(str(current),"w",samplerate=sample_rate,
                        channels=channels,subtype="FLOAT",format=container) as writer:
                    while True:
                        check_cancel();block=reader.read(_frames_per_block(channels),dtype="float32",always_2d=True)
                        if not len(block):break
                        writer.write(block)
                stats=validate(current)
                gain=min(0.,20*math.log10(sample_bound/max(stats["peak"],1e-30)))
                if gain:_gain_loudness_file(current,gain,check_cancel)
                best,best_measure,best_stats=current,measure(current,effective),validate(current)
                report["warning"]="音频静音或短于响度测量窗口，综合LUFS不可测；保留原时长并遵守峰值限制"
                report["status"]="unmeasurable"
                break
            lra=max(7.,min(50.,_loudness_number(first["input_lra"],"input_lra")))
            for key in ("input_i","input_lra","input_tp","input_thresh","target_offset"):
                _loudness_number(first[key],key)
            bounds={"input_i":(-99.,0.),"input_lra":(0.,99.),"input_tp":(-99.,99.),
                    "input_thresh":(-99.,0.),"target_offset":(-99.,99.)}
            two_pass=all(low<=float(first[key])<=high for key,(low,high) in bounds.items())
            af=f"loudnorm=I={effective:.10g}:TP={render_tp:.10g}:LRA={lra:.10g}:"
            if two_pass:
                af+=(f"measured_I={first['input_i']}:measured_LRA={first['input_lra']}:measured_TP={first['input_tp']}:"
                    f"measured_thresh={first['input_thresh']}:offset={first['target_offset']}:linear=true:")
            else:
                af+="linear=false:"  # Valid above-unity FLOAT can meter above measured_I's maximum 0.
            af+="print_format=json"
            rendered=_loudness_process(prefix+["-i",str(source),"-af",af,"-ar",str(sample_rate),
                "-map_metadata","-1","-bitexact","-c:a","pcm_f32le","-rf64","always" if container=="RF64" else "never","-n",str(current)],deadline,check_cancel)
            stats=validate(current);actual=measure(current,effective)
            actual_i=_loudness_number(actual["input_i"],"actual_i",allow_infinite=True)
            actual_tp=_loudness_number(actual["input_tp"],"actual_tp",allow_infinite=True)
            desired_gain=target_db-actual_i if actual_i is not None else 0.
            sample_headroom=20*math.log10(sample_bound/max(stats["peak"],1e-30))
            true_headroom=desired_tp-actual_tp-.04 if actual_tp is not None else 0.
            gain=min(desired_gain,sample_headroom,true_headroom)
            # Positive gain uses only measured true/sample-peak headroom. Negative
            # gain may always safely reduce an overly loud intermediate.
            if abs(gain)>1e-6:
                _gain_loudness_file(current,gain,check_cancel)
                stats=validate(current);actual=measure(current,effective)
                actual_i=_loudness_number(actual["input_i"],"actual_i",allow_infinite=True)
                actual_tp=_loudness_number(actual["input_tp"],"actual_tp",allow_infinite=True)
            if stats["peak"]>peak_limit+1e-7:raise RuntimeError("响度校正未能遵守采样峰值限制")
            error=abs(target_db-actual_i) if actual_i is not None else math.inf
            report["attempts"].append({"effective_target_i":effective,"first_pass":first,"render_stats":rendered,
                "actual_measurement":actual,"sample_peak":stats["peak"],"final_linear_gain_db":gain,
                "target_error_lu":error if math.isfinite(error) else None,
                "two_pass_parameters_supplied":two_pass,
                "render_mode_request":"measured-two-pass" if two_pass else "dynamic-source-stats-outside-parameter-range",
                "peak_limited":desired_gain>min(sample_headroom,true_headroom)+.01})
            if best is None or error<best_error:
                old_best=best
                best,best_error,best_measure,best_stats,best_render=current,error,actual,stats,rendered
                best_attempt=attempt
                if old_best is not None:old_best.unlink()  # Only our private render, never source/candidate media.
            else:
                current.unlink()  # Bound temporary disk to best + current + streamed gain + source.
            if error<=.35:break  # Leave margin for final PCM24/meter rounding.
            if actual_i is None:break
            next_target=max(-60.,min(-5.,effective+max(-3.,min(3.,1.5*(target_db-actual_i)))))
            if abs(next_target-effective)<.01:break
            effective=next_target
        check_cancel()
        signature=(source.stat().st_size,source.stat().st_mtime_ns,source.stat().st_ino)
        if signature!=source_signature:raise RuntimeError("LUFS源文件在处理期间发生变化")
        final_i=_loudness_number(best_measure["input_i"],"final_i",allow_infinite=True)
        final_tp=_loudness_number(best_measure["input_tp"],"final_tp",allow_infinite=True)
        peak_ok=best_stats["peak"]<=peak_limit+1e-7 and (final_tp is None or final_tp<=desired_tp+.02)
        achieved=final_i is not None and abs(final_i-target_db)<=.5 and peak_ok
        report.update(target_achieved=achieved,actual_i=final_i,actual_true_peak_dbfs=final_tp,
            actual_sample_peak=best_stats["peak"],output_i=best_measure["input_i"],output_tp=best_measure["input_tp"],
            output_lra=best_measure["input_lra"],output_thresh=best_measure["input_thresh"],
            peak_limit_achieved=peak_ok,normalization_type=best_render.get("normalization_type") if best_render else "copy-unmeasurable",
            measured_after=best_measure,render_attempts=len(report["attempts"]),selected_attempt=best_attempt)
        if not report.get("status"):report["status"]="achieved" if achieved else "target_not_achieved"
        if not achieved and not report["warning"]:
            actual_label=f"{final_i:g}" if final_i is not None else "不可测"
            report["warning"]=f"LUFS目标{target_db:g}，实际{actual_label}；峰值约束或有限校正次数下未达±0.5LU，未宣称达标"
        # Exclusive bounded final copy preserves any pre-existing destination.
        # A cancellation here leaves diagnostic partial bytes, never a report
        # or a delivery complete marker. Sources and candidates are untouched.
        with best.open("rb") as source_stream,target.open("xb") as destination:
            while True:
                check_cancel();block=source_stream.read(1024*1024)
                if not block:break
                destination.write(block)
            destination.flush();os.fsync(destination.fileno())
        check_cancel()
    return report


def normalize_audio(waveform, sample_rate, *, mode="none", target_db=-20.0, peak_limit=0.98, check_cancel=None):
    wave = audio_array(waveform)
    check_cancel = check_cancel or (lambda:None)
    check_cancel()
    if mode not in {"none", "rms", "peak", "lufs"}:
        raise ValueError("无效响度方式")
    if not -60 <= float(target_db) <= -1 or not 0 < float(peak_limit) <= 1:
        raise ValueError("响度目标或峰值上限无效")
    before = analyze_audio(wave, sample_rate)
    info = {"mode": mode, "target_db": float(target_db), "before": before, "applied": False}
    if mode == "none":
        return wave.copy(), info
    if mode == "lufs":
        with tempfile.TemporaryDirectory(prefix="drama-loudness-") as temporary:
            source, target = Path(temporary) / "source.wav", Path(temporary) / "result.wav"
            sf.write(source, wave.T, sample_rate, subtype="FLOAT",format=_container(wave.shape[-1],wave.shape[0],"FLOAT"))
            info["loudnorm"] = _run_loudnorm_file(source,target,sample_rate,target_db,peak_limit,before["duration_seconds"],check_cancel)
            result, result_rate = read_audio_file(target)
            if result_rate != sample_rate:
                raise RuntimeError("响度处理改变了采样率")
        peak = float(np.max(np.abs(result)))
        if peak > peak_limit:
            result *= peak_limit / peak
        info.update(applied=True, after=analyze_audio(result, sample_rate))
        return result, info
    level = before["rms_dbfs"] if mode == "rms" else 20 * math.log10(max(before["peak"], 1e-12))
    gain = min(10 ** ((target_db - level) / 20), peak_limit / max(before["peak"], 1e-12))
    if before["peak"] < 1e-12:
        gain = 1.0
    result = (wave * gain).astype(np.float32)
    info.update(applied=True, gain_db=20 * math.log10(max(gain, 1e-12)), after=analyze_audio(result, sample_rate))
    return result, info


def map_word_times(words, source_start=0.0, output_start=0.0, time_scale=1.0):
    source_start = _finite(source_start,"源时间起点",minimum=0)
    output_start = _finite(output_start,"输出时间起点",minimum=0)
    time_scale = _finite(time_scale,"时间比例",minimum=0)
    if time_scale <= 0:
        raise ValueError("无效时间映射")
    mapped = []
    for word in words:
        start, end = _finite(word["start"],"词起点"), _finite(word["end"],"词终点")
        if not 0 <= start <= end:
            raise ValueError("无效词时间")
        first,last = output_start+(start-source_start)*time_scale,output_start+(end-source_start)*time_scale
        if not math.isfinite(first) or not math.isfinite(last) or not 0<=first<=last:
            raise ValueError("无效映射结果时间")
        mapped.append({**word,"start":first,"end":last})
    return mapped


def _timecode(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def file_digest(path, *, check_cancel=None):
    digest = hashlib.sha256()
    check_cancel = check_cancel or (lambda:None)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            check_cancel()
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name("."+uuid.uuid4().hex+".pending")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _ensure_disk(path, needed, phase):
    available = shutil.disk_usage(path).free
    if available < needed+DISK_RESERVE_BYTES:
        raise OSError(f"交付磁盘空间不足（{phase}）：需要 {needed} 字节并保留64MiB，剩余 {available} 字节")


def _space_plan(target, archive, output_bytes, archive_bytes, phase):
    if archive is not None and target.stat().st_dev==archive.parent.stat().st_dev:
        _ensure_disk(target,output_bytes+archive_bytes,phase)
    else:
        _ensure_disk(target,output_bytes,phase)
        if archive is not None:_ensure_disk(archive.parent,archive_bytes,phase+" ZIP")


def _frames_per_block(channels):
    return max(1,BLOCK_SAMPLES//channels)


def _silence(writer, frames, channels, check_cancel):
    if not frames:return
    count=min(frames,_frames_per_block(channels))
    zero=np.zeros((count,channels),dtype=np.float32)
    while frames:
        check_cancel();size=min(frames,count);writer.write(zero[:size]);frames-=size


def _float_stage(path, wave, frames, sample_rate, check_cancel):
    channels=wave.shape[0]
    _ensure_disk(path.parent,frames*channels*4+4096,"FLOAT暂存")
    with sf.SoundFile(str(path),"w",samplerate=sample_rate,channels=channels,subtype="FLOAT",
                      format=_container(frames,channels,"FLOAT")) as writer:
        for first in range(0,wave.shape[-1],_frames_per_block(channels)):
            check_cancel();writer.write(wave[:,first:first+_frames_per_block(channels)].T)
        _silence(writer,frames-wave.shape[-1],channels,check_cancel)


def _file_stats(path, check_cancel, gain=1.):
    count=squares=clipped=silent=0;peak=0.
    with sf.SoundFile(str(path)) as reader:
        rate,channels=reader.samplerate,reader.channels
        _sample_rate(rate)
        while True:
            check_cancel();block=reader.read(_frames_per_block(channels),dtype="float32",always_2d=True)
            if not len(block):break
            if not np.isfinite(block).all():raise ValueError("暂存音频包含非有限采样")
            block*=gain;absolute=np.abs(block)
            count+=block.size;squares+=float(np.sum(block.astype(np.float64)**2))
            peak=max(peak,float(np.max(absolute)))
            clipped+=int(np.count_nonzero(absolute>=1));silent+=int(np.count_nonzero(absolute<.001))
        if not count:raise ValueError("暂存音频为空")
    return dict(sample_rate=rate,channels=channels,samples=count//channels,duration_seconds=count/channels/rate,
        peak=peak,rms_dbfs=20*math.log10(max(math.sqrt(squares/count),1e-12)),
        clipped_fraction=clipped/count,silent_fraction=silent/count)


def _copy_stage(writer, item, channels, gain, check_cancel):
    frames=0
    with sf.SoundFile(str(item["stage"])) as reader:
        while True:
            check_cancel();block=reader.read(_frames_per_block(channels),dtype="float32",always_2d=True)
            if not len(block):break
            if reader.channels==1 and channels>1:block=np.repeat(block,channels,axis=1)
            if block.shape[1]!=channels or not np.isfinite(block).all():raise ValueError("交付暂存声道或采样无效")
            block*=gain;writer.write(block);frames+=len(block)
    if frames!=item["frames"]:raise ValueError("交付暂存时长在写出期间发生变化")


def _write_track(path, plan, length, channels, sample_rate, delivery_gain, check_cancel, role=None):
    container=_container(length,channels)
    _ensure_disk(path.parent,length*channels*3+4096,"PCM24音轨")
    with sf.SoundFile(str(path),"w",samplerate=sample_rate,channels=channels,subtype="PCM_24",format=container) as writer:
        cursor=0
        for item in plan:
            if role is not None and item["role"]!=role:continue
            _silence(writer,item["start"]-cursor,channels,check_cancel)
            _copy_stage(writer,item,channels,item["gain"]*delivery_gain,check_cancel)
            cursor=item["end"]
        _silence(writer,length-cursor,channels,check_cancel)
    return container


class _HashingWriter:
    """Non-seekable ZIP sink: data descriptors avoid SHA-invalidating rewrites."""
    def __init__(self, stream):self.stream=stream;self.digest=hashlib.sha256();self.position=0
    def write(self, data):
        count=self.stream.write(data)
        self.digest.update(memoryview(data)[:count]);self.position+=count
        return count
    def tell(self):return self.position
    def seek(self,*args):raise io.UnsupportedOperation("streamed archive")
    def flush(self):return self.stream.flush()


def _pack_delivery(target, archive, files, completion, check_cancel):
    # Partial archives are retained on any error/cancellation, never overwritten.
    with archive.open("xb") as raw:
        sink=_HashingWriter(raw)
        with zipfile.ZipFile(sink,"w",compression=zipfile.ZIP_DEFLATED,allowZip64=True) as zipped:
            for path in files:
                check_cancel()
                with path.open("rb") as source,zipped.open(path.name,"w",force_zip64=True) as destination:
                    while True:
                        check_cancel();block=source.read(BLOCK_SAMPLES)
                        if not block:break
                        destination.write(block)
            # Last cancellation point precedes the archive's completion record.
            check_cancel()
            zipped.writestr("complete.json",json.dumps(completion,ensure_ascii=False).encode("utf-8"))
        raw.flush();os.fsync(raw.fileno())
    return sink.digest.hexdigest()


def export_selected(project, selections, output_dir, *, sample_rate=48000, duration_mode="report",
                    loudness_mode="none", target_db=-20, gap_seconds=.15, check_cancel=None, archive_path=None):
    """Stream a new delivery with bounded aggregate PCM and cancellable ZIP64.

    Only one source clip is processed in RAM. FLOAT stage files retain values
    above unity until one shared necessary peak gain precedes ALL PCM24 writes.
    Timeline/stems/silence buffers contain at most BLOCK_SAMPLES scalar samples.
    """
    check_cancel=check_cancel or (lambda:None)
    if type(sample_rate) is not int or sample_rate not in {44100,48000}:
        raise ValueError("交付采样率应为 44100 或 48000")
    gap_seconds=_finite(gap_seconds,"句间停顿",minimum=0,maximum=30)
    target_db=_finite(target_db,"目标响度",minimum=-60,maximum=-1)
    if duration_mode not in {"report","pad","stretch"} or loudness_mode not in {"none","rms","peak","lufs"}:
        raise ValueError("无效时长或响度方式")
    selections=list(selections)
    if not selections:raise ValueError("请选择音频")
    target=Path(output_dir).resolve()
    if target.exists():raise FileExistsError("交付目录已存在，请另存为新版本")
    archive=Path(archive_path).resolve() if archive_path is not None else None
    if archive is not None and (archive.exists() or archive.is_relative_to(target)):
        raise FileExistsError("ZIP必须另存为交付目录外的新文件")
    check_cancel();target.mkdir(parents=True)
    stage="planning"
    def state(status,**extra):
        value=dict(project_id=project.get("project_id",project.get("id")),status=status,phase=stage,**extra)
        _atomic_json(target/"delivery-state.json",value)
        _atomic_json(target/"incomplete.json",value)
    state("writing")
    try:
        if archive is not None:archive.parent.mkdir(parents=True,exist_ok=True)
        roles=list(dict.fromkeys(str(x.get("role") or "旁白") for x in selections))
        used={"master",*(f"role_{i+1:02}" for i in range(len(roles)))}
        plan=[];cursor=0;float_bytes=line_bytes=0
        for index,selected in enumerate(selections):
            check_cancel();source=Path(selected["audio_path"]).resolve(strict=True);info=sf.info(str(source))
            _sample_rate(info.samplerate)
            if not info.frames or not 1<=info.channels<=8:raise ValueError("无效交付源音频")
            resampled=(info.frames*sample_rate+info.samplerate-1)//info.samplerate
            frames,action,timing=_duration_plan(resampled,sample_rate,selected.get("target_seconds"),duration_mode)
            requested=_finite(selected.get("start_seconds",cursor/sample_rate),"片段起点",minimum=0,maximum=86400)
            start=max(cursor,round(requested*sample_rate));end=start+frames
            if end>sample_rate*86400:raise ValueError("单次交付不能超过24小时")
            gap=_finite(selected.get("gap_after",gap_seconds),"句后停顿",minimum=0,maximum=30)
            cursor=end+round(gap*sample_rate)
            name=str(selected.get("line_id",f"L{index+1:02}"))
            safe="".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:80] or f"L{index+1:02}"
            if safe.casefold() in used:safe+="_"+uuid.uuid4().hex[:8]
            used.add(safe.casefold())
            row={key:selected[key] for key in ("line_id","role","text","readout","subtitle","seed","references",
                "candidate_id","sampling","backend","fingerprint") if key in selected}
            row.update(file=safe+".wav",source_file=str(source),start_seconds=start/sample_rate,end_seconds=end/sample_rate,
                duration_seconds=frames/sample_rate,timing=timing)
            if selected.get("word_times") is not None:
                row["word_times"]=map_word_times(selected["word_times"],selected.get("source_start_seconds",0),start/sample_rate,timing["time_scale"])
            stat=source.stat()
            plan.append(dict(source=source,source_signature=(stat.st_size,stat.st_mtime_ns,stat.st_ino),
                source_rate=info.samplerate,source_frames=info.frames,channels=info.channels,frames=frames,action=action,
                role=str(selected.get("role") or "旁白"),start=start,end=end,row=row))
            float_bytes+=frames*info.channels*4+4096;line_bytes+=frames*info.channels*3+4096
        channels=max(x["channels"] for x in plan);length=max(x["end"] for x in plan)
        if any(x["channels"] not in {1,channels} for x in plan):raise ValueError("交付音频的声道数不兼容")
        metadata_bytes=max(1024*1024,len(json.dumps(selections,ensure_ascii=False).encode("utf-8"))*4+len(plan)*8192)
        final_bytes=line_bytes+(length*channels*3+4096)*(len(roles)+1)+metadata_bytes
        archive_bytes=math.ceil(final_bytes*1.01)+metadata_bytes if archive is not None else 0
        staging_bytes=float_bytes*(4 if loudness_mode=="lufs" else 1)
        _space_plan(target,archive,staging_bytes+final_bytes,archive_bytes,"按真实时轴预估")
        disk_plan=dict(final_bytes=final_bytes,staging_bytes=staging_bytes,archive_bytes=archive_bytes,reserve_bytes=DISK_RESERVE_BYTES)
        work=target/".work";work.mkdir();stage="processing"
        state("writing",disk_estimate=disk_plan)
        for index,item in enumerate(plan):
            check_cancel();source=item["source"]
            item["row"]["source_sha256"]=file_digest(source,check_cancel=check_cancel)
            wave,source_rate=read_audio_file(source);check_cancel()
            if source_rate!=item["source_rate"] or wave.shape!=(item["channels"],item["source_frames"]):
                raise ValueError("交付源音频在读取期间发生变化")
            wave=resample_audio(wave,source_rate,sample_rate);check_cancel()
            if item["action"]=="stretch":
                wave,timing=fit_duration(wave,sample_rate,item["row"]["timing"]["target_seconds"],mode="stretch")
                item["row"]["timing"]=timing
            raw=work/f"{index:06}-float.wav"
            _float_stage(raw,wave,item["frames"],sample_rate,check_cancel)
            del wave
            stat=source.stat()
            if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=item["source_signature"]:
                raise ValueError("交付源文件在处理期间发生变化")
            before=_file_stats(raw,check_cancel);gain=1.;processed=raw
            level=dict(mode=loudness_mode,target_db=target_db,before=before,applied=False)
            if loudness_mode=="lufs":
                processed=work/f"{index:06}-lufs.wav"
                _ensure_disk(work,item["frames"]*item["channels"]*4+4096,"LUFS FLOAT暂存")
                level["loudnorm"]=_run_loudnorm_file(raw,processed,sample_rate,target_db,.98,before["duration_seconds"],check_cancel)
                stats=_file_stats(processed,check_cancel)
                if stats["sample_rate"]!=sample_rate or stats["samples"]!=item["frames"] or stats["channels"]!=item["channels"]:
                    raise ValueError("响度处理改变了音频时长、声道或采样率")
                gain=min(1.,.98/max(stats["peak"],1e-12))
                level.update(applied=True,after=_file_stats(processed,check_cancel,gain))
            elif loudness_mode!="none":
                amplitude=before["rms_dbfs"] if loudness_mode=="rms" else 20*math.log10(max(before["peak"],1e-12))
                gain=min(10**((target_db-amplitude)/20),.98/max(before["peak"],1e-12)) if before["peak"]>=1e-12 else 1.
                level.update(applied=True,gain_db=20*math.log10(max(gain,1e-12)),after=_file_stats(raw,check_cancel,gain))
            item.update(stage=processed,gain=gain,peak=(level.get("after") or before)["peak"])
            item["row"]["loudness"]=level
        # Non-overlap is guaranteed by max(cursor, requested start): the master
        # peak is the maximum per-clip peak, including mono channel duplication.
        delivery_gain=min(1.,.98/max(max(x["peak"] for x in plan),1e-12))
        _space_plan(target,archive,final_bytes,archive_bytes,"最终PCM与ZIP")
        stage="writing_audio";state("writing",disk_estimate=disk_plan)
        master_format=_write_track(target/"master.wav",plan,length,channels,sample_rate,delivery_gain,check_cancel)
        stem_paths=[]
        for index,role in enumerate(roles):
            file=target/f"role_{index+1:02}.wav"
            container=_write_track(file,plan,length,channels,sample_rate,delivery_gain,check_cancel,role)
            stem_paths.append(dict(role=role,file=file.name,sha256=file_digest(file,check_cancel=check_cancel),format=container,subtype="PCM_24"))
        for item in plan:
            check_cancel();file=target/item["row"]["file"]
            _ensure_disk(target,item["frames"]*item["channels"]*3+4096,"逐句PCM24")
            container=_container(item["frames"],item["channels"])
            with sf.SoundFile(str(file),"w",samplerate=sample_rate,channels=item["channels"],subtype="PCM_24",format=container) as writer:
                _copy_stage(writer,item,item["channels"],item["gain"]*delivery_gain,check_cancel)
            item["row"].update(sha256=file_digest(file,check_cancel=check_cancel),format=container,subtype="PCM_24",delivery_gain_db=20*math.log10(delivery_gain))
        stage="metadata";state("writing")
        with (target/"subtitles.srt").open("w",encoding="utf-8") as subtitles,(target/"takes.csv").open("w",encoding="utf-8-sig",newline="") as takes:
            columns=("file","line_id","role","subtitle","start_seconds","end_seconds","duration_seconds","seed","candidate_id")
            writer=csv.writer(takes);writer.writerow(columns)
            for index,item in enumerate(plan):
                check_cancel();row=item["row"]
                subtitles.write(f"{index+1}\n{_timecode(row['start_seconds'])} --> {_timecode(row['end_seconds'])}\n{row.get('subtitle',row.get('text',''))}\n\n")
                writer.writerow([row.get(key,"") for key in columns])
        result=dict(schema="drama3-delivery-v1",project_id=project.get("project_id",project.get("id")),project_revision=project.get("revision"),
            sample_rate=sample_rate,channels=channels,duration_seconds=length/sample_rate,delivery_gain_db=20*math.log10(delivery_gain),
            lines=[x["row"] for x in plan],stems=stem_paths,master="master.wav",master_format=master_format,subtype="PCM_24",
            master_sha256=file_digest(target/"master.wav",check_cancel=check_cancel),disk_estimate=disk_plan,
            processing=dict(intermediate_subtype="FLOAT",aggregate_block_samples=BLOCK_SAMPLES,
                memory_scope="One processed source clip plus bounded stream blocks; no full-timeline PCM or per-role arrays",
                source_preserved=True,peak_gain_scope="Same necessary peak gain applied before master, stems and line PCM24"))
        if archive is not None:result["archive_file"]=archive.name
        _atomic_json(target/"manifest.json",result)
        completion=dict(status="complete",manifest_sha256=file_digest(target/"manifest.json",check_cancel=check_cancel))
        check_cancel()
        if archive is not None:
            stage="packing";state("writing")
            _ensure_disk(archive.parent,archive_bytes,"流式ZIP64")
            files=[target/"master.wav",*(target/x["file"] for x in stem_paths),*(target/x["row"]["file"] for x in plan),
                target/"subtitles.srt",target/"takes.csv",target/"manifest.json"]
            result["archive_sha256"]=_pack_delivery(target,archive,files,completion,check_cancel)
            _atomic_json(target/"archive-receipt.json",dict(file=archive.name,sha256=result["archive_sha256"],
                scope="External receipt; excluded from ZIP to avoid a self-referential archive checksum"))
        # No cancellation callback after the completion commit begins.
        stage="complete";state("complete")
        _atomic_json(target/"complete.json",completion)
        return result
    except BaseException as exc:
        status="cancelled" if type(exc).__name__ in {"Cancelled","BenchmarkCancelled","InterruptProcessingException","KeyboardInterrupt"} else "failed"
        # Only this newly created delivery's completion marker is invalidated;
        # source clips, intermediate evidence and partial archives are retained.
        try:(target/"complete.json").unlink(missing_ok=True)
        except OSError:pass
        try:state(status,error_type=type(exc).__name__,message=str(exc)[:1500],artifacts_preserved=True)
        except OSError:pass
        raise
