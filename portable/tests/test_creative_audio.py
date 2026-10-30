import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import weakref
import zipfile

import numpy as np
import pytest
import soundfile as sf

from drama3_t8.creative_audio import (analyze_audio, export_selected, fit_duration,
                                    map_word_times, normalize_audio, resample_audio)
from drama3_t8 import creative_audio as ca


def available_ffmpeg():
    try:
        return bool(ca.ffmpeg_executable())
    except ValueError:
        return False


def tone(rate=16000, seconds=1, amplitude=.2):
    return (amplitude * np.sin(2 * np.pi * 440 * np.arange(round(rate * seconds)) / rate)).astype(np.float32)[None]


def test_finishing_rejects_bad_audio_and_does_not_destroy_source():
    with pytest.raises(ValueError):
        analyze_audio(np.array([np.nan], dtype=np.float32), 16000)
    source = tone()
    original = source.copy()
    result, report = fit_duration(source, 16000, .5, mode="pad")
    assert np.array_equal(source, original)
    assert np.array_equal(result, source)
    assert report["overflow"] and not report["applied"]
    assert report["warning"]


def test_bounded_stretch_preserves_pitch_and_maps_real_timing():
    source = tone(seconds=1.2)
    result, report = fit_duration(source, 16000, 1, mode="stretch")
    assert result.shape == (1, 16000)
    frequencies = np.fft.rfftfreq(16000, d=1 / 16000)
    dominant = frequencies[np.argmax(np.abs(np.fft.rfft(result[0])))]
    assert dominant == pytest.approx(440, abs=3)
    assert report["time_scale"] == pytest.approx(1 / 1.2)
    words = map_word_times([{"text": "hello", "start": .3, "end": .9}], output_start=2, time_scale=report["time_scale"])
    assert words[0]["start"] == pytest.approx(2.25)
    assert words[0]["end"] == pytest.approx(2.75)
    unchanged, refused = fit_duration(source, 16000, .2, mode="stretch")
    assert np.array_equal(unchanged, source) and refused["warning"]


def test_normalize_and_resample_are_explicit_and_peak_bounded():
    source = tone(amplitude=.9)
    stereo = np.repeat(source, 2, axis=0)
    result = resample_audio(stereo, 16000, 48000)
    assert result.shape == (2, 48000)
    normalized, report = normalize_audio(result, 48000, mode="rms", target_db=-18)
    assert normalized.shape == result.shape
    assert report["after"]["rms_dbfs"] == pytest.approx(-18, abs=.02)
    assert np.max(np.abs(normalized)) <= .980001
    assert not np.array_equal(normalized, result)


def test_delivery_keeps_sources_and_creates_synced_stems_and_original_subtitles(tmp_path):
    first, second = tmp_path / "first.wav", tmp_path / "second.wav"
    sf.write(first, tone()[0], 16000, subtype="FLOAT")
    sf.write(second, tone(seconds=.5)[0], 16000, subtype="FLOAT")
    before = first.read_bytes(), second.read_bytes()
    output = tmp_path / "delivery"
    report = export_selected({"id": "project", "revision": 4}, [
        {"line_id": "L01", "role": "猫", "text": "原稿", "readout": "朗读稿", "subtitle": "原稿", "audio_path": str(first), "seed": 42, "target_seconds": 1.2},
        {"line_id": "L02", "role": "人", "subtitle": "第二句", "audio_path": str(second), "seed": 43, "start_seconds": 2},
    ], output, duration_mode="pad", gap_seconds=.15)
    assert (first.read_bytes(), second.read_bytes()) == before
    assert report["lines"][0]["end_seconds"] == pytest.approx(1.2)
    assert report["lines"][1]["start_seconds"] == pytest.approx(2)
    assert report["duration_seconds"] == pytest.approx(2.5)
    master, rate = sf.read(output / "master.wav", always_2d=True)
    stems = [sf.read(output / item["file"], always_2d=True)[0] for item in report["stems"]]
    assert rate == 48000 and master.shape[0] == 120000
    assert np.max(np.abs(master - sum(stems))) < 4e-7
    assert np.max(np.abs(master[57600:96000])) == 0
    assert "原稿" in (output / "subtitles.srt").read_text(encoding="utf-8")
    assert "朗读稿" not in (output / "subtitles.srt").read_text(encoding="utf-8")
    assert json.loads((output / "complete.json").read_text())["status"] == "complete"
    assert sf.info(output / "master.wav").subtype == "PCM_24"
    with pytest.raises(FileExistsError):
        export_selected({}, [{"audio_path": str(first)}], output)


@pytest.mark.skipif(not available_ffmpeg(), reason="No portable or configured FFmpeg available")
def test_lufs_uses_real_ffmpeg_and_returns_measurements():
    result, report = normalize_audio(tone(seconds=3, amplitude=.04), 16000, mode="lufs", target_db=-20)
    assert result.shape[-1] == 48000
    assert report["applied"] and report["loudnorm"]
    assert float(report["loudnorm"]["output_i"]) == pytest.approx(-20, abs=.5)
    assert np.max(np.abs(result)) <= .980001


def test_delivery_pcm_memory_is_bounded_by_block_not_timeline_or_role_count(tmp_path,monkeypatch):
    source=tmp_path/"source.wav"
    sf.write(source,np.repeat(tone(48000,.1).T,2,axis=1),48000,subtype="FLOAT")
    target=tmp_path/"streaming"
    zeros,reads,processed=[],[],[]
    original_zeros=np.zeros;original_like=np.zeros_like
    original_read=sf.SoundFile.read;original_resample=ca.resample_audio
    def bounded_zeros(shape,*args,**kwargs):
        size=int(np.prod(shape));zeros.append(size)
        assert size<=ca.BLOCK_SAMPLES,"Full-timeline/stem PCM allocation forbidden"
        return original_zeros(shape,*args,**kwargs)
    def bounded_like(value,*args,**kwargs):
        assert value.size<=ca.BLOCK_SAMPLES,"Full-timeline stem allocation forbidden"
        return original_like(value,*args,**kwargs)
    def bounded_read(handle,frames=-1,*args,**kwargs):
        if ".work" in str(handle.name):
            reads.append(frames*handle.channels)
            assert 0<frames*handle.channels<=ca.BLOCK_SAMPLES
        return original_read(handle,frames,*args,**kwargs)
    def one_clip(*args,**kwargs):
        assert all(ref() is None for ref in processed),"Prior processed clips retained in RAM"
        result=original_resample(*args,**kwargs);processed.append(weakref.ref(result));return result
    with monkeypatch.context() as patch:
        patch.setattr(np,"zeros",bounded_zeros);patch.setattr(np,"zeros_like",bounded_like)
        patch.setattr(sf.SoundFile,"read",bounded_read);patch.setattr(ca,"resample_audio",one_clip)
        result=export_selected({"project_id":"book"},[
            dict(line_id=f"L{i}",role=f"角色{i}",audio_path=str(source),start_seconds=i*6,target_seconds=2)
            for i in range(3)],target,duration_mode="pad",gap_seconds=0)
    assert max(zeros)<=ca.BLOCK_SAMPLES and max(reads)<=ca.BLOCK_SAMPLES
    assert len(processed)==3 and all(ref() is None for ref in processed)
    assert result["duration_seconds"]==14 and result["channels"]==2
    master=sf.read(target/"master.wav",always_2d=True)[0]
    stems=[sf.read(target/item["file"],always_2d=True)[0] for item in result["stems"]]
    assert master.shape==(14*48000,2) and np.max(np.abs(master-sum(stems)))<4e-7
    assert result["processing"]["aggregate_block_samples"]==65536


def test_above_unity_float_is_not_clipped_before_one_shared_delivery_gain(tmp_path):
    source=tmp_path/"high-peak.wav";samples=np.linspace(-2,2,8192,dtype=np.float32)
    sf.write(source,samples,48000,subtype="FLOAT");before=source.read_bytes()
    result=export_selected({},[dict(line_id="clip",audio_path=str(source),role="speaker",start_seconds=1)],tmp_path/"out",gap_seconds=0)
    line=sf.read(tmp_path/"out"/result["lines"][0]["file"])[0]
    master=sf.read(tmp_path/"out/master.wav")[0]
    assert np.max(np.abs(line-samples*.49))<2.5e-7
    assert np.max(np.abs(line-master[48000:]))<2.5e-7 and np.max(np.abs(line))<=.980001
    assert result["lines"][0]["loudness"]["before"]["peak"]==2
    assert source.read_bytes()==before and result["delivery_gain_db"]==pytest.approx(20*np.log10(.49))


def test_archive_streams_zip64_and_returned_sha_matches_final_archive(tmp_path):
    source=tmp_path/"tone.wav";sf.write(source,tone(seconds=.2)[0],16000,subtype="FLOAT")
    target=tmp_path/"out";archive=tmp_path/"delivery.zip"
    result=export_selected({"project_id":"p"},[dict(audio_path=str(source),line_id="L1",seed=42,
        subtitle="original",references={"speaker":"v1"},fingerprint="frozen")],target,archive_path=archive)
    assert result["archive_file"]==archive.name
    assert result["archive_sha256"]==hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and "complete.json" in zipped.namelist()
        assert not any(name.startswith(".work/") for name in zipped.namelist())
        manifest=json.loads(zipped.read("manifest.json"));complete=json.loads(zipped.read("complete.json"))
        assert complete["manifest_sha256"]==hashlib.sha256(zipped.read("manifest.json")).hexdigest()
        assert manifest["lines"][0]["seed"]==42 and manifest["lines"][0]["references"]=={"speaker":"v1"}
    assert json.loads((target/"delivery-state.json").read_text())["status"]=="complete"
    assert json.loads((target/"archive-receipt.json").read_text())["sha256"]==result["archive_sha256"]


@pytest.mark.parametrize("phase",["processing","writing_audio","packing"])
def test_cancellation_preserves_partial_evidence_without_complete(tmp_path,phase):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.5)[0],16000,subtype="FLOAT")
    before=source.read_bytes();target=tmp_path/"out";archive=tmp_path/"partial.zip"
    class Cancelled(RuntimeError):pass
    error=Cancelled("requested cancellation")
    def check():
        path=target/"delivery-state.json"
        if not path.exists():return
        state=json.loads(path.read_text())
        if state["phase"]==phase and (phase!="packing" or archive.exists()):raise error
    with pytest.raises(Cancelled) as caught:
        export_selected({},[dict(audio_path=str(source))],target,check_cancel=check,archive_path=archive)
    assert caught.value is error and target.is_dir() and not (target/"complete.json").exists()
    assert json.loads((target/"delivery-state.json").read_text())["status"]=="cancelled"
    assert source.read_bytes()==before
    if phase=="packing":
        assert archive.exists()
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zipped:assert "complete.json" not in zipped.namelist()


def test_real_timeline_disk_estimate_rejects_sparse_hour_before_large_writes(tmp_path,monkeypatch):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000,subtype="FLOAT")
    monkeypatch.setattr(ca.shutil,"disk_usage",lambda path:SimpleNamespace(free=200*1024**2))
    target=tmp_path/"out"
    with pytest.raises(OSError,match="真实时轴"):
        export_selected({},[dict(audio_path=str(source),start_seconds=3600)],target)
    assert not list(target.glob("*.wav")) and not (target/"complete.json").exists()
    assert json.loads((target/"delivery-state.json").read_text(encoding="utf-8"))["status"]=="failed"


def test_disk_reserve_is_not_spent_on_audio(tmp_path,monkeypatch):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000)
    monkeypatch.setattr(ca.shutil,"disk_usage",lambda path:SimpleNamespace(free=ca.DISK_RESERVE_BYTES))
    with pytest.raises(OSError,match="保留64MiB"):export_selected({},[dict(audio_path=str(source))],tmp_path/"out")


def test_rf64_boundary_and_real_small_rf64_output_without_gigabyte_fixture(tmp_path,monkeypatch):
    maximum=(ca._RIFF_LIMIT-4096)//(8*3)
    assert ca._container(maximum,8)=="WAV" and ca._container(maximum+1,8)=="RF64"
    source=tmp_path/"source.wav";sf.write(source,tone(48000,.2)[0],48000,subtype="FLOAT")
    monkeypatch.setattr(ca,"_RIFF_LIMIT",4096)
    result=export_selected({},[dict(audio_path=str(source))],tmp_path/"out")
    assert result["master_format"]=="RF64" and sf.info(tmp_path/"out/master.wav").format=="RF64"
    assert result["lines"][0]["format"]=="RF64" and sf.info(tmp_path/"out"/result["lines"][0]["file"]).format=="RF64"
    assert all(item["format"]=="RF64" for item in result["stems"])


def test_reserved_and_case_colliding_names_cannot_overwrite_tracks(tmp_path):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000)
    result=export_selected({},[dict(audio_path=str(source),line_id=value) for value in ("master","role_01","A","a")],tmp_path/"out")
    names=[item["file"].casefold() for item in result["lines"]]
    assert len(set(names))==4 and "master.wav" not in names and "role_01.wav" not in names


def test_existing_archive_and_source_are_never_overwritten(tmp_path):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000)
    archive=tmp_path/"archive.zip";archive.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export_selected({},[dict(audio_path=str(source))],tmp_path/"out",archive_path=archive)
    assert archive.read_bytes()==b"existing" and not (tmp_path/"out").exists()


def test_export_word_mapping_and_provenance_use_actual_output_timeline(tmp_path):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.5)[0],16000)
    result=export_selected({},[dict(audio_path=str(source),start_seconds=2,target_seconds=1,
        word_times=[dict(text="word",start=.1,end=.3)],seed=19,readout="readout",subtitle="original")],tmp_path/"out",duration_mode="pad")
    row=result["lines"][0]
    assert row["word_times"]==[dict(text="word",start=2.1,end=2.3)] and row["end_seconds"]==3
    assert row["seed"]==19 and row["subtitle"]=="original" and row["readout"]=="readout"


@pytest.mark.parametrize("rate",[0,-1,True,16000.,float("nan"),float("inf")])
def test_fit_duration_rejects_invalid_sample_rates_even_without_target(rate):
    with pytest.raises(ValueError):fit_duration(tone(seconds=.01),rate)


@pytest.mark.parametrize("kwargs",[
    dict(target_seconds=float("nan")),dict(target_seconds=float("inf")),dict(target_seconds=True),dict(target_seconds=0),
    dict(min_rate=float("nan")),dict(max_rate=float("inf")),dict(min_rate=1.1),dict(max_rate=.9)])
def test_fit_duration_checks_finite_target_and_rate_boundaries(kwargs):
    with pytest.raises(ValueError):fit_duration(tone(seconds=.01),16000,**kwargs)


@pytest.mark.parametrize("kwargs",[
    dict(source_start=float("nan")),dict(output_start=float("inf")),dict(time_scale=float("inf")),
    dict(time_scale=True),dict(time_scale=0),dict(source_start=-1),dict(output_start=-1)])
def test_word_mapping_rejects_nonfinite_or_invalid_parameters(kwargs):
    with pytest.raises(ValueError):map_word_times([dict(start=0,end=1)],**kwargs)


@pytest.mark.parametrize("words",[[dict(start=0,end=float("inf"))],[dict(start=float("nan"),end=1)],[dict(start=-1,end=0)]])
def test_word_mapping_rejects_nonfinite_word_intervals(words):
    with pytest.raises(ValueError):map_word_times(words)


@pytest.mark.parametrize("selected",[dict(start_seconds=-1),dict(start_seconds=float("nan")),dict(gap_after=-1),dict(gap_after=float("inf"))])
def test_export_rejects_invalid_timeline_before_audio_write(tmp_path,selected):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000)
    with pytest.raises(ValueError):export_selected({},[dict(audio_path=str(source),**selected)],tmp_path/"out")
    assert not list((tmp_path/"out").glob("*.wav")) and not (tmp_path/"out/complete.json").exists()


def test_streaming_rms_includes_padded_silence_in_real_level(tmp_path):
    source=tmp_path/"source.wav";samples=tone(48000,.25,amplitude=.08)
    sf.write(source,samples[0],48000,subtype="FLOAT")
    padded,_=fit_duration(samples,48000,1,mode="pad")
    expected,level=normalize_audio(padded,48000,mode="rms",target_db=-24)
    result=export_selected({},[dict(audio_path=str(source),target_seconds=1)],tmp_path/"out",duration_mode="pad",loudness_mode="rms",target_db=-24)
    delivered=sf.read(tmp_path/"out"/result["lines"][0]["file"])[0]
    assert np.max(np.abs(delivered-expected[0]))<2.5e-7
    assert result["lines"][0]["loudness"]["after"]["rms_dbfs"]==pytest.approx(level["after"]["rms_dbfs"],abs=1e-5)


@pytest.mark.skipif(not available_ffmpeg(),reason="No portable or configured FFmpeg available")
def test_streaming_delivery_lufs_uses_real_ffmpeg_and_retains_float_evidence(tmp_path):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=3,amplitude=.04)[0],16000,subtype="FLOAT")
    result=export_selected({},[dict(audio_path=str(source))],tmp_path/"out",loudness_mode="lufs",target_db=-20)
    level=result["lines"][0]["loudness"]
    assert level["loudnorm"] and float(level["loudnorm"]["output_i"])==pytest.approx(-20,abs=.5)
    assert sf.info(tmp_path/"out"/result["lines"][0]["file"]).frames==3*48000
    assert len(list((tmp_path/"out/.work").glob("*.wav")))==2


def test_loudnorm_cancellation_terminates_only_its_own_child_and_preserves_error(tmp_path,monkeypatch):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000,subtype="FLOAT")
    class Cancelled(RuntimeError):pass
    error=Cancelled("stop owned ffmpeg")
    process=SimpleNamespace(terminated=False,killed=False)
    def terminate():process.terminated=True
    def wait(timeout):
        if process.terminated:return 0
        raise ca.subprocess.TimeoutExpired("owned ffmpeg",timeout)
    process.terminate=terminate;process.wait=wait;process.kill=lambda:setattr(process,"killed",True)
    calls=[0]
    def check():
        calls[0]+=1
        if calls[0]>=3:raise error
    monkeypatch.setattr(ca,"ffmpeg_executable",lambda:"fake-ffmpeg")
    monkeypatch.setattr(ca.subprocess,"Popen",lambda *args,**kwargs:process)
    with pytest.raises(Cancelled) as caught:ca._run_loudnorm_file(source,tmp_path/"out.wav",16000,-20,.98,.1,check)
    assert caught.value is error and process.terminated and not process.killed


def test_ffmpeg_explicit_path_is_first_and_invalid_override_is_not_silently_replaced(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit ffmpeg.exe"
    explicit.touch()
    monkeypatch.setenv("DRAMA_FFMPEG_PATH", str(explicit))
    monkeypatch.setattr(ca, "_probe_ffmpeg", lambda path, signature: True)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: pytest.fail("override ignored")))
    monkeypatch.setattr(ca.shutil, "which", lambda *_: pytest.fail("override ignored"))
    assert ca.ffmpeg_executable() == str(explicit.resolve())
    monkeypatch.setenv("DRAMA_FFMPEG_PATH", str(tmp_path / "missing.exe"))
    with pytest.raises(ValueError, match="DRAMA_FFMPEG_PATH"):
        ca.ffmpeg_executable()


def test_ffmpeg_existing_non_ffmpeg_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_FFMPEG_PATH", sys.executable)
    with pytest.raises(ValueError, match="DRAMA_FFMPEG_PATH"):
        ca.ffmpeg_executable()


def test_ffmpeg_wheel_precedes_path_and_invalid_wheel_falls_back(tmp_path, monkeypatch):
    bundled, system = tmp_path / "bundled.exe", tmp_path / "system.exe"
    bundled.touch(); system.touch()
    monkeypatch.delenv("DRAMA_FFMPEG_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: str(bundled)))
    probes = []
    def probe(path, signature):
        probes.append(path)
        return True
    monkeypatch.setattr(ca, "_probe_ffmpeg", probe)
    monkeypatch.setattr(ca.shutil, "which", lambda *_: pytest.fail("bundled executable lost priority"))
    assert ca.ffmpeg_executable() == str(bundled.resolve())
    assert probes == [str(bundled.resolve())]
    monkeypatch.setattr(ca, "_probe_ffmpeg", lambda path, signature: Path(path) == system.resolve())
    monkeypatch.setattr(ca.shutil, "which", lambda *_: str(system))
    assert ca.ffmpeg_executable() == str(system.resolve())


def test_ffmpeg_unavailable_package_and_path_returns_none_without_installing(tmp_path, monkeypatch):
    monkeypatch.delenv("DRAMA_FFMPEG_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setattr(ca.shutil, "which", lambda *_: None)
    monkeypatch.setattr(ca.subprocess, "run", lambda *_a, **_k: pytest.fail("no executable should be invoked"))
    assert ca.ffmpeg_executable() is None
    with pytest.raises(ValueError, match="imageio-ffmpeg"):
        normalize_audio(tone(seconds=.1), 16000, mode="lufs")


@pytest.mark.skipif(sys.platform != "win32", reason="Pinned Windows wheel relocation contract")
def test_real_imageio_wheel_relocates_and_processes_lufs_without_path(tmp_path, monkeypatch):
    import importlib.util
    fixture = os.environ.get("DRAMA_TEST_IMAGEIO_FFMPEG_FIXTURE")
    if fixture:
        source = Path(fixture) / "imageio_ffmpeg"
    else:
        spec = importlib.util.find_spec("imageio_ffmpeg")
        if spec is None:
            pytest.skip("Set DRAMA_TEST_IMAGEIO_FFMPEG_FIXTURE to an extracted official0.6.0 wheel; no automatic download")
        source = Path(spec.origin).parent
    expected_binary = source / "binaries/ffmpeg-win-x86_64-v7.1.exe"
    if not expected_binary.is_file():
        pytest.skip("The pinned0.6.0 Windows wheel binary is unavailable")
    before, moved = tmp_path / "portable before", tmp_path / "portable relocated"
    shutil.copytree(source, before / "site-packages/imageio_ffmpeg")
    monkeypatch.delenv("DRAMA_FFMPEG_PATH", raising=False)
    monkeypatch.delenv("IMAGEIO_FFMPEG_EXE", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(ca.shutil, "which", lambda *_: None)
    def fresh_import(path):
        monkeypatch.syspath_prepend(str(path / "site-packages"))
        for name in tuple(sys.modules):
            if name == "imageio_ffmpeg" or name.startswith("imageio_ffmpeg."):
                monkeypatch.delitem(sys.modules, name)
    fresh_import(before)
    first = Path(ca.ffmpeg_executable())
    assert first.is_relative_to(before) and first.name == expected_binary.name
    # Move only this test's own checked temporary tree. Fresh import mirrors a
    # relocated portable process rather than retaining the previous package cache.
    assert before.resolve().is_relative_to(tmp_path.resolve()) and moved.resolve().is_relative_to(tmp_path.resolve())
    shutil.move(str(before), str(moved))
    fresh_import(moved)
    actual = Path(ca.ffmpeg_executable())
    assert actual.is_relative_to(moved) and not before.exists()
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == hashlib.sha256(expected_binary.read_bytes()).hexdigest()
    result, report = normalize_audio(tone(seconds=3, amplitude=.04), 16000, mode="lufs", target_db=-20)
    assert result.shape == (1, 48000) and report["loudnorm"]
    assert float(report["loudnorm"]["output_i"]) == pytest.approx(-20, abs=.5)
    stretched, timing = fit_duration(tone(seconds=1.2), 16000, 1, mode="stretch")
    assert stretched.shape == (1, 16000) and timing["applied"]


def test_pack_failure_never_publishes_complete_marker(tmp_path,monkeypatch):
    source=tmp_path/"source.wav";sf.write(source,tone(seconds=.1)[0],16000)
    def pack(target,archive,*args):archive.write_bytes(b"partial owned zip");raise OSError("simulated packing failure")
    monkeypatch.setattr(ca,"_pack_delivery",pack)
    target=tmp_path/"out";archive=tmp_path/"delivery.zip"
    with pytest.raises(OSError,match="packing failure"):
        export_selected({},[dict(audio_path=str(source))],target,archive_path=archive)
    assert not (target/"complete.json").exists() and archive.read_bytes()==b"partial owned zip"
    assert json.loads((target/"delivery-state.json").read_text(encoding="utf-8"))["status"]=="failed"
