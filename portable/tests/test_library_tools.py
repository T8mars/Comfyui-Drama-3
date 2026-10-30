"""CPU compatibility metadata and injected Codec callbacks, not model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from drama3_t8.library_tools import describe_voice_versions,execute_reencode
from drama3_t8.reference_cache import codec_identity
from drama3_t8.state import Reference,Session
from drama3_t8.voice_library import VoiceLibrary


def tone(seconds,rate=44100):
    return (.12*np.sin(2*np.pi*300*np.arange(round(seconds*rate))/rate)).astype(np.float32)


class Engine:
    def __init__(self,codec):self.codec_path=codec;self.calls=[];self.unloads=0;self.hook=None
    def load_model(self):raise AssertionError("Main model must never be loaded")
    def unload(self):self.unloads+=1
    def encode_reference(self,wave,rate,transcript,name,**options):
        from drama3_t8.audio_processing import prepare_reference
        self.calls.append(dict(wave=wave.clone(),rate=rate,transcript=transcript,name=name,options=options))
        assert wave.device.type=="cpu" and rate==44100 and wave.shape[0]==1
        actual,processing=prepare_reference(wave,rate,**options)
        if self.hook:self.hook()
        return Reference(transcript,torch.ones((10,4),dtype=torch.int64),name,processing=processing)


class Manager:
    def __init__(self,library,codec,inputs):
        self.voices=library;self.inputs=inputs;self.commits=[];engine=Engine(codec)
        def commit(job,progress):progress();self.commits.append(job)
        self.studio=SimpleNamespace(engine=engine,begin_commit=commit)


@pytest.fixture
def setup(tmp_path):
    library=VoiceLibrary(tmp_path/"library");codec=tmp_path/"codec.safetensors";codec.write_bytes(b"current-codec-fixture")
    source=tmp_path/"original.wav";preview=tmp_path/"preview.wav"
    sf.write(source,tone(3),44100,subtype="FLOAT");sf.write(preview,tone(.3),44100,subtype="FLOAT")
    manager=Manager(library,codec,tmp_path/"inputs")
    def build(*,known=True,multiple=False,preview_distinct=True,source_present=True,interval=True,processing=None):
        identity=codec_identity(codec,library.root/"encoding-cache") if known else None
        data=dict(trim_silence=False,normalize=False)
        if identity is not None:data["codec_identity"]=identity
        if interval:data["source_interval"]=dict(start_seconds=1.5,end_seconds=1.8,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if processing:data.update(processing)
        references=[Reference("准确原台词",torch.full((10,4),3,dtype=torch.int16),"原参考",processing=data)]
        if multiple:references.append(Reference("第二段",torch.full((10,4),4,dtype=torch.int16),processing=data))
        voice=library.create_voice("voice",tags=["普通话"])
        version=library.add_version(voice["id"],Session(tuple(references)),source_path=source if source_present else None,
            preview_path=preview if preview_distinct else None,language="zh",delivery="自然",defaults={"temperature":.7})
        return voice,version
    return manager,build,source,preview


def describe(manager):
    return describe_voice_versions(manager.voices,manager.voices.list_voices(),manager.studio.engine.codec_path,
        manager.voices.root/"encoding-cache")


def encode(manager,version,**payload):
    return execute_reencode(manager,"CPU-reencode",dict(version_id=version["id"],**payload),lambda:None)


def test_status_reads_only_codec_and_cpu_session_never_main_model_and_copies_catalog(setup,monkeypatch,tmp_path):
    manager,build,_,_=setup;voice,version=build();catalog=manager.voices.list_voices();before=deepcopy(catalog)
    main=tmp_path/"main-model.safetensors";main.write_bytes(b"forbidden main model")
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.resolve()==main.resolve():raise AssertionError("Status read main model")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    result=describe_voice_versions(manager.voices,catalog,manager.studio.engine.codec_path,manager.voices.root/"encoding-cache")
    item=result[0]["versions"][0]
    assert catalog==before and item["status"]=="ready" and item["encoding"]["verified"]
    assert item["encoding_cache"]["encoded"] and item["encoding_cache"]["last_result"]=="unknown"
    assert manager.studio.engine.calls==[] and manager.studio.engine.unloads==0


def test_legacy_unknown_is_encoded_but_never_verified_ready(setup):
    manager,build,_,_=setup;build(known=False)
    item=describe(manager)[0]["versions"][0]
    assert item["status"]=="legacy_unknown" and item["encoding"]["encoded"] and not item["encoding"]["verified"]


def test_changed_codec_is_incompatible_and_missing_codec_is_explicit(setup):
    manager,build,_,_=setup;build()
    manager.studio.engine.codec_path.write_bytes(b"different-current-codec")
    assert describe(manager)[0]["versions"][0]["status"]=="incompatible"
    manager.studio.engine.codec_path.unlink()
    item=describe(manager)[0]["versions"][0]
    assert item["status"]=="codec_missing" and not item["encoding"]["verified"]


def test_status_metadata_cache_still_hash_checks_actual_session_and_actual_load(setup,monkeypatch):
    manager,build,_,_=setup;_,version=build();calls=[];original=manager.voices.load_version
    def load(value):calls.append(value);return original(value)
    monkeypatch.setattr(manager.voices,"load_version",load)
    assert describe(manager)[0]["versions"][0]["status"]=="ready"
    assert describe(manager)[0]["versions"][0]["status"]=="ready" and len(calls)==1
    path=manager.voices.session_file(version["id"]);path.write_bytes(path.read_bytes()+b"corrupt")
    assert describe(manager)[0]["versions"][0]["status"]=="corrupt"
    with pytest.raises(ValueError):manager.voices.load_version(version["id"])
    assert manager.studio.engine.calls==[]


def test_single_reencode_uses_preview_exactly_preserves_original_and_interval(setup):
    manager,build,source,preview=setup;voice,version=build()
    source_bytes=manager.voices.source_file(version["id"]).read_bytes();old_session=manager.voices.session_file(version["id"]).read_bytes()
    result=encode(manager,version)
    assert result["voice_id"]==voice["id"] and result["version_id"]!=version["id"] and result["encoding_cache"]["status"]=="miss"
    new=manager.voices.get_version(result["version_id"]);reference=manager.voices.load_version(new["id"]).references[0]
    assert manager.voices.source_file(new["id"]).read_bytes()==source_bytes
    assert manager.voices.source_file(version["id"]).read_bytes()==source_bytes and manager.voices.session_file(version["id"]).read_bytes()==old_session
    saved,rate=sf.read(manager.voices.preview_file(new["id"]),dtype="float32",always_2d=True)
    call=manager.studio.engine.calls[0]
    assert rate==44100 and np.array_equal(saved.T,call["wave"].numpy()) and sf.info(manager.voices.preview_file(new["id"])).subtype=="FLOAT"
    assert reference.processing["source_interval"]["start_seconds"]==1.5 and reference.processing["source_interval"]["end_seconds"]==1.8
    assert reference.processing["reencoded_from"]["version_id"]==version["id"] and result["preview_input_kind"]=="processed_preview"
    assert new["defaults"]==version["defaults"] and manager.commits==["CPU-reencode"]
    assert source.read_bytes()==source_bytes and preview.is_file() and manager.studio.engine.unloads==1


def test_transcript_override_reuses_real_codes_cache_without_old_labels(setup):
    manager,build,_,_=setup;_,version=build()
    first=encode(manager,version,transcript="准确校稿一")
    second=encode(manager,version,transcript="准确校稿二")
    assert first["encoding_cache"]["status"]=="miss" and second["encoding_cache"]["status"]=="hit"
    assert len(manager.studio.engine.calls)==1
    assert manager.voices.load_version(first["version_id"]).references[0].transcript=="准确校稿一"
    assert manager.voices.load_version(second["version_id"]).references[0].transcript=="准确校稿二"
    assert manager.voices.load_version(version["id"]).references[0].transcript=="准确原台词"
    item=next(v for v in describe(manager)[0]["versions"] if v["id"]==second["version_id"])
    assert item["status"]=="ready" and item["encoding_cache"]["last_result"]=="hit" and item["encoding_cache"]["entry_valid"]


def test_missing_original_but_valid_preview_still_creates_honest_new_version(setup):
    manager,build,_,_=setup;_,version=build();manager.voices.source_file(version["id"]).unlink()
    result=encode(manager,version)
    new=manager.voices.get_version(result["version_id"]);ref=manager.voices.load_version(new["id"]).references[0]
    assert new["source_path"] is None and manager.voices.preview_file(new["id"]).is_file()
    assert not result["source_available"] and not result["original_source_preserved"]
    assert ref.processing["source_interval"]["start_seconds"]==1.5
    assert ref.processing["reencoded_from"]["original_source_sha256"]==version["source_sha256"]
    assert ref.processing["reencoded_from"]["original_source_available"] is False


def test_legacy_source_preview_extracts_saved_original_interval_once(setup):
    manager,build,_,_=setup;_,version=build(preview_distinct=False)
    source,rate=sf.read(manager.voices.source_file(version["id"]),dtype="float32")
    result=encode(manager,version)
    assert result["preview_input_kind"]=="original_source_interval"
    assert np.array_equal(manager.studio.engine.calls[0]["wave"].numpy()[0],source[round(1.5*rate):round(1.8*rate)])
    assert manager.voices.source_file(result["version_id"]).read_bytes()==manager.voices.source_file(version["id"]).read_bytes()


def test_multi_reference_version_refuses_ambiguous_mapping_before_codec_call(setup):
    manager,build,_,_=setup;voice,version=build(multiple=True)
    with pytest.raises(ValueError,match="多参考"):encode(manager,version)
    assert len(manager.voices.list_versions(voice["id"]))==1 and manager.studio.engine.calls==[] and not manager.commits


def test_no_usable_preview_or_source_refuses_without_new_version(setup):
    manager,build,_,_=setup;voice,version=build();manager.voices.preview_file(version["id"]).unlink();manager.voices.source_file(version["id"]).unlink()
    with pytest.raises(FileNotFoundError):encode(manager,version)
    assert len(manager.voices.list_versions(voice["id"]))==1 and manager.studio.engine.calls==[]


def test_cache_entry_corruption_does_not_falsify_last_hit_or_native_compatibility(setup):
    manager,build,_,_=setup;_,version=build();result=encode(manager,version)
    key=result["encoding_cache"]["key"];(manager.voices.root/"encoding-cache"/(key+".safetensors")).write_bytes(b"bad cached archive")
    item=next(v for v in describe(manager)[0]["versions"] if v["id"]==result["version_id"])
    assert item["status"]=="ready" and item["encoding_cache"]["last_result"]=="miss" and not item["encoding_cache"]["entry_valid"]


@pytest.mark.parametrize("payload",[dict(transcript=""),dict(transcript=None),dict(trim_silence=1),dict(normalize="true"),dict(name=12),dict(language=3)])
def test_invalid_reencode_configuration_never_encodes_or_commits(setup,payload):
    manager,build,_,_=setup;voice,version=build()
    with pytest.raises(ValueError):encode(manager,version,**payload)
    assert manager.studio.engine.calls==[] and manager.commits==[] and len(manager.voices.list_versions(voice["id"]))==1


def test_owning_voice_cannot_be_changed_by_payload(setup):
    manager,build,_,_=setup;_,version=build()
    with pytest.raises(ValueError,match="归属"):encode(manager,version,voice_id="a"*32)
    assert not manager.studio.engine.calls


def test_codec_file_change_during_encode_refuses_commit(setup):
    manager,build,_,_=setup;voice,version=build()
    manager.studio.engine.hook=lambda:manager.studio.engine.codec_path.write_bytes(b"changed during encode")
    with pytest.raises(ValueError,match="Codec在重编码期间变化"):encode(manager,version)
    assert len(manager.voices.list_versions(voice["id"]))==1 and manager.commits==[]


def test_cancel_after_encode_keeps_old_version_and_no_new_catalog_commit(setup):
    manager,build,_,_=setup;voice,version=build();before=manager.voices.session_file(version["id"]).read_bytes()
    class Cancelled(RuntimeError):pass
    error=Cancelled("cancel owned reencode")
    def progress():
        if manager.studio.engine.calls:raise error
    with pytest.raises(Cancelled) as caught:execute_reencode(manager,"job",dict(version_id=version["id"]),progress)
    assert caught.value is error and len(manager.voices.list_versions(voice["id"]))==1 and manager.commits==[]
    assert manager.voices.session_file(version["id"]).read_bytes()==before


def test_missing_interval_stays_unknown_instead_of_inventing_preview_zero_origin(setup):
    manager,build,_,_=setup;_,version=build(interval=False)
    result=encode(manager,version);ref=manager.voices.load_version(result["version_id"]).references[0]
    assert "source_interval" not in ref.processing and ref.processing["reencoded_from"]["source_interval_status"]=="legacy_unknown"


def test_none_codec_path_reports_missing_not_verified(setup):
    manager,build,_,_=setup;build()
    item=describe_voice_versions(manager.voices,manager.voices.list_voices(),None,manager.voices.root/"cache")[0]["versions"][0]
    assert item["status"]=="codec_missing" and not item["encoding"]["verified"]


def test_corrupt_codec_identity_cache_rebuilds_and_preserves_retired_index(setup):
    manager,build,_,_=setup;build();root=manager.voices.root/"encoding-cache"
    (root/"codec-identities.json").write_text(json.dumps({str(manager.studio.engine.codec_path.resolve()):{"unexpected":1}}),encoding="utf-8")
    item=describe(manager)[0]["versions"][0]
    assert item["status"]=="ready" and list(root.glob("codec-identities.json.retired-*")) and manager.studio.engine.calls==[]


def test_codec_identity_cache_error_fallback_hashes_only_codec_and_stays_explicit(setup,monkeypatch):
    from drama3_t8 import reference_cache
    manager,build,_,_=setup;build()
    def unavailable(*args):raise ValueError("injected metadata-cache error")
    monkeypatch.setattr(reference_cache,"codec_identity",unavailable)
    item=describe(manager)[0]["versions"][0]
    assert item["status"]=="ready" and "ValueError" in item["encoding"]["codec_note"]
    assert item["encoding"]["current_codec"]["sha256"]==hashlib.sha256(manager.studio.engine.codec_path.read_bytes()).hexdigest()
    assert manager.studio.engine.calls==[] and manager.studio.engine.unloads==0


def test_explicit_trim_normalization_is_applied_once_and_preview_matches_codec_pcm(setup):
    manager,build,_,_=setup;_,version=build()
    result=encode(manager,version,trim_silence=True,normalize=True)
    call=manager.studio.engine.calls[0];ref=manager.voices.load_version(result["version_id"]).references[0]
    preview,rate=sf.read(manager.voices.preview_file(result["version_id"]),dtype="float32",always_2d=True)
    assert call["options"]==dict(trim_silence=False,normalize=False)
    assert np.array_equal(preview.T,call["wave"].numpy()) and float(np.max(np.abs(preview)))==pytest.approx(.9,abs=1e-6)
    assert ref.processing["reencode_preprocessing"]["normalize"] is True and ref.processing["reencode_preprocessing"]["trim_silence"] is True
    assert ref.processing["source_interval"]["start_seconds"]==1.5 and rate==44100
