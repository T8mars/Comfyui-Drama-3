"""CPU contracts for real-callback orchestration, not model-quality evidence."""
import copy
import hashlib
import json
import math
import sys
import threading
from types import SimpleNamespace

import pytest

from drama3_t8.performance import (
    BenchmarkCancelled, BenchmarkCase, BenchmarkConfig, BenchmarkOutcome,
    BenchmarkProfileStore, BenchmarkUnavailable, PerformanceRecorder, QualityVerdict,
    ResourceProbe, artifact_identity, binding_key, compare_performance,
    recommend_profile, run_benchmark,
)


class Clock:
    now = 100.
    def __call__(self):return self.now
    def advance(self,n):self.now += n


def binding():
    return dict(machine=dict(id="test-machine",ram_bytes=64*1024**3),
        gpu=dict(uuid="test-gpu",name="testGPU"),torch="2.8.0+cu128",cuda="12.8",
        transformers="4.57.1", source="a"*64, backends=dict(kitchen="0.2.40",moe="v1"),
        models=dict(model=dict(sha256="b"*64),codec=dict(sha256="c"*64),text_head=None,line_head=dict(sha256="d"*64)))


def cases():
    return [BenchmarkCase("short",dict(text="hello",seed=42), ("short",)),
            BenchmarkCase("long-reference",dict(text="A longer referenced script.",seed=42), ("long","reference")),
            BenchmarkCase("dialogue",dict(text="A:one\nB:two",seed=42), ("multi_speaker",))]


def measured_benchmark(*, timings=None, bad_config=None, fallback=None, mutate=False,
                       cancelled=None, error=None, quality_error=False, actual=True):
    clock = Clock();events=[]
    configs=[BenchmarkConfig("graph",dict(moe_backend="cuda",cuda_graph=True)),
             BenchmarkConfig("plain",dict(moe_backend="cuda",cuda_graph=False))]
    def prepare(config,case,run):events.append(("prepare",config.id,case.id,run.requested_load_state))
    def run(config,case,recorder,spec):
        events.append(("run",config.id,case.id,spec.requested_load_state))
        if error:error(config,case,spec)
        with recorder.phase("model.load",cache_state=spec.requested_load_state):clock.advance(1 if spec.requested_load_state=="cold" else .01)
        with recorder.phase("generate"):
            clock.advance((timings or {}).get(config.id,1.))
            recorder.mark_first_codes()
        with recorder.phase("decode"):
            clock.advance(.01);recorder.mark_first_pcm(44100,44100)
        if actual:
            recorder.record_backend("moe",config.settings["moe_backend"],"cuda", "unavailable" if config.id==fallback else None)
            recorder.record_backend("graph",config.settings["cuda_graph"],config.settings["cuda_graph"])
        if mutate:case.payload["seed"]=99
        return BenchmarkOutcome(2.,dict(path="output.wav",case=case.id,config=config.id))
    def quality(case,outcome):
        events.append(("quality",outcome.artifact["config"],case.id))
        if quality_error:raise RuntimeError("ASR failed")
        ok=outcome.artifact["config"]!=bad_config
        return QualityVerdict(ok,dict(words=ok,voice=True))
    report=run_benchmark(binding(),cases(),configs,run,quality,prepare=prepare,
                         probe=lambda:{},clock=clock,sample_interval=0,cancelled=cancelled)
    return report,events


def test_recorder_cold_warm_stage_latency_and_sampled_memory():
    clock=Clock()
    def probe():return dict(process_rss_bytes=int(clock.now*10),cuda_allocated_bytes=int(clock.now*20),system_available_bytes=10000-int(clock.now))
    r=PerformanceRecorder(dict(text="original"),binding(),clock=clock,probe=probe,sample_interval=0)
    with r:
        with r.phase("model.load",cache_state="cold"):clock.advance(2)
        with r.phase("codec.load",cache_state="warm"):clock.advance(1)
        with r.phase("generate"):
            clock.advance(1);assert r.mark_first_codes()
            clock.advance(1);assert not r.mark_first_codes()
        with r.phase("decode"):
            clock.advance(1);assert r.mark_first_pcm(2048,44100)
        r.record_backend("graph",True,False,"normalization weights on CPU")
    d=r.report()
    assert d["elapsed_seconds"]==6 and d["first_codes_seconds"]==4 and d["first_pcm_seconds"]==6
    assert [p["cache_state"] for p in d["phases"][:2]]==["cold","warm"]
    assert d["resources"]["sampled_peaks"]["process_rss_bytes"]==1060
    assert d["resources"]["sampled_peaks"]["cuda_allocated_bytes"]==2120
    assert d["resources"]["minimum_system_available_bytes"]==9894
    assert d["backends"][0]["actual"] is False
    d["request"]["text"]="changed"
    assert r.report()["request"]["text"]=="original"
    assert r.finish()["elapsed_seconds"]==6


def test_missing_first_packet_unavailable_probe_is_not_fabricated_zero():
    r=PerformanceRecorder(probe=lambda: (_ for _ in ()).throw(OSError()),sample_interval=0)
    with r:
        assert not r.mark_first_codes(0) and not r.mark_first_pcm(0,44100)
    d=r.report()
    assert d["first_codes_seconds"] is d["first_pcm_seconds"] is None
    assert d["resources"]["sampled_peaks"]=={} and d["resources"]["probe_errors"]>=2


def test_error_identity_and_no_exception_or_tensor_retention():
    error=RuntimeError("original")
    r=PerformanceRecorder(probe=lambda:{},sample_interval=0)
    with pytest.raises(RuntimeError) as caught:
        with r:
            with r.phase("asr.transcribe"):raise error
    assert caught.value is error and r.report()["error_type"]=="RuntimeError"
    assert r.report()["phases"][0]["status"]=="error"
    with pytest.raises(TypeError):PerformanceRecorder(dict(tensor=object()))


def test_sampler_thread_stops_and_report_is_bounded():
    called=threading.Event()
    def probe():called.set();return dict(process_rss_bytes=10)
    r=PerformanceRecorder(probe=probe,sample_interval=.005,max_spans=2,max_backends=1)
    with r:
        assert called.wait(1)
        for i in range(5):
            with r.phase("generate"):pass
        r.record_backend("moe","cuda","cuda");r.record_backend("moe","cuda","cuda")
    assert not r.thread.is_alive()
    assert len(r.report()["phases"])==2 and r.report()["truncated"]==dict(phases=3,backends=1)


def test_resource_probe_does_not_initialize_cuda(monkeypatch):
    def forbidden(*a):raise AssertionError("No GPU initialization")
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,memory_allocated=forbidden)))
    d=ResourceProbe()()
    assert "cuda_allocated_bytes" not in d


def test_compare_requires_same_model_binding_and_preserves_missing_metrics():
    r=PerformanceRecorder(binding=binding(),probe=lambda:{},sample_interval=0)
    with r:pass
    a=r.report();b=copy.deepcopy(a);b["elapsed_seconds"]=a["elapsed_seconds"]+1
    d=compare_performance(a,b)
    assert d["elapsed_seconds_delta"]==1 and d["first_pcm_seconds_delta"] is None
    b["binding"]["models"]["model"]["sha256"]="f"*64
    with pytest.raises(ValueError):compare_performance(a,b)


def test_benchmark_calls_real_runner_for_all_jobs_then_quality_and_prefers_simple_5percent():
    d,events=measured_benchmark(timings=dict(graph=1.,plain=1.04))
    assert d["status"]=="completed" and d["recommendation"]["config_id"]=="plain"
    runs=[i for i,e in enumerate(events) if e[0]=="run"]
    quality=[i for i,e in enumerate(events) if e[0]=="quality"]
    assert len(runs)==len(quality)==18 and max(runs)<min(quality)
    assert all(r["artifact"]["case"]==r["case_id"] for r in d["results"])
    assert sum("consecutive" in r for r in d["results"])==12


def test_quality_failure_excludes_faster_configuration():
    d,_=measured_benchmark(timings=dict(graph=.1,plain=1),bad_config="graph")
    assert d["recommendation"]["config_id"]=="plain"


def test_actual_fallback_never_becomes_a_claimed_fast_profile():
    d,_=measured_benchmark(timings=dict(graph=.1,plain=1),fallback="graph")
    assert d["recommendation"]["config_id"]=="plain"
    d,_=measured_benchmark(actual=False)
    assert d["recommendation"] is None


def test_benchmark_rejects_short_only_and_cannot_relax_coverage_or_precision():
    with pytest.raises(ValueError,match="coverage"):
        run_benchmark(binding(),[BenchmarkCase("short",{},("short",))],[BenchmarkConfig("x",{})],lambda *a:None,lambda *a:None,required_coverage={"short"})
    for settings in (dict(temperature=.3),dict(seed=7),dict(precision="float16"),dict(cuda_graph=1)):
        with pytest.raises(ValueError):BenchmarkConfig("bad",settings)


def test_callback_input_mutation_and_execution_error_stop_matrix():
    d,events=measured_benchmark(mutate=True)
    assert d["status"]=="error" and d["recommendation"] is None
    assert len([e for e in events if e[0]=="run"])==1 and not any(e[0]=="quality" for e in events)
    def error(*a):raise RuntimeError("unsafe runtime failure")
    d,events=measured_benchmark(error=error)
    assert d["status"]=="error" and len([e for e in events if e[0]=="run"])==1


def test_quality_exception_and_cancellation_cannot_save_profile(tmp_path):
    d,_=measured_benchmark(quality_error=True)
    assert d["status"]=="error" and d["recommendation"] is None
    with pytest.raises(ValueError):BenchmarkProfileStore(tmp_path).save(d)
    d,events=measured_benchmark(cancelled=lambda:True)
    assert d["status"]=="cancelled" and not events and d["recommendation"] is None


def test_unavailable_candidate_skips_without_guessing_runtime_recovery():
    def error(config,*a):
        if config.id=="graph":raise BenchmarkUnavailable()
    d,_=measured_benchmark(error=error)
    assert d["status"]=="completed" and d["recommendation"]["config_id"]=="plain"
    assert [r for r in d["results"] if r["config_id"]=="graph"][0]["status"]=="unavailable"


def test_artifact_identity_full_hash_and_stat_only_binding(tmp_path):
    p=tmp_path/"model.bin";p.write_bytes(b"real file content")
    weak=artifact_identity(p);strong=artifact_identity(p,hash_contents=True)
    assert weak["sha256"] is None and strong["sha256"]==hashlib.sha256(p.read_bytes()).hexdigest()
    b=binding();b["models"]["model"]=weak
    with pytest.raises(ValueError):binding_key(b,require_strong=True)
    with pytest.raises(ValueError):artifact_identity(p,sha256="0"*64,hash_contents=True)


def test_profile_exact_binding_atomic_save_and_corruption_miss(tmp_path,monkeypatch):
    d,_=measured_benchmark();store=BenchmarkProfileStore(tmp_path)
    p=store.save(d);assert store.load(binding())==p
    changed=binding();changed["backends"]["kitchen"]="next"
    assert store.load(changed) is None
    changed=binding();changed["models"]["model"]["sha256"]="f"*64
    assert store.load(changed) is None
    path=tmp_path/(binding_key(binding())+".json");before=path.read_bytes()
    import drama3_t8.performance as module
    monkeypatch.setattr(module.os,"replace",lambda *a:(_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):store.save(d)
    assert path.read_bytes()==before and not list(tmp_path.glob("*.tmp"))
    path.write_text('{"schema":false}',encoding="utf-8")
    assert store.load(binding()) is None


def test_incomplete_or_forged_qualification_cannot_be_recommended():
    d,_=measured_benchmark();broken=copy.deepcopy(d)
    broken["coverage"]=["short"]
    assert recommend_profile(broken) is None
    broken=copy.deepcopy(d);broken["results"][0]["quality"]["checks"]["words"]=False
    assert recommend_profile(broken)["config_id"]=="plain"
    broken=copy.deepcopy(d)
    for r in broken["results"]:r["rtf"]=1e-12
    with pytest.raises(ValueError):recommend_profile(broken)


def test_quality_batch_callback_runs_once_after_timing_and_keeps_shared_time():
    clock=Clock();events=[]
    def run(config,case,recorder,spec):
        events.append("run");clock.advance(1)
        recorder.record_backend("moe","torch","torch")
        return BenchmarkOutcome(2.,dict(path=case.id+".wav"))
    def quality(pairs):
        events.append("batch");assert len(pairs)==9
        clock.advance(4)
        return [QualityVerdict(True,{"words":True}) for _ in pairs]
    d=run_benchmark(binding(),cases(),[BenchmarkConfig("plain",{"moe_backend":"torch"})],run,
        quality_batch_gate=quality,clock=clock,probe=lambda:{},sample_interval=0)
    assert events==["run"]*9+["batch"] and d["quality_batch_seconds"]==4
    assert d["recommendation"] and all(r["quality"]["elapsed_seconds"] is None for r in d["results"])


def test_batch_quality_cardinality_and_backend_mismatch_cannot_qualify():
    clock=Clock()
    def run(config,case,r,spec):
        clock.advance(1);r.record_backend("graph",True,False)
        return BenchmarkOutcome(2.,{"path":"output.wav"})
    args=(binding(),cases(),[BenchmarkConfig("graph",{"cuda_graph":True})],run)
    d=run_benchmark(*args,quality_batch_gate=lambda pairs:[],clock=clock,probe=lambda:{},sample_interval=0)
    assert d["status"]=="error" and d["recommendation"] is None
    d=run_benchmark(*args,quality_gate=lambda *a:QualityVerdict(True,{"words":True}),clock=clock,probe=lambda:{},sample_interval=0)
    assert d["status"]=="completed" and d["recommendation"] is None
