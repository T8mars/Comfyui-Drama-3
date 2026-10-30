"""Real isolated host/workspace/quality transactions, no Torch/model import.

Only engine inference/ASR are injected. The native archive format and every
workspace commit/quality/retry rule are the real production implementation.
Queue/Comfy ProgressBar source bodies execute without their Torch import trees.
"""
import ast
from contextlib import contextmanager, nullcontext
import copy
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import traceback
from types import ModuleType, SimpleNamespace

import pytest

from drama3_t8.creative_studio import CreativeStudio
from drama3_t8.creative_workspace import CreativeWorkspace
from drama3_t8.creative_progress import BatchProgress
from test_creative_quality import media, review

ROOT=Path(__file__).resolve().parents[1]


class Cancelled(RuntimeError):pass
class InterruptProcessingException(RuntimeError):pass


class Sink:
    def __init__(self):self.values=[];self.checks=0;self.error=None
    def __call__(self,*args):
        if self.error is not None:raise self.error
        if args:self.values.append(tuple(args))
        else:self.checks+=1


class Engine:
    def __init__(self):self.drama=None;self.depth=0;self.loads=0
    def load_model(self):
        if self.drama is None:self.loads+=1;self.drama=object()
    @contextmanager
    def generation_session(self):
        self.load_model();self.depth+=1
        try:yield
        finally:self.depth-=1
    def unload(self):self.drama=None


class Manager(CreativeStudio):
    def __init__(self,root,decisions=None):
        self.workspace=CreativeWorkspace(root/"workspace")
        self.reports=root/"reports";self.reports.mkdir()
        self.events=[];self.decisions=list(decisions or [True]);self.generation_error=None;self.quality_error=None
        self.after_frame=None;self.calls=0;self.options_seen=[]
        self.studio=SimpleNamespace(engine=Engine(),begin_commit=lambda job,progress:progress())
    def _options(self,payload,**kwargs):
        return dict(mode=payload.get("generation_mode","per_line"),takes_per_line=payload.get("takes_per_line",1),
            base_seed=payload.get("base_seed",42),sampling=payload.get("sampling",{}),model_identity={"test":"frozen CPU identity"})
    def _filter_options(self,pid,payload,options):return options
    def performance_binding(self,*args):return {}
    def model_identity(self,**kwargs):return {"test":"frozen CPU identity"}
    def _generate(self,job,directory,progress,parent_recorder=None):
        assert self.studio.engine.depth==1
        self.calls+=1;self.options_seen.append(copy.deepcopy(job));self.events.append(("generate",job.get("unit_id",tuple(job["line_ids"])),job["take_index"]))
        progress(90,100);progress(0,100);progress()
        if self.after_frame:self.after_frame(job,progress)
        if self.generation_error:raise self.generation_error
        return media(job,directory)
    def execute_quality(self,job_id,payload,progress):
        assert self.studio.engine.depth==0
        self.studio.engine.unload();progress()
        if self.quality_error:raise self.quality_error
        project=self.workspace.get_project(payload["project_id"])
        ids=payload.get("candidate_ids") or [key for key,candidate in project["candidates"].items() if candidate["status"]=="completed"]
        self.events.append(("asr",tuple(ids)))
        decision=self.decisions.pop(0) if self.decisions else True
        answers={key:review(decision(project["candidates"][key]) if callable(decision) else decision) for key in ids}
        progress()  # A real batched result has not returned yet: no fake item count.
        result=self.workspace.run_quality(payload["project_id"],lambda candidate:answers[candidate["candidate_id"]],candidate_ids=ids)
        return dict(checked=len(ids),reviews=answers,project_revision=result["project"]["revision"],errors=result["errors"])


def project(manager,count=3):
    return manager.workspace.create_project("CPU real progress",[
        dict(line_id=f"L{index}",role=("猫" if index%2 else "人"),text=f"第{index}句准确保留") for index in range(count)])


def run(manager,document,sink,**settings):
    return manager.execute_batch("cpu-job",dict(project_id=document["project_id"],**settings),sink)


def assert_monotonic(values):
    fractions=[value/total for value,total in values]
    assert fractions==sorted(fractions)
    assert all(type(value)is int and total==10000 and value<total for value,total in values)


def test_real_dependency_blocked_plan_freezes_denominator_and_commits_all_18(tmp_path):
    manager=Manager(tmp_path);document=project(manager,6);sink=Sink()
    options=manager._options(dict(generation_mode="per_line_context",takes_per_line=3))
    initial=manager.workspace.plan_generation(document["project_id"],**options)
    assert initial["blocked"]>0 and initial["pending"]<18
    result=run(manager,document,sink,generation_mode="per_line_context",takes_per_line=3,quality_after=True)
    assert len(result["generated"])==18 and manager.calls==18
    assert_monotonic(sink.values)
    assert next(value for value,total in sink.values if value)>0
    assert next(value for value,total in sink.values if value)==400  # .9 * 8000 / 18, not 9000.
    assert sink.values[-1]==(9900,10000)
    assert len({(job["unit_id"],job["take_index"]) for job in manager.options_seen})==18
    assert len([event for event in manager.events if event[0]=="asr"])==1


def test_real_reuse_only_quality_runs_no_engine_load_and_stays_in_quality_range(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2)
    initial=run(manager,document,Sink())
    assert len(initial["generated"])==2
    manager.studio.engine.unload();loads=manager.studio.engine.loads;calls=manager.calls
    sink=Sink();result=run(manager,document,sink,quality_after=True)
    assert result["generated"]==[] and manager.calls==calls and manager.studio.engine.loads==loads
    assert all(value>=8000 for value,total in sink.values)
    assert result["quality"]["checked"]==2
    assert_monotonic(sink.values)


def test_blocked_slot_restored_to_real_existing_candidate_is_settled_as_reused(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2)
    options=manager._options(dict(generation_mode="per_line_context"))
    old=manager.workspace.run_generation(document["project_id"],media,**options)["project"]
    first=old["candidates"][old["lines"][0]["selected_candidate_id"]]
    later_id=old["lines"][1]["selected_candidate_id"]
    source_path=Path(first["audio_path"]);saved=source_path.read_bytes();source_path.unlink()
    plan=manager.workspace.plan_generation(document["project_id"],**options)
    assert [job["status"] for job in plan["jobs"]]==["pending","blocked"]
    # A real file rehydration restores the user's earlier explicit selection.
    # Workspace then reuses L2's intact historical-context match, not a fake plan.
    manager.after_frame=lambda job,progress:source_path.write_bytes(saved)
    sink=Sink();result=run(manager,document,sink,generation_mode="per_line_context")
    assert len(result["generated"])==1 and manager.calls==1
    final=manager.workspace.get_project(document["project_id"])
    assert final["lines"][1]["selected_candidate_id"]==later_id
    assert final["candidates"][later_id]==old["candidates"][later_id]
    assert_monotonic(sink.values)


def test_retrying_historical_failure_counts_only_this_real_workspace_run(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2)
    options=manager._options(dict(generation_mode="per_line_context"))
    def fail_second(job,directory):
        if job["line_ids"]==["L1"]:raise ValueError("old failed attempt")
        return media(job,directory)
    old=manager.workspace.run_generation(document["project_id"],fail_second,continue_on_error=True,**options)
    assert len(old["failed"])==1 and len(old["generated"])==1
    sink=Sink();result=run(manager,document,sink,generation_mode="per_line_context")
    assert len(result["generated"])==1 and manager.calls==1
    assert manager.workspace.get_project(document["project_id"])["candidates"][old["failed"][0]]["status"]=="failed"
    assert_monotonic(sink.values)


def test_two_real_retry_rounds_get_scoped_generation_and_shared_quality(tmp_path):
    manager=Manager(tmp_path,[False,False,True]);document=project(manager,2);sink=Sink()
    result=run(manager,document,sink,max_quality_retries=2)
    assert len(result["quality"]["rounds"])==2 and manager.calls==6
    assert [len(event[1]) for event in manager.events if event[0]=="asr"]==[2,2,2]
    assert len(result["quality"]["suggested_candidates"])==2
    assert result["quality"]["selection_changed"] is False
    assert_monotonic(sink.values)
    assert any(7000<value<8400 for value,total in sink.values)
    assert any(8400<value<9800 for value,total in sink.values)


def test_no_eligible_retry_explicitly_skips_both_reserved_rounds(tmp_path,monkeypatch):
    from drama3_t8.creative_progress import RetryProgress
    receipts=[];original=RetryProgress.finish
    def finish(scope):
        original(scope);receipts.append(list(scope.skipped_rounds))
    monkeypatch.setattr(RetryProgress,"finish",finish)
    manager=Manager(tmp_path,[True]);document=project(manager,2);sink=Sink()
    result=run(manager,document,sink,max_quality_retries=2)
    assert result["quality"]["rounds"]==[] and manager.calls==2
    assert receipts==[[1,2]]
    assert_monotonic(sink.values)


def test_existing_passing_alternative_suggests_without_retake_or_selection(tmp_path):
    manager=Manager(tmp_path,[lambda candidate:candidate["take_index"]==2]);document=project(manager,1);sink=Sink()
    result=run(manager,document,sink,takes_per_line=2,max_quality_retries=2)
    assert manager.calls==2 and result["quality"]["rounds"]==[]
    assert result["quality"]["suggested_candidates"][0]["origin"]=="existing"
    assert result["quality"]["selection_changed"] is False
    assert_monotonic(sink.values)


@pytest.mark.parametrize("error",[MemoryError("retain memory guard"),Cancelled("user cancellation"),OSError("source IO failure")])
def test_real_workspace_generation_error_preserves_identity_and_never_completes(tmp_path,error):
    manager=Manager(tmp_path);document=project(manager,2);sink=Sink();manager.generation_error=error
    with pytest.raises(type(error)) as caught:run(manager,document,sink,continue_on_error=False)
    assert caught.value is error
    assert manager.workspace.get_project(document["project_id"])["candidates"]
    assert max(value for value,total in sink.values)<9800
    assert_monotonic(sink.values)


def test_cancel_during_real_quality_keeps_generation_progress_and_never_commit(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2);sink=Sink();failure=Cancelled("during ASR")
    manager.quality_error=failure
    with pytest.raises(Cancelled) as caught:run(manager,document,sink,quality_after=True,max_quality_retries=2)
    assert caught.value is failure
    assert max(value for value,total in sink.values)==6000
    assert_monotonic(sink.values)


def test_workspace_rejects_bad_real_media_before_the_unit_completes(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2);sink=Sink();generate=manager._generate
    def corrupt(*args,**kwargs):
        result=generate(*args,**kwargs)
        Path(result["session_path"]).write_bytes(b"invalid native State archive")
        return result
    manager._generate=corrupt
    with pytest.raises(ValueError):run(manager,document,sink,continue_on_error=False)
    assert max(value for value,total in sink.values)<4900
    assert manager.workspace.get_project(document["project_id"])["candidates"]
    assert_monotonic(sink.values)


def test_continue_on_error_counts_committed_failed_attempt_but_not_success(tmp_path):
    manager=Manager(tmp_path);document=project(manager,2);sink=Sink()
    def fail_first(job,progress):
        if manager.calls==1:raise ValueError("controlled structural output error")
    manager.after_frame=fail_first
    with pytest.raises(ValueError,match="partial"):run(manager,document,sink,continue_on_error=True)
    statuses={candidate["status"] for candidate in manager.workspace.get_project(document["project_id"])["candidates"].values()}
    assert statuses=={"failed","completed"}
    assert max(value for value,total in sink.values)==9800
    assert_monotonic(sink.values)


def test_bounded_structure_failures_use_scoped_rounds_and_skip_absent_asr(tmp_path,monkeypatch):
    manager=Manager(tmp_path,[False]);document=project(manager,2);sink=Sink()
    fake_engine=ModuleType("drama3_t8.engine")
    class SpeechGenerationError(RuntimeError):pass
    fake_engine.SpeechGenerationError=SpeechGenerationError
    monkeypatch.setitem(sys.modules,"drama3_t8.engine",fake_engine)
    def fail_retries(job,progress):
        if manager.calls>2:raise SpeechGenerationError("bounded model structure failure")
    manager.after_frame=fail_retries
    result=run(manager,document,sink,max_quality_retries=2)
    assert result["quality"]["structure_failures"]==4
    assert result["quality"]["status"]=="partial_structure_failure"
    assert manager.calls==6 and len([event for event in manager.events if event[0]=="asr"])==1
    assert result["quality"]["suggested_candidates"]==[]
    assert_monotonic(sink.values)


def extract(path,cls,name,namespace):
    tree=ast.parse(path.read_text(encoding="utf-8"))
    if cls:
        tree=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==cls)
    selected=next(node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),str(path),"exec"),namespace)
    return namespace[name]


class Clock:
    def __init__(self):self.tick=0
    def monotonic(self):self.tick+=.5;return self.tick
    def time(self):return time.time()
    perf_counter=monotonic


def queue_host(tmp_path,manager,*,fail_after=None):
    clock=Clock()
    namespace=dict(math=math,time=clock,json=json,traceback=traceback,_error_text=str,
        Cancelled=Cancelled,torch=SimpleNamespace(inference_mode=nullcontext),
        mm=SimpleNamespace(InterruptProcessingException=InterruptProcessingException,interrupt_current_processing=lambda value:None))
    method=extract(ROOT/"drama3_t8/studio.py","Studio","_run_queue_item",namespace)
    class Host:
        _run_queue_item=method
        def __init__(self):
            self.data=tmp_path/"queue";self.data.mkdir();(self.data/"logs").mkdir()
            self.database=self.data/"jobs.sqlite3";self.cancel_events={"job":threading.Event()}
            self._activity_lock=threading.RLock();self._queue_error=None;self._job_records={"job":{}}
            self.writes=[]
            with self.connect() as db:
                db.execute("CREATE TABLE jobs(id TEXT,status TEXT,result TEXT,error TEXT,updated REAL,progress REAL)")
                db.execute("INSERT INTO jobs VALUES('job','queued',NULL,NULL,0,0)")
        @contextmanager
        def connect(self):
            with sqlite3.connect(self.database) as connection:
                owner=self
                class DB:
                    def execute(self,query,parameters=()):
                        if query.startswith("UPDATE jobs SET progress="):owner.writes.append(parameters[0])
                        return connection.execute(query,parameters)
                yield DB()
        def execute(self,job,payload,progress):return manager.execute_batch(job,payload,progress)
        def _fail_queue(self,*args,**kwargs):raise AssertionError("Unexpected queue storage fault")
    return Host()


def test_real_queue_body_with_real_workspace_noarg_does_not_reset_and_success_commits(tmp_path):
    manager=Manager(tmp_path);document=project(manager,3);host=queue_host(tmp_path,manager)
    host._run_queue_item("job",dict(project_id=document["project_id"],quality_after=True))
    with sqlite3.connect(host.database) as db:status,progress=db.execute("SELECT status,progress FROM jobs").fetchone()
    assert status=="done" and progress==1
    assert host.writes==sorted(host.writes) and 0<max(host.writes)<1
    assert host.writes.count(0)<=1


@pytest.mark.parametrize("failure,expected_status",[(MemoryError("unmodified resource reserve"),"failed"),(Cancelled("cancel during frame"),"cancelled")])
def test_real_queue_retains_failed_fraction_without_100_percent(tmp_path,failure,expected_status):
    manager=Manager(tmp_path);document=project(manager,2);host=queue_host(tmp_path,manager)
    manager.generation_error=failure
    host._run_queue_item("job",dict(project_id=document["project_id"],continue_on_error=False))
    with sqlite3.connect(host.database) as db:status,progress=db.execute("SELECT status,progress FROM jobs").fetchone()
    assert status==expected_status and 0<progress<1 and progress==host.writes[-1]


def test_native_host_uses_real_comfy_progressbar_body_integer_ticks(tmp_path,monkeypatch):
    from drama3_t8 import creative_nodes
    hooks=[];clock=Clock()
    namespace=dict(time=clock,PROGRESS_BAR_HOOK=lambda value,total,preview,**kwargs:hooks.append((value,total)),
        PROGRESS_THROTTLE_MIN_INTERVAL=.1,PROGRESS_THROTTLE_MIN_PERCENT=.5)
    host=Path(os.environ.get("DRAMA_COMFY_ROOT",ROOT/"vendor/ComfyUI"))
    bar=extract(host/"comfy/utils.py",None,"ProgressBar",namespace)
    comfy=ModuleType("comfy");utils=ModuleType("comfy.utils");mm=ModuleType("comfy.model_management")
    utils.ProgressBar=bar;mm.throw_exception_if_processing_interrupted=lambda:None
    comfy.utils=utils;comfy.model_management=mm
    for name,module in (("comfy",comfy),("comfy.utils",utils),("comfy.model_management",mm)):monkeypatch.setitem(sys.modules,name,module)
    progress=creative_nodes._Progress();manager=Manager(tmp_path);document=project(manager,3)
    run(manager,document,progress,quality_after=True)
    assert progress.bar.current==9900 and progress.bar.total==10000
    assert hooks and any(0<value<total for value,total in hooks)
    assert all(value<total for value,total in hooks)
    progress.complete()
    assert hooks[-1]==(10000,10000)


def test_native_commit_barrier_failure_never_calls_host_completion(tmp_path,monkeypatch):
    # The parent/node controls complete(). This proves the shared pipeline only
    # emits pre-commit ticks; existing native output-failure tests cover its node.
    manager=Manager(tmp_path);document=project(manager,1);sink=Sink()
    def reject_commit(job,progress):progress();raise OSError("native output barrier")
    manager.studio.begin_commit=reject_commit
    with pytest.raises(OSError):run(manager,document,sink)
    assert sink.values[-1]==(9800,10000) and all(value<total for value,total in sink.values)
