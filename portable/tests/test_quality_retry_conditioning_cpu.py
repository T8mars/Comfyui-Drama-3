"""Actual _generate and State round trips with a deliberately synthetic CPU engine."""
from contextlib import contextmanager
from types import MethodType,SimpleNamespace

import numpy as np
import pytest
import torch

from drama3_t8.state import Session,Take,load_session
from drama3_t8.creative_studio import CreativeStudio
from drama3_t8.creative_quality import run_quality_retries,REFERENCE_ONLY_POLICY
from test_creative_quality import setup


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError("CPU conditioning regression must not query or initialize CUDA")
    monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
    monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    from drama3_t8.bootstrap import standalone_engine
    standalone_engine()
    from comfy.cli_args import args
    monkeypatch.setattr(args,"cpu",True)
    import comfy_kitchen
    monkeypatch.setattr(comfy_kitchen,"int8_attention_is_available",lambda *a,**k:False)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda *a,**k:"CPU conditioning test")
    yield
    assert not torch.cuda.is_initialized()


class Engine:
    moe_backend=gdn_backend=requested_backend="torch"
    expert_placement="ram"
    cuda_graph=False
    def __init__(self,owner):self.owner=owner;self.depth=0;self.drama=self.codec=None;self.seen=[];self.loads=0
    def load_model(self):
        if self.drama is None:self.drama=SimpleNamespace(last_graph_stats={});self.loads+=1
        return self.drama
    @contextmanager
    def generation_session(self):
        self.depth+=1
        try:yield
        finally:self.depth-=1
    def unload(self):self.drama=self.codec=None
    def expert_stats(self):return dict(actual_backend="torch",placement="ram",fallback_reason=None)
    def _make(self,lines,session,sampling,on_codes,scene):
        from drama3_t8.prompts import spoken_text
        self.seen.append(dict(lines=list(lines),history=len(session.takes),references=tuple(session.references),direction=session.direction,scene=scene))
        pieces=[]
        for index,line in enumerate(lines):
            frames=torch.zeros((3,11),dtype=torch.int64);frames[:,0]=248099;pieces.append(frames)
            if index+1<len(lines):
                separator=torch.zeros((1,11),dtype=torch.int64);separator[0,0]=252196;pieces.append(separator)
        if on_codes:on_codes(pieces[0][0])
        return session.append(Take("\n".join(lines),spoken_text("\n".join(lines),scene=scene),torch.cat(pieces),metadata=dict(
            seed=sampling.seed,requested_seed=sampling.seed,finish_reason="eos",length_limit_reached=False,
            task="scene" if scene else "tts",line_count=len(lines),gdn_calls_by_backend={"torch":1})))
    def speak(self,text,session,sampling,on_codes,progress,*,note=""):
        from drama3_t8.prompts import directed_text
        progress(1,2);return self._make([directed_text(text,session.direction,note)],session,sampling,on_codes,False)
    def scene(self,lines,scene,characters,*,references,sampling,progress,session,note,on_codes,**kwargs):
        assert references is session.references
        progress(1,2);return self._make(lines,session,sampling,on_codes,True)
    def decode_take(self,take,with_timeline=False):
        count=take.metadata["line_count"];n=count*3*2048
        waveform=torch.from_numpy((np.sin(np.arange(n)*.08)*.1).astype("float32"))[None,None]
        self.codec=object();audio={"waveform":waveform,"sample_rate":44100}
        timeline=[dict(line=index,start=index*6144/44100,end=(index+1)*6144/44100) for index in range(count)]
        return (audio,timeline) if with_timeline else audio


@pytest.mark.parametrize("mode,lines",[("per_line",["保留目标原稿"]),("together",["第一句原稿","第二句原稿"]),("together",["单行场景原稿"])])
def test_real_generate_auto_retry_keeps_references_but_never_bad_history(setup,tmp_path,mode,lines):
    manager,build=setup;project,initial=build(mode=mode,lines=lines)
    source=project["candidates"][project["lines"][0]["selected_candidate_id"]]
    before=load_session(source["session_path"])
    manager.reports=tmp_path/"reports";manager.reports.mkdir()
    manager.performance_binding=lambda *args:{}
    manager.studio.engine=Engine(manager)
    manager._generate=MethodType(CreativeStudio._generate,manager)
    result=run_quality_retries(manager,"cpu-job",dict(project_id=project["project_id"],max_quality_retries=1),initial,lambda *args:None)
    seen=manager.studio.engine.seen
    assert len(seen)==1 and seen[0]["history"]==0 and len(seen[0]["lines"])==len(lines)
    assert len(seen[0]["references"])==len(before.references)>0
    assert torch.equal(seen[0]["references"][0].codes,before.references[0].codes)
    new_id=result["rounds"][0]["generated"][0]
    current=manager.workspace.get_project(project["project_id"])
    loaded=load_session(current["candidates"][new_id]["session_path"])
    assert loaded.takes[-1].metadata["creative_context"]["policy"]==REFERENCE_ONLY_POLICY
    assert loaded.takes[-1].metadata["creative_context"]["candidate_ids"]==[]
    assert current["lines"][0]["selected_candidate_id"]==source["candidate_id"]


def test_real_generate_plain_manual_revision_preserves_source_history(setup,tmp_path):
    manager,build=setup;project,initial=build(lines=["手动改稿"])
    source=project["candidates"][project["lines"][0]["selected_candidate_id"]]
    manager.reports=tmp_path/"reports";manager.reports.mkdir();manager.performance_binding=lambda *args:{}
    manager.studio.engine=Engine(manager)
    result=manager.workspace.run_variation(project["project_id"],source["line_ids"][0],source["candidate_id"],
        lambda job,directory:CreativeStudio._generate(manager,job,directory,lambda *args:None),
        dict(kind="revise",text="手动改稿",note="仅改变表演"),sampling={"seed":43},model_identity=source["input_snapshot"]["model_identity"])
    assert result["status"]=="completed" and manager.studio.engine.seen[0]["history"]==1
