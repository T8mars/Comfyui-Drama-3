"""Real Torch-CPU State archive round trips, separate from the no-Torch suite."""
import sys

import pytest
import torch


def forbidden(*args,**kwargs):
    raise AssertionError("CPU State progress test must not query or initialize CUDA")


from drama3_t8.state import Reference,Session,Take,save_session,load_session
from test_creative_progress_integration import Manager,project,run,Sink,assert_monotonic


@pytest.fixture(autouse=True)
def cuda_forbidden(monkeypatch):
    monkeypatch.setattr(torch.cuda,"_lazy_init",forbidden)
    monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    yield
    assert not torch.cuda.is_initialized()


class StateManager(Manager):
    def _generate(self,job,directory,progress,parent_recorder=None):
        result=super()._generate(job,directory,progress,parent_recorder)
        take=Take(job["readout_text"],job["readout_text"],torch.tensor([[248099,*([0]*10)]],dtype=torch.int64),
            metadata=dict(seed=job["seed"],requested_seed=job["seed"],finish_reason="eos",length_limit_reached=False))
        reference=Reference("固定音色的参考文本",torch.zeros((10,1),dtype=torch.int64))
        save_session(Session(references=(reference,),takes=(take,)),result["session_path"])
        restored=load_session(result["session_path"])
        assert torch.equal(restored.takes[-1].frames,take.frames)
        assert restored.takes[-1].transcript==job["readout_text"]
        return result


@pytest.mark.parametrize("retries",[0,2])
def test_actual_state_round_trip_and_workspace_quality_transactions(tmp_path,retries):
    manager=StateManager(tmp_path,[False,False,True] if retries else [True])
    document=project(manager,2);sink=Sink()
    result=run(manager,document,sink,max_quality_retries=retries,quality_after=True)
    final=manager.workspace.get_project(document["project_id"])
    assert len(final["candidates"])==(6 if retries else 2)
    for candidate in final["candidates"].values():
        loaded=load_session(candidate["session_path"])
        assert loaded.takes[-1].transcript==candidate["transcript"]
    assert_monotonic(sink.values)
    assert not torch.cuda.is_initialized()
