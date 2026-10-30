"""Real durable checkpoints with the complete preferences saved by Studio UI.

The injected callback writes PCM/native envelopes; these are CPU persistence and
dependency tests, not inference/audio-quality acceptance.
"""
from copy import deepcopy

import pytest

from drama3_t8.creative_workspace import CreativeWorkspace
from test_creative_workspace import media


SAMPLE = {"temperature": .7, "top_p": .7, "top_k": 30, "max_frames": 420, "opening_retries": 0}
MODEL = {"source": "current-code", "codec": {"sha256": "a" * 64, "size": 100}}


def ui_settings(takes=3, quality=True):
    return {"direction": "轻松对白", "scene": "拍摄现场", "characters": "三人吐槽",
            "chunk_bytes": 300, "split_long_text": True, "script_source": "完整六句剧本",
            "script_kind": "lines", "generation_mode": "per_line_context", "takes_per_line": takes,
            "base_seed": "42", "lookahead": 1, "sampling": deepcopy(SAMPLE),
            "quality_after": quality, "max_quality_retries": 0}


def setup(tmp_path, count=3):
    workspace = CreativeWorkspace(tmp_path / "workspace")
    project = workspace.create_project("完整UI偏好", [
        {"line_id": f"L{i+1:02}", "role": "ABC"[i % 3], "text": f"第{i+1}句"}
        for i in range(count)], settings=ui_settings(), cast={
            role: {"speaker": index, "voice_id": f"voice-{role}", "version_id": f"version-{role}"}
            for index, role in enumerate("ABC"[:min(count, 3)])})
    return workspace, project["project_id"]


def options(**overrides):
    return {"takes_per_line": 3, "base_seed": 42, "mode": "per_line_context", "lookahead": 1,
            "sampling": deepcopy(SAMPLE), "model_identity": deepcopy(MODEL), **overrides}


def test_ui_saved_n1_parents_to_n3_target_and_qa_toggle_do_not_stale_context(tmp_path):
    workspace, pid = setup(tmp_path)
    workspace.update_project(pid, {"settings": ui_settings(1, False)})
    parent = workspace.run_generation(pid, media, **options(takes_per_line=1, line_ids=["L01", "L02"]))
    parents = {line["line_id"]: line["selected_candidate_id"] for line in parent["project"]["lines"][:2]}
    workspace.update_project(pid, {"settings": ui_settings(3, True)})
    for line_id, candidate_id in parents.items():
        workspace.set_candidate_review(pid, line_id, candidate_id, locked=True)
    plan = workspace.plan_generation(pid, **options(line_ids=["L03"]))
    assert (plan["pending"], plan["blocked"]) == (3, 0)
    target = [job for job in plan["jobs"] if job["status"] == "pending"]
    assert [job["seed"] for job in target] == [2042, 2043, 2044]
    assert all(job["context_candidate_ids"] == [parents["L02"]] for job in target)
    assert all(job["input_snapshot"]["context"][0]["session_sha256"] == next(
        asset["sha256"] for asset in parent["project"]["candidates"][parents["L02"]]["artifacts"]
        if asset["kind"] == "session") for job in target)
    done = workspace.run_generation(pid, media, **options(line_ids=["L03"]))
    assert len(done["generated"]) == 3
    assert all(done["project"]["lines"][i]["selected_candidate_id"] == parents[line_id]
               for i, line_id in enumerate(("L01", "L02")))
    assert done["project"]["settings"] == ui_settings(3, True), "UI preferences remain complete on disk"


def test_ui_window_three_to_six_only_adds_missing_takes(tmp_path):
    workspace, pid = setup(tmp_path, 1)
    first = workspace.run_generation(pid, media, **options())
    workspace.update_project(pid, {"settings": ui_settings(6, False)})
    plan = workspace.plan_generation(pid, **options(takes_per_line=6))
    assert (plan["completed"], plan["pending"], plan["blocked"]) == (3, 3, 0)
    done = workspace.run_generation(pid, media, **options(takes_per_line=6))
    assert len(done["generated"]) == 3
    assert set(first["generated"]).isdisjoint(done["generated"])
    assert len(done["project"]["candidates"]) == 6


def test_ui_window_four_to_three_keeps_deliberately_selected_take_four_as_context(tmp_path):
    workspace, pid = setup(tmp_path, 2)
    workspace.update_project(pid, {"settings": ui_settings(4, False)})
    first = workspace.run_generation(pid, media, **options(takes_per_line=4, line_ids=["L01"]))
    chosen = first["generated"][-1]
    workspace.set_candidate_review(pid, "L01", chosen, selected=True, locked=True)
    workspace.update_project(pid, {"settings": ui_settings(3, True)})
    plan = workspace.plan_generation(pid, **options(line_ids=["L02"]))
    assert (plan["pending"], plan["blocked"]) == (3, 0)
    assert all(job["context_candidate_ids"] == [chosen] for job in plan["jobs"] if job["line_id"] == "L02")
    assert plan["jobs"][0]["selected_candidate"]["seed"] == 45


@pytest.mark.parametrize("mode", ["per_line", "per_line_context", "long_form"])
def test_non_stream_future_text_and_editor_script_do_not_bind_unused_lookahead(tmp_path, mode):
    workspace, pid = setup(tmp_path, 2)
    before = workspace.plan_generation(pid, **options(mode=mode))["jobs"][0]
    current = workspace.get_project(pid)
    current["lines"][1]["original_text"] = "只改第二句"
    settings = {**current["settings"], "script_source": "另一份包含第二句变化的完整稿", "script_kind": "tagged", "lookahead": 9}
    workspace.update_project(pid, {"settings": settings, "lines": current["lines"]})
    after = workspace.plan_generation(pid, **options(mode=mode, lookahead=9))["jobs"][0]
    assert before["fingerprint"] == after["fingerprint"]
    assert before["lookahead_lines"] == after["lookahead_lines"] == []
    assert before["input_snapshot"]["speaker_map"] == {"A": 0}


def test_stream_actual_future_text_and_speaker_map_remain_bound(tmp_path):
    workspace, pid = setup(tmp_path, 2)
    before = workspace.plan_generation(pid, **options(mode="text_stream"))["jobs"][0]
    current = workspace.get_project(pid)
    current["lines"][1]["original_text"] = "真实前瞻内容更改"
    workspace.update_project(pid, {"lines": current["lines"]})
    changed = workspace.plan_generation(pid, **options(mode="text_stream"))["jobs"][0]
    assert changed["fingerprint"] != before["fingerprint"]
    assert changed["lookahead_lines"][0]["readout_text"] == "真实前瞻内容更改"
    cast = deepcopy(current["cast"])
    cast["B"]["speaker"] = 9
    workspace.update_project(pid, {"cast": cast})
    mapped = workspace.plan_generation(pid, **options(mode="text_stream"))["jobs"][0]
    assert mapped["fingerprint"] != changed["fingerprint"]
    assert mapped["input_snapshot"]["speaker_map"] == {"A": 0, "B": 9}


@pytest.mark.parametrize("change", ["sampling", "direction", "unknown_setting", "reference", "model"])
def test_real_effective_generation_inputs_still_invalidate(tmp_path, change):
    workspace, pid = setup(tmp_path, 1)
    done = workspace.run_generation(pid, media, **options())
    original = done["project"]["lines"][0]["selected_candidate_id"]
    opts = options()
    current = workspace.get_project(pid)
    if change == "sampling":
        opts["sampling"]["temperature"] = .9
    elif change == "model":
        opts["model_identity"]["source"] = "different-code"
    elif change == "reference":
        cast = deepcopy(current["cast"])
        cast["A"].update(voice_id="voice-new", version_id="reference-new")
        workspace.update_project(pid, {"cast": cast})
    else:
        settings = deepcopy(current["settings"])
        settings["direction" if change == "direction" else "unknown_future_inference_flag"] = "真正改变"
        workspace.update_project(pid, {"settings": settings})
    plan = workspace.plan_generation(pid, **opts)
    assert plan["pending"] == 3 and plan["completed"] == 0
    assert all(job["selected_candidate"] is None for job in plan["jobs"])
    assert workspace.get_project(pid)["lines"][0]["selected_candidate_id"] == original


def test_variation_and_legacy_selection_use_same_conservative_projection(tmp_path):
    from drama3_t8.creative_workspace import generation_settings, _selection_fingerprint
    workspace, pid = setup(tmp_path, 1)
    done = workspace.run_generation(pid, media, **options())
    source = done["project"]["candidates"][done["generated"][0]]
    old = deepcopy(source["input_snapshot"])
    old["settings"] = ui_settings(17, False)
    assert _selection_fingerprint(old) == _selection_fingerprint(source["input_snapshot"])
    old["settings"]["direction"] = "真正改变"
    assert _selection_fingerprint(old) != _selection_fingerprint(source["input_snapshot"])
    observed = []
    workspace.update_project(pid, {"settings": ui_settings(8, False)})
    workspace.run_variation(pid, "L01", source["candidate_id"],
        lambda job, directory: (observed.append(deepcopy(job)) or media(job, directory)),
        {"kind": "revise", "text": "新的台词"})
    assert observed[0]["settings"] == generation_settings(ui_settings(8, False))
    assert observed[0]["input_snapshot"]["settings"] == observed[0]["settings"]
    assert generation_settings({"unknown_future": {"a": 1}}) == {"unknown_future": {"a": 1}}
