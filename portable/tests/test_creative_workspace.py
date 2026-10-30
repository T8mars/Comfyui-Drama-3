"""Durable project behavior with real PCM and native-envelope fixture files.

Inference is injected; these CPU tests are not model/audio-quality validation.
"""
import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import uuid
import wave
import zipfile

import pytest

from drama3_t8.creative_workspace import (
    CreativeWorkspace, InvalidCheckpoint, ProjectConflict, model_file_identity,
    normalize_dictionary, preview_text, text_quality,
)
from drama3_t8.workspace_lock import WorkspaceBusy


def native_session(path, text="你好", *, complete=True, with_take=True, seed=None, requested_seed=None, from_source_view=None):
    key = "take_0"
    metadata = {"version": 1, "session_id": uuid.uuid4().hex, "direction": "", "references": [],
                "takes": [{"key": key, "take_id": uuid.uuid4().hex, "transcript": text,
                           "prompt_text": text, "semantic_start": 248099, "semantic_end": 252194,
                           "separator_id": 252196, "metadata": {"length_limit_reached": not complete,
                                                                  "finish_reason": "eos" if complete else "stalled"}}] if with_take else []}
    header = {"__metadata__": {"drama3_t8_session": json.dumps(metadata)},
              key: {"dtype": "I64", "shape": [1, 11], "data_offsets": [0, 88]}}
    if seed is not None and with_take:
        metadata["takes"][-1]["metadata"].update(seed=seed, requested_seed=requested_seed)
        header["__metadata__"]["drama3_t8_session"] = json.dumps(metadata)
    if from_source_view is not None and with_take:
        metadata["takes"][-1]["metadata"]["from_source_view"] = from_source_view
        header["__metadata__"]["drama3_t8_session"] = json.dumps(metadata)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<11q", 248099, *([0] * 10)))


def media(job, directory, **changes):
    audio = directory / "audio.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(struct.pack("<80h", *([100] * 80)))
    session = directory / "session.safetensors"
    native_session(session, job["readout_text"])
    return {"complete": True, "seed": job["seed"], "audio_path": str(audio), "session_path": str(session),
            "duration_seconds": .01, "sample_rate": 8000,
            "metadata": {"finish_reason": "eos", "actual_segments": len(job["lines"])}, **changes}


@pytest.fixture
def workspace(tmp_path):
    return CreativeWorkspace(tmp_path / "creative")


def project(workspace, *, lines=None, **kwargs):
    return workspace.create_project("测试工程", lines or [{"line_id": "L01", "role": "导演", "text": "各部门注意"},
                                                       {"line_id": "L02", "role": "猫", "text": "又要加班了"}], **kwargs)


def test_original_readout_subtitle_and_protected_speaker_tags():
    result = preview_text("<|speaker:12|> Drama3：2026-02-03 发布12个模型。", {
        "aliases": {"Drama3": "抓马三"}, "number_mode": "cardinal", "expand_dates": True})
    assert result["original_text"] == result["subtitle_text"]
    assert result["readout_text"] == "<|speaker:12|> 抓马三：二零二六年二月三日 发布十二个模型。"
    assert result["applied_rules"][0]["replacement"] == "抓马三"


def test_dictionary_longest_non_cascading_case_rules_and_version():
    rules = {"rules": [{"source": "ABC", "replacement": "XYZ", "case_sensitive": False},
                       {"source": "AB", "replacement": "短"}, {"source": "XYZ", "replacement": "误读"}]}
    assert preview_text("abc ABC XYZ AB", rules)["readout_text"] == "XYZ XYZ 误读 短"
    assert normalize_dictionary(rules)["version"] == normalize_dictionary(rules)["version"]
    assert normalize_dictionary(rules)["version"] != normalize_dictionary({"aliases": {"ABC": "甲"}})["version"]


@pytest.mark.parametrize("text,expected", [
    ("2024-02-29", "二零二四年二月二十九日"),
    ("2000/2/29", "二零零零年二月二十九日"),
    ("2026年10月09日", "二零二六年十月九日"),
    ("09:05", "九点五分"), ("00:00", "零点零分"), ("23:59", "二十三点五十九分"),
    ("2026-02-29", "2026-02-29"), ("1900年2月29日", "1900年2月29日"),
    ("2026-04-31", "2026-04-31"), ("0000-01-01", "0000-01-01"),
    ("2026年13月1日", "2026年13月1日"), ("24:00", "24:00"), ("12:60", "12:60"),
])
@pytest.mark.parametrize("mode", ["none", "digits", "cardinal"])
def test_temporal_preview_calendar_validation_and_clock(text, expected, mode):
    result = preview_text(text, {"expand_dates": True, "number_mode": mode})
    assert result["readout_text"] == expected
    assert result["original_text"] == result["subtitle_text"] == text


def test_date_option_preserves_tags_and_expands_only_surrounding_numbers():
    result = preview_text("<|speaker:12|><date:2026-02-31> 2026-02-31 有12条，2026年3月1日09:05。", {
        "expand_dates": True, "number_mode": "cardinal"})
    assert result["readout_text"] == "<|speaker:12|><date:2026-02-31> 2026-02-31 有十二条，二零二六年三月一日九点五分。"
    assert preview_text("2026年3月1日09:05", {"expand_dates": False})["readout_text"] == "2026年3月1日09:05"


@pytest.mark.parametrize("text,expected", [("10", "十"), ("101", "一百零一"), ("10010", "一万零一十"),
                                           ("100000001", "一亿零一"), ("0012", "零零一二"),
                                           ("-3.5", "负三点五")])
def test_number_preview(text, expected):
    assert preview_text(text, {"number_mode": "cardinal"})["readout_text"] == expected


def test_project_roundtrip_manual_override_and_revision_conflict(workspace):
    first = project(workspace, dictionary={"aliases": {"加班": "下班"}})
    assert first["lines"][1]["original_text"] == "又要加班了"
    assert first["lines"][1]["readout_text"] == "又要下班了"
    lines = first["lines"]
    lines[1]["readout_override"] = "我先走了"
    second = workspace.update_project(first["project_id"], {"lines": lines}, first["revision"])
    third = workspace.update_project(first["project_id"], {"dictionary": {"aliases": {"加班": "开会"}}})
    assert third["lines"][1]["readout_text"] == "我先走了"
    assert third["lines"][1]["subtitle_text"] == "又要加班了"
    with pytest.raises(ProjectConflict):
        workspace.update_project(first["project_id"], {"name": "过期编辑"}, first["revision"])
    assert len(list((workspace.projects / first["project_id"] / "history").glob("*.json"))) == 2
    assert second["revision"] == first["revision"] + 1


def test_stable_line_ids_archive_removed_lines_and_reorder(workspace):
    first = project(workspace)
    updated = workspace.update_project(first["project_id"], {"lines": list(reversed(first["lines"]))})
    assert [line["line_id"] for line in updated["lines"]] == ["L02", "L01"]
    archived = workspace.update_project(first["project_id"], {"lines": [updated["lines"][0]]})
    assert archived["archived_lines"][0]["line_id"] == "L01"
    with pytest.raises(ValueError):
        workspace.update_project(first["project_id"], {"lines": [{"line_id": "../bad", "text": "不行"}]})


def test_multi_take_restart_resume_and_additional_target_count(workspace):
    first = project(workspace)
    jobs = []
    def generate(job, directory):
        jobs.append(job)
        return media(job, directory)
    finished = workspace.run_generation(first["project_id"], generate, takes_per_line=3, base_seed=100)
    assert finished["status"] == "completed"
    assert [job["seed"] for job in jobs] == [100, 101, 102, 1100, 1101, 1102]
    assert all(job["sampling"]["seed"] == job["seed"] for job in jobs)
    assert len(finished["project"]["candidates"]) == 6
    reopened = CreativeWorkspace(workspace.root)
    resumed = reopened.run_generation(first["project_id"], lambda *_: pytest.fail("reused checkpoint generated"),
                                      takes_per_line=3, base_seed=100)
    assert resumed["generated"] == []
    extra = reopened.run_generation(first["project_id"], generate, takes_per_line=4, base_seed=100)
    assert len(extra["generated"]) == 2
    assert len(extra["project"]["candidates"]) == 8


def test_corrupt_audio_is_not_reused_and_original_candidate_remains(workspace):
    first = project(workspace, lines=["你好"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    original = next(iter(done["project"]["candidates"].values()))
    Path(original["audio_path"]).write_bytes(b"broken WAV")
    assert workspace.plan_generation(first["project_id"], takes_per_line=1)["pending"] == 1
    repaired = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    assert len(repaired["generated"]) == 1
    assert len(repaired["project"]["candidates"]) == 2
    assert Path(original["audio_path"]).read_bytes() == b"broken WAV"
    assert repaired["project"]["lines"][0]["selected_candidate_id"] != original["candidate_id"]


def test_file_exists_with_bad_session_is_failed_not_completed(workspace):
    first = project(workspace, lines=["你好"])
    def broken(job, directory):
        returned = media(job, directory)
        Path(returned["session_path"]).write_bytes(b"not a session")
        return returned
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], broken, takes_per_line=1)
    saved = workspace.get_project(first["project_id"])
    assert next(iter(saved["candidates"].values()))["status"] == "failed"
    skipped = workspace.run_generation(first["project_id"], lambda *_: pytest.fail("failed retry disabled"),
                                       takes_per_line=1, retry_failed=False)
    assert skipped["status"] == "partial"
    resumed = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    assert resumed["status"] == "completed"


@pytest.mark.parametrize("changes", [{"complete": False}, {"seed": 999}, {"duration_seconds": 3.0},
                                     {"sample_rate": 44100}, {"metadata": {"length_limit_reached": True}},
                                     {"metadata": {"actual_segments": 7}}])
def test_callback_contract_prevents_false_completion(workspace, changes):
    first = project(workspace, lines=["你好"])
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], lambda job, directory: media(job, directory, **changes), takes_per_line=1)
    assert not any(item["status"] == "completed" for item in workspace.get_project(first["project_id"])["candidates"].values())


def test_reference_only_session_cannot_be_a_generation_checkpoint(workspace):
    first = project(workspace, lines=["你好"])
    def reference_only(job, directory):
        returned = media(job, directory)
        native_session(Path(returned["session_path"]), with_take=False)
        return returned
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], reference_only, takes_per_line=1)


def test_context_recovery_supplies_intact_session_and_selection_invalidates_descendants(workspace):
    first = project(workspace)
    seen = []
    def generate(job, directory):
        seen.append(job)
        return media(job, directory)
    done = workspace.run_generation(first["project_id"], generate, takes_per_line=2, mode="per_line_context")
    assert len(done["generated"]) == 4
    context_job = next(job for job in seen if job["line_id"] == "L02")
    assert Path(context_job["context_candidates"][0]["session_path"]).is_file()
    assert context_job["input_snapshot"]["context"][0]["session_sha256"]
    assert workspace.plan_generation(first["project_id"], takes_per_line=2, mode="per_line_context")["pending"] == 0
    another = done["project"]["lines"][0]["candidates"][1]
    workspace.set_candidate_review(first["project_id"], "L01", another["candidate_id"], selected=True)
    plan = workspace.plan_generation(first["project_id"], takes_per_line=2, mode="per_line_context")
    assert {job["line_id"] for job in plan["jobs"] if job["status"] == "pending"} == {"L02"}


def test_line_text_model_reference_seed_dictionary_and_lookahead_fingerprints(workspace):
    first = project(workspace, cast={"导演": {"voice_id": "a", "version": "1"}, "猫": {"voice_id": "b", "version": "1"}})
    baseline = workspace.plan_generation(first["project_id"], takes_per_line=1)["jobs"]
    model_changed = workspace.plan_generation(first["project_id"], takes_per_line=1, model_identity={"codec": "new"})["jobs"]
    seed_changed = workspace.plan_generation(first["project_id"], takes_per_line=1, base_seed=43)["jobs"]
    assert all(a["fingerprint"] != b["fingerprint"] for a, b in zip(baseline, model_changed))
    assert all(a["fingerprint"] != b["fingerprint"] for a, b in zip(baseline, seed_changed))
    old_look = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="text_stream", lookahead=1)["jobs"][0]["fingerprint"]
    lines = first["lines"]
    lines[1]["original_text"] = "终于下班了"
    workspace.update_project(first["project_id"], {"lines": lines})
    changed = workspace.plan_generation(first["project_id"], takes_per_line=1)["jobs"]
    assert changed[0]["fingerprint"] == baseline[0]["fingerprint"]
    assert changed[1]["fingerprint"] != baseline[1]["fingerprint"]
    assert workspace.plan_generation(first["project_id"], takes_per_line=1, mode="text_stream", lookahead=1)["jobs"][0]["fingerprint"] != old_look
    workspace.update_project(first["project_id"], {"cast": {"导演": {"voice_id": "a", "version": "2"}, "猫": {"voice_id": "b", "version": "1"}}})
    assert workspace.plan_generation(first["project_id"], takes_per_line=1)["jobs"][0]["fingerprint"] != changed[0]["fingerprint"]
    previous = workspace.plan_generation(first["project_id"], takes_per_line=1)["jobs"][1]["fingerprint"]
    workspace.update_project(first["project_id"], {"dictionary": {"aliases": {"终于": "总算"}}})
    assert workspace.plan_generation(first["project_id"], takes_per_line=1)["jobs"][1]["fingerprint"] != previous


@pytest.mark.parametrize("mode,expected_units,groups", [("together", 1, [["L01", "L02", "L03"]]),
                                                        ("per_speaker", 2, [["L01", "L03"], ["L02"]])])
def test_atomic_scene_and_role_units_are_not_faked_per_line(workspace, mode, expected_units, groups):
    first = project(workspace, lines=[{"line_id": "L01", "role": "猫", "text": "一"},
                                     {"line_id": "L02", "role": "人", "text": "二"},
                                     {"line_id": "L03", "role": "猫", "text": "三"}])
    plan = workspace.plan_generation(first["project_id"], mode=mode, takes_per_line=1)
    assert plan["unit_count"] == expected_units
    assert [job["line_ids"] for job in plan["jobs"]] == groups
    done = workspace.run_generation(first["project_id"], media, mode=mode, takes_per_line=1)
    assert len(done["generated"]) == expected_units
    assert done["project"]["lines"][0]["selected_candidate_id"] == done["project"]["lines"][2]["selected_candidate_id"]


def test_locked_stale_media_never_becomes_new_context(workspace):
    first = project(workspace)
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1, mode="per_line_context")
    candidate_id = done["project"]["lines"][0]["selected_candidate_id"]
    workspace.set_candidate_review(first["project_id"], "L01", candidate_id, locked=True, rating=5, favorite=True, notes="保留这个表演")
    plan = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context")
    assert plan["jobs"][0]["status"] == "locked"
    changed = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context", model_identity={"model": "new"})
    assert [job["status"] for job in changed["jobs"]] == ["locked_stale", "blocked"]
    assert changed["jobs"][1]["context_candidates"] == []
    frozen = workspace.get_project(first["project_id"])["lines"][0]["selected_candidate"]
    assert frozen["rating"] == 5 and frozen["favorite"] and frozen["notes"]


def test_generation_locks_mutations_but_allows_polling(workspace):
    first = project(workspace, lines=["你好"])
    results = []
    def generate(job, directory):
        assert workspace.get_project(first["project_id"])["candidates"]
        def edit():
            try:
                workspace.update_project(first["project_id"], {"name": "并发修改"})
            except Exception as exc:
                results.append(exc)
        worker = threading.Thread(target=edit)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        return media(job, directory)
    workspace.run_generation(first["project_id"], generate, takes_per_line=1)
    assert isinstance(results[0], WorkspaceBusy)
    assert workspace.get_project(first["project_id"])["name"] == "测试工程"


def test_interrupted_candidate_files_preserved_and_retried(workspace):
    first = project(workspace, lines=["你好"])
    old_files = []
    def crash(job, directory):
        returned = media(job, directory)
        old_files.append(Path(returned["audio_path"]))
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        workspace.run_generation(first["project_id"], crash, takes_per_line=1)
    assert old_files[0].is_file()
    done = CreativeWorkspace(workspace.root).run_generation(first["project_id"], media, takes_per_line=1)
    assert done["status"] == "completed"
    assert len(done["project"]["candidates"]) == 2
    assert old_files[0].is_file()


def test_quality_is_separate_soft_phase_and_keeps_selected_candidate(workspace):
    first = project(workspace, lines=["你好"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    selected = done["project"]["lines"][0]["selected_candidate_id"]
    reviewed = workspace.run_quality(first["project_id"], lambda candidate: {"transcript": "笑声", "independent": True})
    quality = reviewed["project"]["candidates"][selected]["quality"]
    assert quality["text_check"]["review_recommended"]
    assert quality["text_check"]["independent"]
    assert reviewed["project"]["candidates"][selected]["status"] == "completed"
    assert reviewed["project"]["lines"][0]["selected_candidate_id"] == selected
    assert text_quality("你好！", "你好")["normalized_match"]


def test_portable_project_pack_copies_intact_context_and_retains_old_project(workspace, tmp_path):
    first = project(workspace)
    done = workspace.run_generation(first["project_id"], media, takes_per_line=2, mode="per_line_context")
    path = tmp_path / "project.drama.zip"
    exported = workspace.export_project(first["project_id"], path)
    assert not exported["voice_assets_included"]
    assert len(exported["included_candidates"]) == 2
    imported = workspace.import_project(path)
    assert imported["project_id"] != first["project_id"]
    assert imported["origin_project_id"] == first["project_id"]
    assert len(workspace.list_projects()) == 2
    assert workspace.plan_generation(imported["project_id"], takes_per_line=1, mode="per_line_context")["pending"] == 0
    assert sum(candidate["status"] == "not_included" for candidate in imported["candidates"].values()) == 2
    assert Path(imported["lines"][0]["selected_candidate"]["audio_path"]).is_file()
    with pytest.raises(FileExistsError):
        workspace.export_project(first["project_id"], path)
    assert len(done["project"]["candidates"]) == 4


@pytest.mark.parametrize("member", ["../outside", "/absolute", "C:/absolute", "a\\b", "CON", "a/../b"])
def test_archive_path_attacks_rejected_before_project_commit(workspace, tmp_path, member):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr(member, "bad")
    with pytest.raises(ValueError):
        workspace.import_project(path)
    assert workspace.list_projects() == []


def test_archive_tampering_rejected(workspace, tmp_path):
    first = project(workspace, lines=["你好"])
    workspace.run_generation(first["project_id"], media, takes_per_line=1)
    path = tmp_path / "good.zip"
    workspace.export_project(first["project_id"], path)
    broken = tmp_path / "bad.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(broken, "w") as target:
        for item in source.infolist():
            target.writestr(item.filename, source.read(item.filename) + (b"tamper" if item.filename.endswith(".wav") else b""))
    with pytest.raises(ValueError):
        workspace.import_project(broken)
    assert len(workspace.list_projects()) == 1


def test_model_identity_detects_weight_replacement(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"first")
    first = model_file_identity({"model": path}, content_hash=True)
    path.write_bytes(b"second")
    assert first != model_file_identity({"model": path}, content_hash=True)


def test_actual_retry_seed_bound_to_native_session_and_spaced_candidates(workspace):
    first = project(workspace, lines=["你好"])
    requested = []
    def retry(job, directory):
        returned = media(job, directory)
        effective = (job["seed"] + 1) % 2**64
        native_session(Path(returned["session_path"]), seed=effective, requested_seed=job["seed"])
        returned["seed"] = effective
        requested.append(job["seed"])
        return returned
    done = workspace.run_generation(first["project_id"], retry, takes_per_line=3, sampling={"opening_retries": 1})
    assert requested == [42, 44, 46]
    candidates = list(done["project"]["candidates"].values())
    assert [candidate["seed"] for candidate in candidates] == [43, 45, 47]
    assert [candidate["requested_seed"] for candidate in candidates] == requested
    assert workspace.plan_generation(first["project_id"], takes_per_line=3, sampling={"opening_retries": 1})["pending"] == 0


def test_callback_cannot_lie_about_seed_in_native_session(workspace):
    first = project(workspace, lines=["你好"])
    def mismatch(job, directory):
        returned = media(job, directory)
        native_session(Path(returned["session_path"]), seed=200, requested_seed=job["seed"])
        return returned
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], mismatch, takes_per_line=1)


def test_long_form_anchor_changes_invalidate_every_later_chunk(workspace):
    first = project(workspace, lines=["第一段", "第二段", "第三段"])
    done = workspace.run_generation(first["project_id"], media, mode="long_form", takes_per_line=2)
    assert len(done["generated"]) == 6
    plan = workspace.plan_generation(first["project_id"], mode="long_form", takes_per_line=2)
    assert plan["pending"] == 0
    anchor = done["project"]["lines"][0]["candidates"][1]
    workspace.set_candidate_review(first["project_id"], "L01", anchor["candidate_id"], selected=True)
    changed = workspace.plan_generation(first["project_id"], mode="long_form", takes_per_line=2)
    assert {job["line_id"] for job in changed["jobs"] if job["status"] == "pending"} == {"L02", "L03"}
    for job in changed["jobs"]:
        if job["line_id"] != "L01":
            assert [candidate["candidate_id"] for candidate in job["context_candidates"]] == [anchor["candidate_id"]]


def test_partial_line_edit_preserves_role_direction_and_reference_selection(workspace):
    first = project(workspace, lines=[{"line_id": "L01", "role": "导演", "text": "原稿", "direction": "小声说",
                                     "reference_group": "耳语", "reference_ids": ["ref-1"]}])
    updated = workspace.update_project(first["project_id"], {"lines": [{"line_id": "L01", "text": "改稿"}]})
    line = updated["lines"][0]
    assert line["role"] == "导演" and line["direction"] == "小声说"
    assert line["reference_ids"] == ["ref-1"] and line["reference_group"] == "耳语"
    assert line["readout_text"] == "改稿"
    with pytest.raises(ValueError):
        workspace.update_project(first["project_id"], {"lines": [{"line_id": "L01", "text": "改稿", "target_seconds": 0}]})


def test_start_generation_expected_revision_rejects_stale_editor(workspace):
    first = project(workspace, lines=["你好"])
    workspace.update_project(first["project_id"], {"name": "新版"})
    with pytest.raises(ProjectConflict):
        workspace.run_generation(first["project_id"], lambda *_: pytest.fail("stale run executed"), expected_revision=first["revision"])


@pytest.mark.parametrize("value", [[], 1, "bad"])
def test_invalid_dictionary_container_rejected(value):
    with pytest.raises(ValueError):
        normalize_dictionary(value)


def test_cancelled_phase_keeps_old_candidates_and_stops_next_callback(workspace):
    first = project(workspace)
    cancelled = []
    def generate(job, directory):
        returned = media(job, directory)
        cancelled.append(True)
        return returned
    done = workspace.run_generation(first["project_id"], generate, cancelled=lambda: bool(cancelled))
    assert done["status"] == "cancelled"
    assert len(done["generated"]) == 1
    assert next(iter(done["project"]["candidates"].values()))["status"] == "completed"


@pytest.mark.parametrize("error_kind", ["Cancelled", "InterruptProcessingException", "MemoryError", "OSError", "OutOfMemoryError", "callback_cancelled"])
def test_cancel_and_resource_errors_stop_even_when_continue_on_error_is_enabled(workspace, error_kind):
    first = project(workspace, lines=["第一句", "第二句", "第三句"])
    calls, cancellation = [], []
    if error_kind == "OutOfMemoryError":
        import torch
        error = torch.OutOfMemoryError("actual torch OOM exception without allocating CUDA")
    elif error_kind == "MemoryError":
        error = MemoryError("host exhausted")
    elif error_kind == "OSError":
        error = OSError("storage unavailable")
    elif error_kind == "callback_cancelled":
        error = RuntimeError("cancelled by host callback")
    else:
        error = type(error_kind, (RuntimeError,), {})("stop requested")

    def generate(job, directory):
        calls.append(job["line_id"])
        if len(calls) == 2:
            (directory / "partial-evidence.txt").write_text("preserve", encoding="utf-8")
            if error_kind == "callback_cancelled":
                cancellation.append(True)
            raise error
        return media(job, directory)

    with pytest.raises(type(error)) as raised:
        workspace.run_generation(first["project_id"], generate, takes_per_line=1,
                                 continue_on_error=True, cancelled=lambda: bool(cancellation))
    assert raised.value is error
    assert calls == ["L01", "L02"]
    restored = CreativeWorkspace(workspace.root).get_project(first["project_id"])
    assert len(restored["candidates"]) == 2
    assert restored["lines"][0]["selected_candidate"]["status"] == "completed"
    assert Path(restored["lines"][0]["selected_candidate"]["audio_path"]).is_file()
    state = "cancelled" if error_kind in {"Cancelled", "InterruptProcessingException", "callback_cancelled"} else "interrupted"
    assert restored["runs"][-1]["status"] == state
    assert list(workspace.projects.glob("*/artifacts/*/partial-evidence.txt"))


def test_ordinary_generation_value_error_can_continue_to_next_independent_line(workspace):
    first = project(workspace, lines=["第一句", "第二句", "第三句"])
    calls = []
    def generate(job, directory):
        calls.append(job["line_id"])
        if len(calls) == 2:
            raise ValueError("missing generation separator")
        return media(job, directory)
    result = workspace.run_generation(first["project_id"], generate, takes_per_line=1, continue_on_error=True)
    assert calls == ["L01", "L02", "L03"]
    assert result["status"] == "partial" and len(result["generated"]) == 2 and len(result["failed"]) == 1


def test_native_checkpoint_read_oserror_is_not_hidden_as_recoverable_generation_failure(workspace, monkeypatch):
    first = project(workspace, lines=["一", "二", "三"])
    real_open, calls = Path.open, []
    def read_failure(path, mode="r", *args, **kwargs):
        if path.name == "session.safetensors" and mode == "rb":
            raise OSError("device read failure")
        return real_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", read_failure)
    def generate(job, directory):
        calls.append(job["line_id"])
        return media(job, directory)
    with pytest.raises(OSError, match="device read"):
        workspace.run_generation(first["project_id"], generate, takes_per_line=1, continue_on_error=True)
    assert calls == ["L01"]
    assert list(workspace.projects.glob("*/artifacts/*/audio.wav"))


def test_quality_failure_does_not_remove_or_fail_completed_generation(workspace):
    first = project(workspace, lines=["你好"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    def failure(_):
        raise RuntimeError("识别器不可用")
    reviewed = workspace.run_quality(first["project_id"], failure)
    candidate = reviewed["project"]["candidates"][done["generated"][0]]
    assert candidate["status"] == "completed"
    assert "识别器不可用" in candidate["quality"]["review_error"]


def test_archive_duplicate_case_members_rejected(workspace, tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", "{}")
        archive.writestr("Manifest.json", "{}")
    with pytest.raises(ValueError):
        workspace.import_project(path)


def test_project_edit_atomic_failure_keeps_old_document_and_pending_evidence(workspace, monkeypatch):
    import drama3_t8.creative_workspace as module
    first = project(workspace, lines=["你好"])
    path = workspace.projects / first["project_id"] / "project.json"
    original = path.read_bytes()
    real_replace = module.os.replace
    def fail_replace(source, target):
        if Path(target) == path:
            raise OSError("模拟原子提交失败")
        return real_replace(source, target)
    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        workspace.update_project(first["project_id"], {"name": "未提交"})
    assert path.read_bytes() == original
    assert list(path.parent.glob("*.pending"))
    assert workspace.get_project(first["project_id"])["revision"] == first["revision"]


def test_status_checkpoints_do_not_copy_full_history_per_candidate(workspace):
    first = project(workspace, lines=[f"第{i}句" for i in range(10)])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=3)
    assert len(done["generated"]) == 30
    history = list((workspace.projects / first["project_id"] / "history").glob("*.json"))
    assert len(history) == 1


def test_core_import_does_not_import_torch_or_comfy():
    code = "import sys; import drama3_t8.creative_workspace; assert 'torch' not in sys.modules; assert 'comfy' not in sys.modules"
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stderr


def test_other_process_cannot_edit_active_generation_project(workspace):
    first = project(workspace, lines=["你好"])
    code = ("from drama3_t8.creative_workspace import CreativeWorkspace; "
            "from drama3_t8.workspace_lock import WorkspaceBusy; "
            f"w=CreativeWorkspace({str(workspace.root)!r});\n"
            "try:\n"
            f" w.update_project({first['project_id']!r},{{'name':'bad'}})\n"
            "except WorkspaceBusy:\n raise SystemExit(13)\n"
            "raise SystemExit(0)\n")
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    with workspace._operation(first["project_id"]):
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, **kwargs)
    assert result.returncode == 13, result.stderr
    assert workspace.get_project(first["project_id"])["name"] == first["name"]


def test_quality_uses_native_spoken_transcript_not_direction_markup(workspace):
    first = project(workspace, lines=["<|speaker:0|> [angry] 别加班了"])
    def generate(job, directory):
        returned = media(job, directory)
        native_session(Path(returned["session_path"]), "别加班了")
        return returned
    workspace.run_generation(first["project_id"], generate, takes_per_line=1)
    reviewed = workspace.run_quality(first["project_id"], lambda _: {"transcript": "别加班了", "independent": True})
    assert next(iter(reviewed["project"]["candidates"].values()))["quality"]["text_check"]["normalized_match"]


def test_line_audio_is_byte_bound_portable_and_resolved_for_delivery(workspace, tmp_path):
    first = project(workspace, lines=["你好"])
    def split(job, directory):
        returned = media(job, directory)
        line_path = directory / "L01.wav"
        line_path.write_bytes(Path(returned["audio_path"]).read_bytes())
        returned["line_audio"] = [{"line_id": "L01", "audio_path": str(line_path), "start_seconds": 0, "end_seconds": .01}]
        return returned
    done = workspace.run_generation(first["project_id"], split, takes_per_line=2)
    candidate = done["project"]["lines"][0]["selected_candidate"]
    assert Path(candidate["line_audio"][0]["audio_path"]).is_file()
    assert candidate["line_audio"][0]["relative_audio_path"].startswith("artifacts/")
    assert "line:L01" in {artifact["kind"] for artifact in candidate["artifacts"]}
    path = tmp_path / "line.zip"
    workspace.export_project(first["project_id"], path)
    imported = workspace.import_project(path)
    assert Path(imported["lines"][0]["selected_candidate"]["line_audio"][0]["audio_path"]).is_file()
    Path(candidate["line_audio"][0]["audio_path"]).write_bytes(b"changed")
    assert workspace.plan_generation(first["project_id"], takes_per_line=1)["pending"] == 1


def test_line_audio_with_same_duration_but_wrong_pcm_rejected(workspace):
    first = project(workspace, lines=["你好"])
    def wrong(job, directory):
        returned = media(job, directory)
        line_path = directory / "L01.wav"
        with wave.open(str(line_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(struct.pack("<80h", *([222] * 80)))
        returned["line_audio"] = [{"line_id": "L01", "audio_path": str(line_path), "start_seconds": 0, "end_seconds": .01}]
        return returned
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], wrong, takes_per_line=1)


def test_real_variation_is_new_candidate_and_adoption_updates_script_explicitly(workspace):
    first = project(workspace, lines=["周一开工"])
    original = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    source_id = original["project"]["lines"][0]["selected_candidate_id"]
    source_audio = Path(original["project"]["candidates"][source_id]["audio_path"])
    old_bytes = source_audio.read_bytes()
    seen = []
    def revise(job, directory):
        seen.append(job)
        assert job["source_candidate"]["candidate_id"] == source_id
        assert Path(job["source_candidate"]["session_path"]).is_file()
        return media(job, directory)
    variation = workspace.run_variation(first["project_id"], "L01", source_id, revise,
                                        {"kind": "revise", "text": "周五放假"}, expected_revision=original["project"]["revision"])
    candidate_id = variation["candidate"]["candidate_id"]
    assert candidate_id != source_id and source_audio.read_bytes() == old_bytes
    assert variation["project"]["lines"][0]["original_text"] == "周一开工"
    assert variation["project"]["lines"][0]["selected_candidate_id"] == source_id
    assert seen[0]["readout_text"] == "周五放假"
    assert variation["candidate"]["variation"]["source_artifacts"]
    adopted = workspace.set_candidate_review(first["project_id"], "L01", candidate_id, selected=True, apply_text=True)
    assert adopted["lines"][0]["original_text"] == "周五放假"
    assert adopted["lines"][0]["subtitle_text"] == "周五放假"
    assert workspace.plan_generation(first["project_id"], takes_per_line=1)["pending"] == 0


def test_variation_retry_reuses_intact_candidate_and_export_keeps_source_chain(workspace, tmp_path):
    first = project(workspace, lines=["旧稿"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    source_id = done["generated"][0]
    operation = {"kind": "infill", "text": "新稿", "start_seconds": 0, "end_seconds": .005}
    changed = workspace.run_variation(first["project_id"], "L01", source_id, media, operation)
    reused = workspace.run_variation(first["project_id"], "L01", source_id, lambda *_: pytest.fail("variation regenerated"), operation)
    assert reused["reused"] and reused["generated"] == []
    workspace.set_candidate_review(first["project_id"], "L01", changed["candidate"]["candidate_id"], selected=True, apply_text=True)
    exported = workspace.export_project(first["project_id"], tmp_path / "variation.zip")
    assert set(exported["included_candidates"]) == {source_id, changed["candidate"]["candidate_id"]}


@pytest.mark.parametrize("operation", [{"kind": "infill", "text": "新", "start_seconds": .02, "end_seconds": .03},
                                       {"kind": "infill", "text": "新", "start_seconds": .005, "end_seconds": .001},
                                       {"kind": "bad", "text": "新"}])
def test_variation_invalid_operation_does_not_generate(workspace, operation):
    first = project(workspace, lines=["旧稿"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1)
    with pytest.raises(ValueError):
        workspace.run_variation(first["project_id"], "L01", done["generated"][0], lambda *_: pytest.fail("invalid variation generated"), operation)


def test_locked_selection_cannot_be_silently_replaced(workspace):
    first = project(workspace, lines=["你好"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=2)
    one, two = done["generated"]
    workspace.set_candidate_review(first["project_id"], "L01", one, locked=True)
    with pytest.raises(ValueError):
        workspace.set_candidate_review(first["project_id"], "L01", two, selected=True)
    adopted = workspace.set_candidate_review(first["project_id"], "L01", two, selected=True, locked=False)
    assert adopted["lines"][0]["selected_candidate_id"] == two


def test_windows_reader_contention_retries_only_atomic_commit(workspace, monkeypatch):
    import drama3_t8.creative_workspace as module
    if module.os.name != "nt":
        pytest.skip("Windows sharing retry")
    first = project(workspace, lines=["你好"])
    target = workspace.projects / first["project_id"] / "project.json"
    real_replace, calls, sleeps = module.os.replace, [], []
    def transient(source, destination):
        if Path(destination) == target:
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError("reader handle")
        return real_replace(source, destination)
    monkeypatch.setattr(module.os, "replace", transient)
    monkeypatch.setattr(module.time, "sleep", lambda delay: sleeps.append(delay))
    changed = workspace.update_project(first["project_id"], {"name": "提交成功"})
    assert changed["name"] == "提交成功" and len(calls) == 3
    assert sleeps == [.05, .1]


def test_windows_root_below_260_child_above_260_stays_in_workspace(tmp_path):
    if sys.platform != "win32":
        pytest.skip("Windows extended-prefix transition")
    base = str(tmp_path.resolve())
    root = tmp_path / ("w" * max(1, 238 - len(base) - 1))
    workspace = CreativeWorkspace(root)
    first = workspace.create_project("长路径工程", ["你好"])
    assert workspace.get_project(first["project_id"])["name"] == "长路径工程"


def test_selected_take_four_remains_actual_context_when_window_returns_to_three(workspace):
    first = project(workspace)
    done = workspace.run_generation(first["project_id"], media, takes_per_line=4, mode="per_line_context")
    fourth = done["project"]["lines"][0]["candidates"][3]
    workspace.set_candidate_review(first["project_id"], "L01", fourth["candidate_id"], selected=True)
    seen = []
    def record(job, directory):
        seen.append(job)
        return media(job, directory)
    continued = workspace.run_generation(first["project_id"], record, takes_per_line=3, mode="per_line_context")
    assert {job["line_id"] for job in seen} == {"L02"}
    assert all(job["context_candidate_ids"] == [fourth["candidate_id"]] for job in seen)
    assert continued["project"]["lines"][0]["selected_candidate_id"] == fourth["candidate_id"]
    # Repairing another target-window candidate must not replace selected Take4.
    first_take = continued["project"]["lines"][0]["candidates"][0]
    Path(first_take["audio_path"]).write_bytes(b"damaged")
    repaired = workspace.run_generation(first["project_id"], record, takes_per_line=3, mode="per_line_context")
    assert repaired["project"]["lines"][0]["selected_candidate_id"] == fourth["candidate_id"]
    workspace.set_candidate_review(first["project_id"], "L01", fourth["candidate_id"], locked=True)
    plan = workspace.plan_generation(first["project_id"], takes_per_line=3, mode="per_line_context")
    assert all(job["status"] == "locked" and job["selected_candidate"]["candidate_id"] == fourth["candidate_id"]
               for job in plan["jobs"] if job["line_id"] == "L01")


def test_selected_different_seed_variation_supplies_real_continuation_context(workspace):
    first = project(workspace)
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1, mode="per_line_context")
    changed = workspace.run_variation(first["project_id"], "L01", done["generated"][0], media,
                                     {"kind": "revise", "text": "新的台词"}, sampling={"seed": 999})
    variant = changed["candidate"]
    workspace.set_candidate_review(first["project_id"], "L01", variant["candidate_id"], selected=True, apply_text=True)
    plan = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context", line_ids=["L02"])
    second = next(job for job in plan["jobs"] if job["line_id"] == "L02")
    assert second["context_candidate_ids"] == [variant["candidate_id"]]
    assert second["context_candidates"][0]["seed"] == 999
    assert second["input_snapshot"]["context"][0]["session_sha256"] == next(
        item["sha256"] for item in variant["artifacts"] if item["kind"] == "session")
    # A real non-seed sampling change invalidates this contextual compatibility.
    changed_sampling = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context",
                                                 line_ids=["L02"], sampling={"temperature": .9})
    assert next(job for job in changed_sampling["jobs"] if job["line_id"] == "L02")["status"] == "blocked"


def test_uint64_seed_text_is_exact_for_browser_views(workspace):
    first = project(workspace, lines=["你好"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1, base_seed=2**64 - 1)
    candidate = done["project"]["lines"][0]["selected_candidate"]
    assert candidate["seed_text"] == candidate["requested_seed_text"] == "18446744073709551615"
    assert workspace.plan_generation(first["project_id"], takes_per_line=1, base_seed=2**64 - 1)["jobs"][0]["seed_text"] == "18446744073709551615"


def test_direct_context_dependencies_are_bounded_and_causal(workspace):
    first = project(workspace, lines=[{"role": f"角色{i%4}", "text": f"第{i}句"} for i in range(200)])
    plan = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context")
    counts = [len(job["input_snapshot"]["context"]) for job in plan["jobs"]]
    assert sum(counts) == 587 and max(counts) == 3
    # A real regenerated context ancestor changes its Session SHA, and that
    # change propagates down the bounded direct graph rather than being lost.
    shorter = project(workspace, lines=[{"role": f"角色{i%3}", "text": f"第{i}句"} for i in range(8)])
    done = workspace.run_generation(shorter["project_id"], media, takes_per_line=1, mode="per_line_context")
    lines = done["project"]["lines"]
    lines[0]["original_text"] = "改掉第一句"
    workspace.update_project(shorter["project_id"], {"lines": lines})
    continued = workspace.run_generation(shorter["project_id"], media, takes_per_line=1, mode="per_line_context")
    assert len(continued["generated"]) == 8


def test_global_speaker_map_binds_only_effective_lookahead_roles(workspace):
    cast = {"甲": {"speaker": 0}, "乙": {"speaker": 1}, "丙": {"speaker": 2}}
    first = project(workspace, lines=[{"role": role, "text": role} for role in cast], cast=cast)
    before = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="text_stream", lookahead=1)["jobs"][0]
    assert before["speaker_map"] == {"甲": 0, "乙": 1, "丙": 2}
    assert before["input_snapshot"]["speaker_map"] == {"甲": 0, "乙": 1}
    workspace.update_project(first["project_id"], {"cast": {**cast, "丙": {"speaker": 3}}})
    unrelated = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="text_stream", lookahead=1)["jobs"][0]
    assert unrelated["fingerprint"] == before["fingerprint"]
    workspace.update_project(first["project_id"], {"cast": {**cast, "乙": {"speaker": 3}}})
    changed = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="text_stream", lookahead=1)["jobs"][0]
    assert changed["fingerprint"] != before["fingerprint"]


def test_shared_atomic_candidate_is_expanded_once_per_project_view(workspace):
    first = project(workspace, lines=["一", "二", "三"])
    done = workspace.run_generation(first["project_id"], media, takes_per_line=1, mode="together")
    view = done["project"]
    candidate = view["candidates"][done["generated"][0]]
    assert all(line["candidates"][0] is candidate and line["selected_candidate"] is candidate for line in view["lines"])
    # The returned view is still independent of the durable document.
    candidate["notes"] = "只改视图"
    assert workspace.get_project(first["project_id"])["candidates"][candidate["candidate_id"]]["notes"] == ""


def conditioning_result(job, directory, *, wrong=False):
    from drama3_t8.creative_workspace import _file_record, _native_session_info
    returned = media(job, directory)
    info = _native_session_info(Path(returned["session_path"]))
    binding = {"audio_sha256": _file_record(Path(returned["audio_path"]))["sha256"],
               "source_take_id": info["take_id"], "line_id": job["line_id"],
               "codec_identity": job["model_identity"].get("codec")}
    if wrong:
        binding["audio_sha256"] = "0" * 64
    condition = directory / "conditioning.safetensors"
    native_session(condition, job["readout_text"], seed=job["seed"], requested_seed=job["seed"], from_source_view=binding)
    return {**returned, "conditioning_session_path": str(condition)}


def test_actual_pcm_condition_archive_is_bound_used_and_portable(workspace, tmp_path):
    first = project(workspace)
    identity = {"codec": {"sha256": "a" * 64, "bytes": 10}}
    done = workspace.run_generation(first["project_id"], conditioning_result, takes_per_line=1, mode="per_line_context", model_identity=identity)
    candidate = done["project"]["lines"][0]["selected_candidate"]
    assert Path(candidate["conditioning_session_path"]).is_file()
    conditioning_sha = next(artifact["sha256"] for artifact in candidate["artifacts"] if artifact["kind"] == "conditioning")
    source_sha = next(artifact["sha256"] for artifact in candidate["artifacts"] if artifact["kind"] == "session")
    job = workspace.plan_generation(first["project_id"], takes_per_line=1, mode="per_line_context", model_identity=identity)["jobs"][1]
    assert job["input_snapshot"]["context"][0]["session_sha256"] == conditioning_sha
    assert job["input_snapshot"]["context"][0]["source_session_sha256"] == source_sha
    path = tmp_path / "conditioning.zip"
    workspace.export_project(first["project_id"], path)
    imported = workspace.import_project(path)
    assert Path(imported["lines"][0]["selected_candidate"]["conditioning_session_path"]).is_file()
    assert workspace.plan_generation(imported["project_id"], takes_per_line=1, mode="per_line_context", model_identity=identity)["pending"] == 0


def test_wrong_pcm_condition_binding_never_becomes_completed(workspace):
    first = project(workspace, lines=["单句"])
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], lambda job, directory: conditioning_result(job, directory, wrong=True), takes_per_line=1)


def test_native_condition_binary_codes_are_validated_not_just_header(workspace):
    first = project(workspace, lines=["单句"])
    def invalid(job, directory):
        returned = conditioning_result(job, directory)
        path = Path(returned["conditioning_session_path"])
        data = bytearray(path.read_bytes())
        data[-8:] = struct.pack("<q", 5000)
        path.write_bytes(data)
        return returned
    with pytest.raises(InvalidCheckpoint):
        workspace.run_generation(first["project_id"], invalid, takes_per_line=1)
