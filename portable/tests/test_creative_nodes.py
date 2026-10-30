"""Actual V3 schemas and in-process creative-core node contracts on CPU."""
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from types import SimpleNamespace
import wave

import pytest
import torch

from drama3_t8.bootstrap import standalone_engine
standalone_engine()
from comfy.cli_args import args
args.cpu = True
from comfy_api.latest import io
import folder_paths

from drama3_t8 import creative_nodes
from drama3_t8.creative_workspace import CreativeWorkspace
from drama3_t8.state import Reference, Session, Take
from drama3_t8.voice_library import VoiceLibrary


CLASSES = creative_nodes.get_creative_nodes(io.Custom("DRAMA3_TEST_ENGINE"), io.Custom("DRAMA3_TEST_REFS"), io.Custom("DRAMA3_TEST_SESSION"))
NODES = {node.__name__: node for node in CLASSES}


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for name in ("user", "input", "output", "models"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(folder_paths, "get_user_directory", lambda: str(tmp_path / "user"))
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "models"))
    monkeypatch.delenv("DRAMA3_VOICE_LIBRARY", raising=False)
    # Kitchen 0.2.40 probes device capabilities at attention import, even with
    # Comfy's cpu flag. These contract tests deliberately expose no CUDA device.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tmp_path


def add_voice(path, name="主角", code=3):
    library = VoiceLibrary(path)
    voice = library.create_voice(name)
    refs = Session((Reference("<|speaker:0|> 参考台词", torch.full((10, 4), code, dtype=torch.int16)),))
    version = library.add_version(voice["id"], refs)
    return library, voice, version, refs


def test_all_v3_schemas_have_stable_unique_ids_and_native_sockets(paths):
    ids = []
    for node in CLASSES:
        info = node.GET_NODE_INFO_V1()
        ids.append(info["name"])
        assert info["category"] == "T8/Drama3/Creative"
    assert len(ids) == len(set(ids)) == 12
    load = NODES["T8DramaVoiceLibraryLoad"].GET_NODE_INFO_V1()
    assert load["output"][:2] == ["DRAMA3_TEST_REFS", "DRAMA3_TEST_SESSION"]
    batch = NODES["T8DramaCreativeBatch"].GET_NODE_INFO_V1()
    assert batch["input"]["required"]["engine"][0] == "DRAMA3_TEST_ENGINE"
    assert batch["input"]["required"]["seed_text"][0] == "STRING"


def test_default_data_paths_are_outside_plugin_sources(paths):
    assert creative_nodes._workspace_path().is_relative_to(paths / "user")
    assert creative_nodes._library_path().is_relative_to(paths / "user")
    with pytest.raises(ValueError):
        creative_nodes._workspace_path(str(Path(creative_nodes.__file__).parent / "bad-data"))


def test_native_host_preserves_actual_r2t2_paths_and_scene_head_settings(paths):
    engine = SimpleNamespace(r2t2_paths={key: paths / "models" / key for key in ("r2t2_model", "r2t2_projector", "r2t2_vad")},
                             scene_head_mode="full")
    host = creative_nodes._NativeHost(engine)
    assert all(host.model_paths[key] == value for key, value in engine.r2t2_paths.items())
    assert host.settings["scene_head_mode"] == "full"


def test_project_node_fingerprint_binds_actual_private_reference_bytes(paths):
    library, voice, version, _ = add_voice(paths / "voices")
    cast = {"主角": {"voice_id": voice["id"], "version_id": version["id"], "speaker": 0}}
    handle = NODES["T8DramaCreativeProjectCreate"].execute("工程", '[{"role":"主角","text":"你好"}]',
        cast_json=json.dumps(cast), workspace_path=str(paths / "workspace"), library_path=str(library.root))[0]
    bundle = json.loads(NODES["T8DramaCreativeProjectExport"].execute(handle)[0])["path"]
    imported = NODES["T8DramaCreativeProjectImport"].execute(bundle, workspace_path=str(paths / "destination"))[0]
    from drama3_t8.creative_bundle import project_voice_library
    workspace = CreativeWorkspace(imported["workspace_path"])
    private = project_voice_library(workspace, imported["project_id"], library)
    node = NODES["T8DramaCreativeProjectLoad"]
    before = node.fingerprint_inputs(imported["project_id"], imported["workspace_path"])
    target = private.session_file(version["id"])
    info = target.stat()
    content = bytearray(target.read_bytes())
    content[-1] ^= 1
    target.write_bytes(content)
    os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
    after = node.fingerprint_inputs(imported["project_id"], imported["workspace_path"])
    assert before != after and before["voices"] != after["voices"]


def test_voice_list_load_name_uuid_and_immutable_version_with_custom_path(paths):
    library, voice, version, refs = add_voice(paths / "custom-voices")
    listed = NODES["T8DramaVoiceLibraryList"].execute(str(library.root))
    assert json.loads(listed[0])["voices"][0]["id"] == voice["id"]
    loaded = NODES["T8DramaVoiceLibraryLoad"].execute("主角", str(library.root), version["id"], speaker=2)
    assert loaded[1].references is loaded[0]
    assert loaded[0][0].transcript.startswith("<|speaker:2|>")
    assert torch.equal(loaded[0][0].codes, refs.references[0].codes)
    another = NODES["T8DramaVoiceLibraryLoad"].execute(voice["id"], str(library.root), version["id"])
    assert json.loads(another[2])["id"] == version["id"]
    add_voice(library.root, name="主角", code=4)
    with pytest.raises(ValueError, match="UUID"):
        NODES["T8DramaVoiceLibraryLoad"].execute("主角", str(library.root))


def test_voice_fingerprint_observes_real_reference_bytes_even_same_size_mtime(paths):
    library, voice, version, _ = add_voice(paths / "voices")
    node = NODES["T8DramaVoiceLibraryLoad"]
    before = node.fingerprint_inputs(voice["id"], str(library.root), version["id"])
    path = library.session_file(version["id"])
    info = path.stat()
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert node.fingerprint_inputs(voice["id"], str(library.root), version["id"]) != before
    with pytest.raises(ValueError):
        node.execute(voice["id"], str(library.root), version["id"])


def test_save_reference_session_and_load_cast_native(paths):
    library, voice, version, refs = add_voice(paths / "voices")
    saved = NODES["T8DramaVoiceLibrarySave"].execute(refs, mode="new_version", voice=voice["id"], library_path=str(library.root))
    assert saved[0] == voice["id"] and saved[1] != version["id"]
    cast = library.create_cast("喜剧组合", {"主角": {"voice_id": voice["id"], "version_id": saved[1], "speaker": 0}})
    loaded = NODES["T8DramaVoiceCastLoad"].execute(cast["id"], str(library.root))
    assert loaded[0] is loaded[1].references
    assert json.loads(loaded[2])["主角"]["version_id"] == saved[1]


def test_cast_builder_connects_saved_version_to_project_and_checks_ownership(paths):
    library, voice, version, refs = add_voice(paths / "voices")
    saved = NODES["T8DramaVoiceLibrarySave"].execute(refs, mode="new_version", voice=voice["id"], library_path=str(library.root))
    build = NODES["T8DramaVoiceCastBuild"]
    cast = build.execute(saved[0], saved[1], "主角", 2, str(library.root), reference_groups_json=json.dumps({"平静": [version["id"]]}))[0]
    handle, document = NODES["T8DramaCreativeProjectCreate"].execute("工程", '[{"role":"主角","text":"你好"}]',
        cast_json=cast, workspace_path=str(paths / "workspace"), library_path=str(library.root))
    assert json.loads(document)["cast"]["主角"]["version_id"] == saved[1]
    assert NODES["T8DramaCreativeProjectLoad"].fingerprint_inputs(handle["project_id"], handle["workspace_path"], str(library.root))["voices"]
    _, other, foreign, _ = add_voice(library.root, "另一声音", 4)
    with pytest.raises(ValueError, match="绑定音色"):
        build.execute(voice["id"], foreign["id"], library_path=str(library.root))


def test_text_preview_and_project_create_load_share_workspace(paths):
    node = NODES["T8DramaCreativeTextPreview"]
    original, readout, subtitle, _ = node.execute("Drama3有12个案例", '{"aliases":{"Drama3":"抓马三"},"number_mode":"cardinal"}')
    assert original == subtitle and readout == "抓马三有十二个案例"
    created = NODES["T8DramaCreativeProjectCreate"].execute("工程", '["你好"]', workspace_path=str(paths / "workspace"))
    project = created[0]
    loaded = NODES["T8DramaCreativeProjectLoad"].execute(project["project_id"], project["workspace_path"])
    assert loaded[0]["project_id"] == project["project_id"]
    assert json.loads(loaded[1])["lines"][0]["readout_text"] == "你好"


def test_project_create_assigns_unreferenced_roles_over_the_whole_script(paths):
    lines = [{"role": "导演", "text": "一"}, {"role": "猫", "text": "二"}, {"role": "导演", "text": "三"}]
    output = NODES["T8DramaCreativeProjectCreate"].execute("多人", json.dumps(lines), workspace_path=str(paths / "workspace"))
    document = json.loads(output[1])
    assert document["cast"]["导演"]["speaker"] == 0
    assert document["cast"]["猫"]["speaker"] == 1


class FakeEngine:
    """CPU engine contract double; this is not model-generation validation."""
    def __init__(self, root):
        for attribute in ("model_path", "codec_path", "text_head_path", "line_head_path"):
            path = root / (attribute + ".safetensors")
            path.write_bytes(attribute.encode())
            setattr(self, attribute, path)
        self.drama = self.codec = None
        self.gdn_backend = self.moe_backend = "torch"
        self.expert_placement, self.expert_cache_gb, self.cuda_graph = "ram", 0, False
        self.calls, self.loads, self.unloads = [], 0, 0

    def load_model(self):
        if self.drama is None:
            self.loads += 1
            self.drama = SimpleNamespace(last_graph_stats={})
        return self.drama

    @contextmanager
    def generation_session(self):
        assert self.drama is not None
        yield

    def speak(self, text, session, sampling, on_codes=None, progress=None):
        from drama3_t8.prompts import spoken_text
        self.calls.append(sampling.seed)
        frames = torch.zeros((2, 11), dtype=torch.int64)
        frames[:, 0] = 248099
        take = Take(text, spoken_text(text), frames, metadata={"seed": sampling.seed, "requested_seed": sampling.seed,
            "finish_reason": "eos", "length_limit_reached": False, "gdn_calls_by_backend": {"torch": 2}})
        if on_codes:
            on_codes(frames[0])
        return session.append(take)

    def decode_take(self, take, with_timeline=False):
        from drama3_t8.native.codec import FRAME_LENGTH
        audio = {"waveform": torch.full((1, 1, 2 * FRAME_LENGTH), .1), "sample_rate": 44100}
        return (audio, []) if with_timeline else audio

    def expert_stats(self):
        return {"actual_backend": "torch", "placement": "ram", "fallback_reason": None}

    def unload(self):
        self.unloads += 1
        self.drama = self.codec = None


def test_native_batch_uses_supplied_engine_and_shared_checkpoints_without_network_or_queue(paths, monkeypatch):
    from drama3_t8.creative_studio import CreativeStudio
    monkeypatch.setattr(CreativeStudio, "source_identity", lambda self: "fixture-source")
    monkeypatch.setattr(CreativeStudio, "performance_binding", lambda self, models=None: {"machine": "cpu-contract", "source": "fixture"})
    monkeypatch.setattr(socket, "create_connection", lambda *_a, **_k: pytest.fail("network bridge attempted"))
    before_threads = {thread.name for thread in __import__("threading").enumerate()}
    created = NODES["T8DramaCreativeProjectCreate"].execute("工程", '["你好","再见"]', workspace_path=str(paths / "workspace"))
    engine = FakeEngine(paths / "models")
    batch = NODES["T8DramaCreativeBatch"]
    result = batch.execute(engine, created[0], takes_per_line=2, seed_text="18446744073709551615")
    assert len(engine.calls) == 4 and engine.calls[0] == 2**64 - 1
    assert isinstance(result[1], Session) and result[1].takes
    workspace = CreativeWorkspace(created[0]["workspace_path"])
    project = workspace.get_project(created[0]["project_id"])
    assert len(project["candidates"]) == 4
    assert all(candidate["status"] == "completed" for candidate in project["candidates"].values())
    assert all(candidate["line_audio"] for candidate in project["candidates"].values())
    batch.execute(engine, created[0], takes_per_line=2, seed_text="18446744073709551615")
    assert len(engine.calls) == 4
    assert not ({thread.name for thread in __import__("threading").enumerate()} - before_threads) & {"Drama-GPU-queue"}
    assert not torch.cuda.is_initialized()


def test_project_fingerprint_observes_candidate_file_change(paths, monkeypatch):
    from drama3_t8.creative_studio import CreativeStudio
    monkeypatch.setattr(CreativeStudio, "source_identity", lambda self: "fixture-source")
    monkeypatch.setattr(CreativeStudio, "performance_binding", lambda self, models=None: {"machine": "cpu-contract"})
    handle = NODES["T8DramaCreativeProjectCreate"].execute("工程", '["你好"]', workspace_path=str(paths / "workspace"))[0]
    NODES["T8DramaCreativeBatch"].execute(FakeEngine(paths / "models"), handle, takes_per_line=1)
    node = NODES["T8DramaCreativeProjectLoad"]
    before = node.fingerprint_inputs(handle["project_id"], handle["workspace_path"])
    document = CreativeWorkspace(handle["workspace_path"]).get_project(handle["project_id"])
    audio = Path(document["lines"][0]["selected_candidate"]["audio_path"])
    info = audio.stat()
    content = bytearray(audio.read_bytes())
    content[-1] ^= 1
    audio.write_bytes(content)
    os.utime(audio, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert node.fingerprint_inputs(handle["project_id"], handle["workspace_path"]) != before


def test_native_project_bundle_and_delivery_have_real_local_files(paths, monkeypatch):
    from drama3_t8.creative_studio import CreativeStudio
    monkeypatch.setattr(CreativeStudio, "source_identity", lambda self: "fixture-source")
    monkeypatch.setattr(CreativeStudio, "performance_binding", lambda self, models=None: {"machine": "cpu-contract"})
    handle = NODES["T8DramaCreativeProjectCreate"].execute("工程", '["你好"]', workspace_path=str(paths / "workspace"))[0]
    NODES["T8DramaCreativeBatch"].execute(FakeEngine(paths / "models"), handle, takes_per_line=1)
    delivery = NODES["T8DramaCreativeDelivery"].execute(handle)
    assert Path(delivery[0]).is_file()
    assert json.loads(delivery[1])["master"] == "master.wav"
    exported = NODES["T8DramaCreativeProjectExport"].execute(handle)
    bundle = json.loads(exported[0])["path"]
    assert Path(bundle).is_file()
    imported = NODES["T8DramaCreativeProjectImport"].execute(bundle, workspace_path=str(paths / "destination"))
    assert imported[0]["project_id"] != handle["project_id"]
    assert json.loads(imported[1])["lines"][0]["selected_candidate"]


def test_inspection_json_keeps_rich_atomic_candidate_once(paths):
    identity = "a" * 32
    candidate = {"candidate_id": identity, "status": "completed", "seed_text": "42",
                 "input_snapshot": {"marker": "unique-large-candidate-details", "lines": ["one", "two"]}}
    line = {"line_id": "L01", "candidate_ids": [identity], "candidates": [candidate], "selected_candidate": candidate}
    project = {"lines": [line, {**line, "line_id": "L02"}], "candidates": {identity: candidate}}
    text = creative_nodes._project_json(project)
    parsed = json.loads(text)
    assert text.count("unique-large-candidate-details") == 1
    assert parsed["lines"][0]["candidates"] == [identity]
    assert parsed["lines"][0]["selected_candidate"]["status"] == "completed"


def _creative_canvases():
    folder = Path(__file__).resolve().parents[1] / "comfyui-drama3-T8/workflows"
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in folder.glob("workflow_creative_*.json")}


def test_three_shipped_creative_examples_are_canvas_and_links_match_actual_schema(paths):
    from tools import make_workflows as maker
    infos = maker.node_infos()
    graphs = _creative_canvases()
    assert len(graphs) == 3
    for graph in graphs.values():
        assert graph["version"] == .4 and "prompt" not in graph and graph["nodes"]
        nodes = {node["id"]: node for node in graph["nodes"]}
        links = {link[0]: link for link in graph["links"]}
        assert graph["last_node_id"] == max(nodes) and graph["last_link_id"] == len(links)
        for node in nodes.values():
            info = infos[node["type"]]
            specs = dict(maker.input_specs(info))
            assert len(node["widgets_values"]) == len(maker.widget_names(info))
            assert [item["type"] for item in node["outputs"]] == info["output"]
            expected = [name for name, spec in maker.input_specs(info) if not maker.is_widget(spec)]
            assert [item["name"] for item in node["inputs"] if "widget" not in item] == expected
            for slot, item in enumerate(node["inputs"]):
                spec = specs[item["name"]]
                if "widget" in item:
                    assert maker.is_widget(spec) and item["widget"] == {"name": item["name"]}
                assert item["type"] == spec[0]
                if item["link"] is None:
                    assert item["name"] in info["input"].get("optional", {})
                    continue
                link = links[item["link"]]
                assert link[3:5] == [node["id"], slot]
                origin = nodes[link[1]]["outputs"][link[2]]
                assert link[5] == origin["type"] == item["type"]
                assert link[0] in origin["links"]


def test_creative_canvas_widget_values_restore_in_installed_frontend_order(paths, tmp_path):
    from tools import make_workflows as maker
    frontend = importlib.util.find_spec("comfyui_frontend_package")
    executable = shutil.which("node")
    if frontend is None or executable is None:
        pytest.skip("Installed frontend and Node.js are needed for widget restoration")
    payload = {"assets": str(Path(frontend.origin).parent / "static/assets"), "infos": maker.node_infos(),
               "nodes": [{**node, "id": f"{name}:{node['id']}"} for name, graph in _creative_canvases().items() for node in graph["nodes"]]}
    request = tmp_path / "frontend-request.json"
    request.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    oracle = Path(__file__).with_name("frontend_canvas_oracle.mjs")
    process = subprocess.run([executable, str(oracle), str(request)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    actual = {node["id"]: node for node in json.loads(process.stdout)["nodes"]}
    for node in payload["nodes"]:
        result = actual[node["id"]]
        names = [widget["name"] for widget in result["widgets"]]
        assert len(names) == len(node["widgets_values"])
        expected = dict(zip(names, node["widgets_values"]))
        assert result["restored"] == {widget["name"]: expected[widget["name"]] for widget in result["widgets"] if widget["serialize"]}
