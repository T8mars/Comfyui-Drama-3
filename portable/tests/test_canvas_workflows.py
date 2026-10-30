"""Validate shipped canvas JSON against runtime schemas and the real frontend."""
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from copy import deepcopy

import pytest

from tools import make_workflows as maker

FIXTURES = Path(__file__).parent / "fixtures/seed-workflows-v0.1.11"
# Published examples target these public assets. Their dropdown lists are
# discovered from each installation, but the published asset selectors stay
# deterministic before installation. No local filenames or inventory-dependent
# placeholders may enter distributed examples.
PUBLIC_MODEL_SELECTIONS = {
    ("T8DramaLoader", "model"): "drama3-t8-fp8.safetensors",
    ("T8DramaLoader", "codec"): "drama3-codec.safetensors",
    ("T8DramaTranscribe", "asr_model"): "qwen3-asr-06b.safetensors",
    ("T8DramaAlign", "aligner"): "qwen3_forced_aligner_0.6b_bf16.safetensors",
}


def published_nodes():
    classes = set(json.loads((FIXTURES / "node-info-v1.json").read_text(encoding="utf-8")))
    result = {}
    for path in FIXTURES.glob("workflow_*.json"):
        for node in json.loads(path.read_text(encoding="utf-8"))["nodes"]:
            if node["type"] in classes:
                result[f"v011:{path.stem}:{node['id']}"] = node
    return result


@pytest.fixture(scope="module")
def generated_workflows():
    return maker.make_workflows()


def shipped_workflows():
    return {path.stem.removeprefix("workflow_"): json.loads(path.read_text(encoding="utf-8"))
        for path in maker.FOLDER.glob("*.json")}


@pytest.fixture(scope="module", params=["generated", "shipped"])
def workflows(request, generated_workflows):
    # Validate the actual distributed files through the official frontend too,
    # including hosts whose local model inventory differs from the publisher's.
    return generated_workflows if request.param == "generated" else shipped_workflows()


@pytest.fixture(scope="module")
def oracle(tmp_path_factory, workflows):
    node = shutil.which("node")
    frontend = importlib.util.find_spec("comfyui_frontend_package")
    if not node or not frontend:
        pytest.skip("The installed ComfyUI frontend and Node.js are needed for frontend restoration validation")
    request = tmp_path_factory.mktemp("canvas-oracle") / "request.json"
    seen = {}
    # Unique ids permit all graphs to be checked in one lightweight frontend run.
    for graph_name, workflow in workflows.items():
        for canvas_node in workflow["nodes"]:
            seen[f"{graph_name}:{canvas_node['id']}"] = {**canvas_node, "id": f"{graph_name}:{canvas_node['id']}"}
    infos = {**maker.node_infos()}
    published_infos = json.loads((FIXTURES / "node-info-v1.json").read_text(encoding="utf-8"))
    for key, canvas_node in published_nodes().items():
        seen[key] = {**canvas_node, "id": key, "roundtripChoice": "decrement"}
        alias = "PublishedV011:" + canvas_node["type"]
        infos[alias] = published_infos[canvas_node["type"]]
        seen["original:" + key] = {**canvas_node, "id": "original:" + key, "type": alias}
    original = next(node for node in published_nodes().values() if node["type"] == "T8DramaGenerate")
    layout = maker.widget_names(published_infos["T8DramaGenerate"])
    precision = deepcopy(original)
    precision["id"] = "precision"
    precision["widgets_values"][layout.index("seed")] = 2**53 - 1
    precision["widgets_values"][layout.index("seed_text")] = " 18446744073709551615 "
    seen["precision"] = precision
    prior = deepcopy(original)
    prior.update(id="prior-hook", priorChoice="increment")
    seen["prior-hook"] = prior
    invalid = deepcopy(original)
    invalid["id"] = "invalid-control"
    invalid["widgets_values"][layout.index("control_after_generate")] = "not-a-control"
    seen["invalid-control"] = invalid
    before_chunks=json.loads((Path(__file__).parent/'fixtures/scene-before-chunk-controls.json').read_text(encoding='utf-8'))
    scene=deepcopy(next(node for node in before_chunks['nodes'] if node['type']=='T8DramaScene'))
    scene['id']='before-scene-chunks'
    scene['widgets_values'][-1]='per_speaker'
    seen[scene['id']]=scene
    # Preserve the exact shipped10-widget Scene node as a version-bound old
    # fixture; do not regenerate it to conceal migration of linked controls.
    before_sampling=json.loads((Path(__file__).parent/'fixtures/scene-before-sampling-controls.json').read_text(encoding='utf-8'))
    old_scene=deepcopy(before_sampling)
    old_scene.update(id='before-scene-sampling',roundtripChoice='increment')
    old_scene['widgets_values'][3:10]=[2**53-1,'decrement',1200,'18446744073709551615','per_line_context',True,48]
    seen[old_scene['id']]=old_scene
    current_scene=deepcopy(before_sampling)
    current_scene['id']='current-scene-sampling'
    current_scene['widgets_values']=maker.widget_values('T8DramaScene',dict(script='A: current',seed=42,
        seed_text='18446744073709551615',control_after_generate='decrement',generation_mode='per_speaker',
        split_long_text=True,chunk_bytes=111,temperature=1.2,top_p=.8,top_k=0,opening_retries=1,sampling_preset='custom'))
    seen[current_scene['id']]=current_scene
    request.write_text(json.dumps({"assets": str(Path(frontend.origin).parent / "static/assets"),
        "infos": infos, "nodes": list(seen.values()), "seedCompatibility": str(maker.ROOT / "web/seed-compat.js")}), encoding="utf-8")
    result = subprocess.run([node, str(Path(__file__).with_name("frontend_canvas_oracle.mjs")), str(request)],
        capture_output=True, text=True, encoding="utf-8", timeout=90, check=True)
    return {item["id"]: item for item in json.loads(result.stdout)["nodes"]}


def test_all_original_examples_remain_and_editor_is_importable_canvas(workflows):
    original = {"basic", "long", "stream", "revision", "reference", "scene", "infill", "srt",
                "codec", "asr", "align", "r2t2", "live", "save_session", "load_session"}
    assert set(workflows) == original | {"editor", "creative_voice_project", "creative_bundle_resume", "creative_project_resume"}
    for workflow in workflows.values():
        assert workflow["version"] == .4 and isinstance(workflow["nodes"], list)
        assert isinstance(workflow["links"], list) and "prompt" not in workflow
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == len(workflow["links"])


def test_every_canvas_link_matches_actual_input_type_and_output_slot(workflows, oracle):
    infos = maker.node_infos()
    for graph_name, workflow in workflows.items():
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        assert len(nodes) == len(workflow["nodes"]) and len(links) == len(workflow["links"])
        for node in nodes.values():
            actual = oracle[f"{graph_name}:{node['id']}"]
            assert [(item["name"], item["type"]) for item in node["inputs"]] == [
                (item["name"], item["type"]) for item in actual["sockets"]]
            assert [output["type"] for output in node["outputs"]] == infos[node["type"]]["output"]
            for slot, socket in enumerate(node["inputs"]):
                if socket["link"] is None:
                    assert actual["sockets"][slot]["isOptional"], f"Missing required input {node['type']}.{socket['name']}"
                    continue
                link = links[socket["link"]]
                assert link[3:5] == [node["id"], slot]
                source = nodes[link[1]]
                assert link[0] in source["outputs"][link[2]]["links"]
                source_type = infos[source["type"]]["output"][link[2]]
                assert link[5] == source_type and (socket["type"] == "*" or socket["type"] == source_type)
            for slot, output in enumerate(node["outputs"]):
                assert output["links"] == [link[0] for link in workflow["links"] if link[1:3] == [node["id"], slot]]


def test_saved_reference_canvas_feeds_scene_from_appended_output_and_saves_voice_only_session(workflows):
    graph=workflows['load_session']
    by_id={node['id']:node for node in graph['nodes']}
    loader=next(node for node in graph['nodes'] if node['type']=='T8DramaLoadSession')
    reference_links=[link for link in graph['links'] if link[1:3]==[loader['id'],1]]
    assert len(reference_links)==1
    link=reference_links[0];target=by_id[link[3]]
    assert target['type']=='T8DramaScene' and target['inputs'][link[4]]['name']=='references'
    assert link[5]=='DRAMA3_T8_REFERENCES' and loader['outputs'][0]['type']=='DRAMA3_T8_SESSION'
    # Reference-only archives have no take; this example decodes the new Scene,
    # never an empty loaded Session.
    assert not any(link[1:3]==[loader['id'],0] for link in graph['links'])
    voice=workflows['reference'];by_id={node['id']:node for node in voice['nodes']}
    saved=next(node for node in voice['nodes'] if node['type']=='T8DramaSaveSession')
    incoming=next(link for link in voice['links'] if link[3]==saved['id'])
    assert by_id[incoming[1]]['type']=='T8DramaSession'


def test_widget_defaults_restore_by_actual_frontend_order_without_offset(workflows, oracle):
    infos = maker.node_infos()
    for graph_name, workflow in workflows.items():
        for node in workflow["nodes"]:
            actual = oracle[f"{graph_name}:{node['id']}"]
            widget_names = [widget["name"] for widget in actual["widgets"]]
            assert len(node["widgets_values"]) == len(widget_names)
            expected = dict(zip(widget_names, node["widgets_values"]))
            # New frontends do not serialize their linked seed control; the
            # official legacy migration must still restore every model input.
            assert actual["restored"] == {widget["name"]: expected[widget["name"]]
                for widget in actual["widgets"] if widget["serialize"]}
            for group in ("required", "optional"):
                for name, raw in infos[node["type"]]["input"].get(group, {}).items():
                    if name not in actual["restored"]:
                        continue
                    value = actual["restored"][name]
                    kind, options = raw[0], raw[1] if len(raw) > 1 else {}
                    if kind == "BOOLEAN":
                        assert type(value) is bool
                    elif kind == "INT":
                        assert type(value) is int
                    elif kind == "FLOAT":
                        assert type(value) in (int, float) and math.isfinite(value)
                    elif kind == "STRING":
                        assert isinstance(value, str)
                    choices = kind if isinstance(kind, list) else options.get("options")
                    requires_selection = (node["type"], name) in {("LoadAudio", "audio"), ("T8DramaLoadSession", "session_file")}
                    if requires_selection:
                        assert value == ""  # Never distribute an author's private audio/session filename.
                    elif choices and value not in choices:
                        # Missing public model files still restore as their
                        # exact published names. Running requires installing
                        # those files or choosing/filling the user's own path.
                        assert (node["type"], name) in PUBLIC_MODEL_SELECTIONS
                        assert value == PUBLIC_MODEL_SELECTIONS[(node["type"], name)]
                    if type(value) in (int, float):
                        assert options.get("min", -math.inf) <= value <= options.get("max", math.inf)


def test_example_seed_values_and_legacy_control_positions_are_exact(workflows, oracle):
    for graph_name, workflow in workflows.items():
        for node in workflow["nodes"]:
            actual = oracle[f"{graph_name}:{node['id']}"]
            names = [widget["name"] for widget in actual["widgets"]]
            if "seed" in names:
                values = dict(zip(names, node["widgets_values"]))
                assert values["control_after_generate"] == "fixed"
                assert 0 <= values["seed"] <= 2**53 - 1 and type(values["seed"]) is int
                assert values["seed_text"] == "" and actual["restored"]["seed"] == values["seed"]
                compatible = actual["compatibility"]
                assert compatible["configuredControl"] == "fixed"
                assert compatible["configuredValues"]["seed"] == values["seed"]
    # Runtime metadata must preserve the old widget prefix, including seed_text.
    generate = maker.node_infos()["T8DramaGenerate"]
    optional = generate["input_order"]["optional"]
    assert optional.index("seed_text") < optional.index("opening_retries") < optional.index("sampling_preset")


def test_published_v011_fixtures_are_version_bound_and_restored_without_corrupting_seed_text(oracle):
    source = json.loads((FIXTURES / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["commit"] == "f838deaee67f2bce467ce6a76e9b3dc3de615a86"
    for name, checksum in source["files"].items():
        # Historical checksums captured CRLF working-tree bytes. Rebuild exactly
        # that framing after either a Git LF checkout or a Windows CRLF checkout.
        assert hashlib.sha256((FIXTURES / name).read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest() == checksum
    for key, node in published_nodes().items():
        original = oracle["original:" + key]
        names = [widget["name"] for widget in original["widgets"]]
        expected = dict(zip(names, node["widgets_values"]))
        compatible = oracle[key]["compatibility"]
        assert compatible["configuredControl"] == expected["control_after_generate"]
        for name, value in expected.items():
            if name != "control_after_generate":
                assert compatible["configuredValues"][name] == value
        assert compatible["configuredReturn"] == "prior-configure-result"
        assert compatible["serializedReturn"] == "prior-serialize-result"
        assert compatible["configureArgumentPreserved"] and compatible["serializeArgumentPreserved"]
        assert compatible["properties"]["other_extension"] == "preserved"
        assert compatible["properties"]["drama3_t8_seed_control"] == "decrement"
        assert compatible["roundtripControl"] == "decrement"


def test_seed_compatibility_preserves_uint64_text_prior_hooks_and_rejects_unrecognized_controls(oracle):
    precision = oracle["precision"]["compatibility"]
    assert precision["configuredValues"]["seed"] == 2**53 - 1
    assert precision["configuredValues"]["seed_text"] == " 18446744073709551615 "
    assert precision["configuredControl"] == "fixed"
    assert oracle["prior-hook"]["compatibility"]["configuredControl"] == "increment"
    assert oracle["prior-hook"]["compatibility"]["roundtripControl"] == "increment"
    assert oracle["invalid-control"]["compatibility"]["configuredControl"] == "randomize"


def test_known_pre_chunk_scene_restores_mode_seed_and_appended_default_controls(oracle):
    fixture=Path(__file__).parent/'fixtures/scene-before-chunk-controls.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='8b3c26bc2a2d6b611d3c53a61ce897dd2103a7c04a314b8031efd026cf6aa9ee'
    saved=oracle['before-scene-chunks']['compatibility']
    assert saved['configuredControl']=='fixed'
    values=saved['configuredValues']
    assert values['generation_mode']=='per_speaker' and values['seed']==44 and values['seed_text']==''
    assert values['split_long_text'] is False and values['chunk_bytes']==300


def test_scene_sampling_old7_8_10_and_new15_restore_actual_official_frontend(oracle):
    fixed=Path(__file__).parent/'fixtures/scene-before-sampling-controls.json'
    metadata=json.loads(fixed.with_name('scene-before-sampling-controls-SOURCE.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(fixed.read_bytes()).hexdigest()==metadata['fixtures'][fixed.name]
    published_key=next(key for key,value in published_nodes().items() if value['type']=='T8DramaScene')
    old7=oracle[published_key]['compatibility']['configuredValues']
    old8=oracle['before-scene-chunks']['compatibility']['configuredValues']
    old10=oracle['before-scene-sampling']['compatibility']['configuredValues']
    new15=oracle['current-scene-sampling']['compatibility']['configuredValues']
    for values in (old7,old8,old10):
        assert (values['temperature'],values['top_p'],values['top_k'],values['opening_retries'],values['sampling_preset'])==(.7,.7,30,0,'custom')
    assert (old7['generation_mode'],old7['split_long_text'],old7['chunk_bytes'])==('together',False,300)
    assert (old8['generation_mode'],old8['split_long_text'],old8['chunk_bytes'])==('per_speaker',False,300)
    assert (old10['generation_mode'],old10['split_long_text'],old10['chunk_bytes'])==('per_line_context',True,48)
    assert old10['seed']==2**53-1 and old10['seed_text']=='18446744073709551615'
    control=oracle['before-scene-sampling']['compatibility']
    assert control['configuredControl']=='decrement' and control['roundtripControl']=='increment'
    assert (new15['temperature'],new15['top_p'],new15['top_k'],new15['opening_retries'],new15['sampling_preset'])==(1.2,.8,0,1,'custom')
    assert new15['generation_mode']=='per_speaker' and new15['chunk_bytes']==111 and new15['seed_text']=='18446744073709551615'
    assert oracle['current-scene-sampling']['compatibility']['configuredControl']=='decrement'


def test_editor_preserves_session_audio_and_subtitle_outputs(workflows):
    workflow = workflows["editor"]
    nodes = {node["id"]: node for node in workflow["nodes"]}
    editor = next(node for node in nodes.values() if node["type"] == "T8DramaEditor")
    consumers = {(link[2], nodes[link[3]]["type"]) for link in workflow["links"] if link[1] == editor["id"]}
    assert {(0, "T8DramaSaveSession"), (1, "SaveAudio"), (2, "PreviewAny"), (2, "SaveText")} <= consumers
    assert any(node["type"] == "T8DramaLoadSession" for node in nodes.values())
    options = maker.node_infos()["T8DramaScene"]["input"]["optional"]["generation_mode"][1]["options"]
    assert options == ["together", "per_line", "per_line_context", "per_speaker"]
    assert maker.EXAMPLES["T8DramaAlign"]["model_path"].endswith(".safetensors")


def assert_publication_matches_runtime(shipped, generated, infos):
    comparable = deepcopy(shipped)
    assert len(comparable["nodes"]) == len(generated["nodes"])
    for node, actual in zip(comparable["nodes"], generated["nodes"]):
        assert (node["id"], node["type"]) == (actual["id"], actual["type"])
        info = infos[node["type"]]
        names = maker.widget_names(info)
        for (node_type, name), public_name in PUBLIC_MODEL_SELECTIONS.items():
            if node_type != node["type"]:
                continue
            index = names.index(name)
            assert node["widgets_values"][index] == public_name
            if node["widgets_values"][index] == actual["widgets_values"][index]:
                continue
            # The generator must preserve public selections regardless of the
            # author's installed inventory; no placeholder substitution is valid.
            assert actual["widgets_values"][index] == public_name
    # Ports, links, all other widget values, layout and metadata remain strict.
    assert comparable == generated


def test_shipped_examples_equal_generator_and_contain_no_api_workflows(generated_workflows):
    paths = list(maker.FOLDER.glob("*.json"))
    assert {path.name for path in paths} == {f"workflow_{name}.json" for name in generated_workflows}
    published = shipped_workflows()
    for name, generated in generated_workflows.items():
        assert_publication_matches_runtime(published[name], generated, maker.node_infos())


@pytest.mark.parametrize("change", ["arbitrary_model", "wrong_fallback", "other_widget", "link_slot", "available_model"])
def test_inventory_equivalence_does_not_hide_changed_public_models_parameters_or_links(generated_workflows, change):
    shipped = shipped_workflows()["basic"]
    runtime = deepcopy(generated_workflows["basic"])
    infos = deepcopy(maker.node_infos())
    loader = next(node for node in shipped["nodes"] if node["type"] == "T8DramaLoader")
    model_index = maker.widget_names(infos["T8DramaLoader"]).index("model")
    if change == "arbitrary_model":
        loader["widgets_values"][model_index] = "some-authors-private-model.safetensors"
    elif change == "wrong_fallback":
        runtime["nodes"][0]["widgets_values"][model_index] = "not-the-actual-schema-fallback.safetensors"
    elif change == "other_widget":
        generate = next(node for node in shipped["nodes"] if node["type"] == "T8DramaGenerate")
        index = maker.widget_names(infos["T8DramaGenerate"]).index("max_frames")
        generate["widgets_values"][index] += 1
    elif change == "link_slot":
        shipped["links"][0][4] += 1
    else:
        # A published model that is actually available may never be replaced
        # under the inventory exception, even with another valid combo choice.
        for group in ("required", "optional"):
            spec = infos["T8DramaLoader"]["input"].get(group, {}).get("model")
            if spec is not None:
                options = spec[1]
                if isinstance(spec[0], list):
                    spec[0][:] = ["drama3-t8-fp8.safetensors", "<custom path>"]
                else:
                    options["options"] = ["drama3-t8-fp8.safetensors", "<custom path>"]
                options["default"] = "<custom path>"
        runtime["nodes"][0]["widgets_values"][model_index] = "<custom path>"
    with pytest.raises(AssertionError):
        assert_publication_matches_runtime(shipped, runtime, infos)
