import json
from pathlib import Path

import pytest

from drama3_t8.model_paths import DEFAULT_PATHS, model_path, resolve_models
from drama3_t8.bootstrap import standalone_engine

standalone_engine()
from drama3_t8.studio import Studio


def model_settings(tmp_path):
    settings = {"asr_backend": "r2t2"}
    for key, default in DEFAULT_PATHS.items():
        path = tmp_path / ("分开的模型 " + key) / Path(default).name
        if key in {"aligner", "r2t2_vad"}:
            path.mkdir(parents=True)
            names = ["config.json"] if key == "aligner" else ["cmvn.ark", "fireredvad_stream_vad_with_cache.onnx"]
            for name in names:
                (path / name).write_text("{}")
        else:
            path.parent.mkdir(parents=True)
            path.touch()
        settings[key] = str(path)
    return settings


def test_external_paths_survive_settings_reload_without_rebasing(tmp_path):
    root = tmp_path / "应用目录"
    settings = model_settings(tmp_path)
    studio = Studio(root)
    studio.apply_settings(settings)
    studio.close()
    restored = Studio(root)
    assert restored.engine.model_path == Path(settings["model"])
    assert restored.engine.codec_path == Path(settings["codec"])
    assert restored.engine.text_head_path == Path(settings["text_head"])
    assert restored.engine.asr_path == Path(settings["asr_model"])
    assert restored.engine.r2t2_paths["r2t2_projector"] == Path(settings["r2t2_projector"])
    assert restored.model_paths["aligner"] == Path(settings["aligner"])
    assert json.loads(studio.settings_path.read_text(encoding="utf-8"))["r2t2_vad"] == settings["r2t2_vad"]
    restored.close()


def test_bad_paths_do_not_replace_working_settings_or_engine(tmp_path):
    studio = Studio(tmp_path / "app")
    studio.apply_settings(model_settings(tmp_path))
    previous, saved = studio.engine, studio.settings_path.read_bytes()
    with pytest.raises(FileNotFoundError):
        studio.apply_settings({"codec": str(tmp_path / "missing.safetensors")})
    assert studio.engine is previous
    assert studio.settings_path.read_bytes() == saved
    with pytest.raises(ValueError):
        studio.apply_settings({"model": "model.bin"})


def test_relative_and_environment_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("DRAMA_TEST_MODELS", str(tmp_path / "外置"))
    assert model_path("%DRAMA_TEST_MODELS%/a.gguf", tmp_path) == (tmp_path / "外置/a.gguf").resolve()
    assert resolve_models(DEFAULT_PATHS, tmp_path)["model"] == tmp_path / DEFAULT_PATHS["model"]


def test_single_file_aligner_and_compact_head_can_replace_legacy_directories(tmp_path):
    settings=model_settings(tmp_path)
    aligner=tmp_path/'native-aligner.safetensors'
    aligner.touch()
    settings['aligner']=str(aligner)
    Path(settings['text_head']).unlink()
    paths=resolve_models(settings,tmp_path,check=True)
    assert paths['aligner']==aligner and paths['line_head'].is_file()
    with pytest.raises(FileNotFoundError):
        resolve_models({**settings,'scene_head_mode':'full'},tmp_path,check=True)
