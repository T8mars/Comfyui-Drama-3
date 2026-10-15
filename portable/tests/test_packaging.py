import pytest
from tools import build_portable


@pytest.mark.parametrize("suffix", [".dll", ".safetensors", ".json"])
def test_same_size_replacement_is_included_in_next_package(tmp_path, monkeypatch, suffix):
    target = tmp_path / "package"
    monkeypatch.setattr(build_portable, "TARGET", target)
    monkeypatch.setattr(build_portable, "COPIED", set())
    source = tmp_path / ("source" + suffix)
    destination = target / source.name
    source.write_bytes(b"old")
    build_portable.copy_file(source, destination)
    source.write_bytes(b"new")
    build_portable.copy_file(source, destination)
    assert destination.read_bytes() == b"new"
