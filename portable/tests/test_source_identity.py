from drama3_t8.source_identity import semantic_source_identity


def test_shared_identity_ignores_studio_only_files_and_git_line_endings(tmp_path):
    studio, node = tmp_path / "studio", tmp_path / "node"
    studio.mkdir()
    node.mkdir()
    (studio / "engine.py").write_bytes(b"def infer():\r\n    return 1\r\n")
    (node / "engine.py").write_bytes(b"def infer():\n    return 1\n")
    for extra in ("studio.py", "studio_editor.py", "bootstrap.py", "updater.py"):
        (studio / extra).write_text("only the local application", encoding="utf-8")
    assert semantic_source_identity(studio) == semantic_source_identity(node)
    (node / "kernel.cu").write_text("void kernel() {}", encoding="utf-8")
    assert semantic_source_identity(studio) != semantic_source_identity(node)
