"""Real temporary portable updates; no network, Torch, models or user workspace."""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import pytest

from drama3_t8 import updater as update


def test_comfy_source_packages_named_input_and_models_are_not_runtime_data():
    assert update.is_update_path('vendor/ComfyUI/comfy_api/input/__init__.py')
    assert update.is_update_path('vendor/ComfyUI/comfy/ldm/example/models/core.py')
    assert not update.is_update_path('vendor/ComfyUI/input/private.json')
    assert not update.is_update_path('vendor/ComfyUI/models/custom.py')
    assert not update.is_update_path('vendor/ComfyUI/custom_nodes/other.py')
    assert not update.is_update_path('vendor/ComfyUI/comfy_api/input/weights.safetensors')


def record(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture
def portable(tmp_path):
    root = tmp_path / "便携 包 (测试)"
    files = {"app.py": b"print('old app')\n", "drama3_t8/__init__.py": b'__version__ = "0.1.6"\n',
        "web/app.js": b"old code", "package-manifest.json": b'{"version":"0.1.6"}',
        "runtime/python/python.exe": b"preserved interpreter", "models/main.gguf": b"preserved weights",
        "userdata/studio.lock": b"\0", "userdata/studio.sqlite3": b"preserved database",
        "userdata/runtime-settings.json": b'{"model":"E:/custom/model.safetensors"}',
        "outputs/prior.wav": b"preserved user audio", "roadmap.md": b"private local research"}
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def bundle(tmp_path, *, extra=None, contents=None, zip_contents=None):
    files = contents or {"app.py": b"print('new app')\n", "drama3_t8/__init__.py": b'__version__ = "0.1.7"\n',
        "web/app.js": b"new code", "tools/update_portable.py": b"# new updater entry\n",
        "drama3_t8/native/fish/config.json": b'{"version":1}',
        "drama3_t8/vendor_qwen/inference/assets/korean_dict_jieba.dict": b"tokenizer data"}
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in (zip_contents or files).items():
            output.writestr(name, data)
        if extra:
            for name, data in extra:
                output.writestr(name, data)
    manifest = {"schema": 1, "version": "0.1.7", "min_version": "0.1.3",
        "asset": {"name": "Drama3-T8-Portable-0.1.7-code.zip", **record(archive.read_bytes())},
        "files": {name: record(data) for name, data in files.items()}}
    return archive, manifest, files


def protected_snapshot(root):
    return {name: (root / name).read_bytes() for name in ("runtime/python/python.exe", "models/main.gguf",
        "userdata/studio.lock", "userdata/studio.sqlite3", "userdata/runtime-settings.json", "outputs/prior.wav", "roadmap.md")}


def test_real_portable_update_keeps_models_runtime_data_and_commits_entrypoint_last(portable, tmp_path, monkeypatch):
    archive, manifest, contents = bundle(tmp_path)
    protected = protected_snapshot(portable)
    changes = []
    original = update.os.replace
    def replace(source, destination):
        if Path(destination).is_relative_to(portable) and ".updates" not in Path(destination).relative_to(portable).parts:
            changes.append(Path(destination).relative_to(portable).as_posix())
        return original(source, destination)
    monkeypatch.setattr(update.os, "replace", replace)
    assert update.install_archive(portable, archive, manifest)["version"] == "0.1.7"
    assert changes[-1] == "app.py"
    assert all((portable / name).read_bytes() == data for name, data in contents.items())
    assert protected_snapshot(portable) == protected
    transaction = json.loads((portable / ".updates/transaction.json").read_text(encoding="utf-8"))
    assert transaction["state"] == "committed" and (portable / ".updates" / transaction["backup"]).is_dir()


def test_failed_replace_restores_all_originals_and_removes_new_files(portable, tmp_path, monkeypatch):
    archive, manifest, _ = bundle(tmp_path)
    original_bytes = {name: (portable / name).read_bytes() for name in ("app.py", "web/app.js", "drama3_t8/__init__.py")}
    protected = protected_snapshot(portable)
    original, failed = update.os.replace, []
    def replace(source, destination):
        if Path(destination) == portable / "app.py" and not failed:
            failed.append(True)
            raise PermissionError("scanner holds app")
        return original(source, destination)
    monkeypatch.setattr(update.os, "replace", replace)
    with pytest.raises(update.UpdateError, match="已恢复原版本"):
        update.install_archive(portable, archive, manifest)
    assert all((portable / name).read_bytes() == data for name, data in original_bytes.items())
    assert not (portable / "tools/update_portable.py").exists()
    assert protected_snapshot(portable) == protected
    assert json.loads((portable / ".updates/transaction.json").read_text(encoding="utf-8"))["state"] == "rolled_back"


def test_failed_rollback_stays_recoverable_and_is_completed_before_next_network_check(portable, tmp_path, monkeypatch):
    archive, manifest, _ = bundle(tmp_path)
    original = update.os.replace
    def replace(source, destination):
        if Path(destination) == portable / "app.py" or ".restore-" in str(source) and Path(destination) == portable / "web/app.js":
            raise PermissionError("locked test file")
        return original(source, destination)
    monkeypatch.setattr(update.os, "replace", replace)
    with pytest.raises(update.RecoveryError, match="web/app.js"):
        update.install_archive(portable, archive, manifest)
    assert (portable / "web/app.js").read_bytes() == b"new code"
    monkeypatch.setattr(update.os, "replace", original)
    def offline(progress):
        assert (portable / "web/app.js").read_bytes() == b"old code"
        assert update._read_version(portable) == "0.1.6"
        raise OSError("offline after recovery")
    monkeypatch.setattr(update, "_release", offline)
    with pytest.raises(OSError, match="offline after recovery"):
        update.check_and_update(portable)
    assert json.loads((portable / ".updates/transaction.json").read_text(encoding="utf-8"))["state"] == "rolled_back"


@pytest.mark.parametrize("name", ["../userdata/studio.sqlite3", "userdata/new.json", "models/main.gguf",
    "runtime/python/python.exe", "roadmap.md", "drama3_t8/model.dll", "web/../app.py", "web\\app.js",
    "web/CON.txt", "web/name. ", "web/foo:bar.js", "web//app.js", "/app.py",
    "vendor/ComfyUI/custom_nodes/secret.py", "comfyui-drama3-T8/nodes.py"])
def test_manifest_rejects_protected_windows_paths_and_binaries(tmp_path, name):
    _, manifest, _ = bundle(tmp_path)
    manifest["files"][name] = record(b"bad")
    with pytest.raises(update.UpdateError):
        update.validate_manifest(manifest, "0.1.6")


@pytest.mark.parametrize("extra", [[("web/app.js", b"duplicate")], [("userdata/studio.sqlite3", b"bad")]])
def test_zip_duplicate_or_extra_file_is_rejected_before_application_changes(portable, tmp_path, extra):
    archive, manifest, _ = bundle(tmp_path, extra=extra)
    with pytest.raises(update.UpdateError, match="ZIP"):
        update.install_archive(portable, archive, manifest)
    assert update._read_version(portable) == "0.1.6" and (portable / "web/app.js").read_bytes() == b"old code"


def test_zip_symbolic_link_rejected(portable, tmp_path):
    archive, manifest, contents = bundle(tmp_path)
    with zipfile.ZipFile(archive, "w") as output:
        for name, data in contents.items():
            member = zipfile.ZipInfo(name)
            if name == "web/app.js":
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(member, data)
    manifest["asset"].update(record(archive.read_bytes()))
    with pytest.raises(update.UpdateError, match="链接"):
        update.install_archive(portable, archive, manifest)
    assert update._read_version(portable) == "0.1.6"


def test_real_windows_junction_parent_cannot_redirect_code_update(portable, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows junction contract")
    import shutil
    assert (portable / "web").resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(portable / "web")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "app.js").write_bytes(b"outside data")
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(portable / "web"), str(outside)], capture_output=True)
    assert result.returncode == 0, result.stderr
    archive, manifest, _ = bundle(tmp_path)
    with pytest.raises(update.UpdateError, match="重解析"):
        update.install_archive(portable, archive, manifest)
    assert (outside / "app.js").read_bytes() == b"outside data"


@pytest.mark.parametrize("where", ["asset", "file"])
def test_sha_failure_never_changes_installed_code(portable, tmp_path, where):
    archive, manifest, _ = bundle(tmp_path)
    (manifest["asset"] if where == "asset" else manifest["files"]["web/app.js"])["sha256"] = "0" * 64
    with pytest.raises(update.UpdateError, match="SHA256"):
        update.install_archive(portable, archive, manifest)
    assert update._read_version(portable) == "0.1.6"


def test_workspace_lock_in_another_process_prevents_updates(portable, tmp_path):
    archive, manifest, _ = bundle(tmp_path)
    code = ("import sys; from drama3_t8.workspace_lock import WorkspaceLock; "
            "owner=WorkspaceLock(sys.argv[1]); print('ready',flush=True); sys.stdin.readline(); owner.close()")
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-c", code, str(portable / "userdata/studio.lock")],
        cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(update.WorkspaceInUse):
            update.install_archive(portable, archive, manifest)
    finally:
        process.communicate("\n", timeout=10)
    assert update._read_version(portable) == "0.1.6"


def test_second_update_is_blocked_by_same_update_lock(portable, tmp_path):
    _, updates = update._prepare_root(portable)
    archive, manifest, _ = bundle(tmp_path)
    with update._file_lock(updates / "update.lock", create=True, busy_type=update.UpdateBusy):
        with pytest.raises(update.UpdateBusy):
            update.install_archive(portable, archive, manifest)


def test_never_started_bundle_does_not_create_userdata(portable, tmp_path):
    import shutil
    assert (portable / "userdata").resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(portable / "userdata")
    archive, manifest, _ = bundle(tmp_path)
    with pytest.raises(update.WorkspaceInUse, match="初始化"):
        update.install_archive(portable, archive, manifest)
    assert not (portable / "userdata").exists() and update._read_version(portable) == "0.1.6"


def test_invalid_recovery_json_blocks_startup_instead_of_treating_it_as_offline(portable, monkeypatch):
    _, updates = update._prepare_root(portable)
    (updates / "transaction.json").write_bytes(b"{invalid")
    monkeypatch.setattr(update, "_release", lambda p: pytest.fail("network before recovery"))
    with pytest.raises(update.RecoveryError):
        update.check_and_update(portable)


def test_malicious_recovery_path_is_rejected_without_touching_userdata(portable, monkeypatch):
    _, updates = update._prepare_root(portable)
    protected = protected_snapshot(portable)
    journal = {"schema": 1, "state": "applying", "backup": "backup-" + "a" * 32,
        "files": [{"path": "userdata/studio.sqlite3", "existed": False}, {"path": "app.py", "existed": False}]}
    (updates / "transaction.json").write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(update.RecoveryError):
        update.check_and_update(portable)
    assert protected_snapshot(portable) == protected and (portable / "app.py").exists()


def test_download_and_release_check_apply_verified_code_using_separate_time_budgets(portable, tmp_path, monkeypatch):
    archive, manifest, contents = bundle(tmp_path)
    monkeypatch.setattr(update, "_release", lambda p: (manifest, "https://github.com/T8mars/Comfyui-Drama-3/releases/download/v0.1.7/code.zip"))
    def download(url, target=None, **kwargs):
        assert kwargs["socket_timeout"] == 20 and kwargs["deadline"] - time.monotonic() > 590
        assert kwargs["expected"] == len(archive.read_bytes())
        target.write_bytes(archive.read_bytes())
    monkeypatch.setattr(update, "_download", download)
    assert update.check_and_update(portable, lambda s: None)["status"] == "updated"
    assert (portable / "app.py").read_bytes() == contents["app.py"]
    assert not list((portable / ".updates").glob("download-*"))


def test_offline_check_never_changes_code_and_numeric_versions_compare_correctly(portable, monkeypatch):
    monkeypatch.setattr(update, "_release", lambda p: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(OSError, match="offline"):
        update.check_and_update(portable)
    assert update._read_version(portable) == "0.1.6"
    assert update.version_tuple("0.1.10") > update.version_tuple("0.1.9")


def test_python_syntax_error_is_rejected_before_commit(portable, tmp_path):
    archive, manifest, _ = bundle(tmp_path, contents={"app.py": b"def invalid(\n", "drama3_t8/__init__.py": b'__version__ = "0.1.7"\n'})
    with pytest.raises(update.UpdateError, match="不能编译"):
        update.install_archive(portable, archive, manifest)
    assert update._read_version(portable) == "0.1.6"


def test_stale_bytecode_is_invalidated_after_commit(portable, tmp_path):
    archive, manifest, _ = bundle(tmp_path)
    cache = portable / "drama3_t8/__pycache__/__init__.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"stale bytecode")
    update.install_archive(portable, archive, manifest)
    assert not cache.exists()


def test_license_and_tokenizer_assets_allowed_but_weights_remain_forbidden():
    for name in ("licenses/LICENSE-Apache-2.0", "licenses/NVIDIA-CUDA.txt",
            "drama3_t8/native/fish/tokenizer.json", "drama3_t8/qwen3_asr/added_tokens.json",
            "drama3_t8/vendor_qwen/inference/assets/korean_dict_jieba.dict", "vendor/ComfyUI/comfy/tokenizer/merges.txt"):
        assert update.is_update_path(name), name
    for name in ("licenses/model.safetensors", "drama3_t8/native/model.pt", "vendor/ComfyUI/comfy/model.bin"):
        assert not update.is_update_path(name), name


def test_release_checks_have_short_deadline_and_require_official_matching_assets(tmp_path, monkeypatch):
    _, manifest, _ = bundle(tmp_path)
    prefix = f"https://github.com/{update.REPOSITORY}/releases/download/v0.1.7/"
    release = {"tag_name": "v0.1.7", "draft": False, "prerelease": False,
        "assets": [{"name": update.MANIFEST_NAME, "browser_download_url": prefix + update.MANIFEST_NAME},
            {"name": manifest["asset"]["name"], "browser_download_url": prefix + manifest["asset"]["name"], "size": manifest["asset"]["bytes"]}]}
    responses = iter([release, manifest])
    def download(url, **kwargs):
        assert kwargs["socket_timeout"] == 2 and 0 < kwargs["deadline"] - time.monotonic() <= 8
        return json.dumps(next(responses)).encode()
    monkeypatch.setattr(update, "_download", download)
    found, url = update._release(lambda s: None)
    assert found == manifest and url.startswith(prefix)


def test_corrupt_backup_does_not_start_partial_rollback(portable, tmp_path):
    archive, manifest, _ = bundle(tmp_path)
    update.install_archive(portable, archive, manifest)
    journal_path = portable / ".updates/transaction.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["state"] = "applying"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    (portable / ".updates" / journal["backup"] / "web/app.js").write_bytes(b"corrupt backup")
    with pytest.raises(update.RecoveryError, match="SHA256"):
        update.check_and_update(portable)
    assert update._read_version(portable) == "0.1.7" and (portable / "web/app.js").read_bytes() == b"new code"


def test_cli_automatic_offline_continues_but_incomplete_recovery_stops(monkeypatch, capsys):
    import tools.update_portable as cli
    monkeypatch.setattr(cli, "check_and_update", lambda *a: (_ for _ in ()).throw(OSError("offline")))
    assert cli.main(["--automatic"]) == 0
    assert cli.main([]) == 1
    monkeypatch.setattr(cli, "check_and_update", lambda *a: (_ for _ in ()).throw(update.RecoveryError("cannot restore")))
    assert cli.main(["--automatic"]) == 2
    monkeypatch.setattr(cli, "check_and_update", lambda *a: (_ for _ in ()).throw(update.UpdateBusy("other startup")))
    assert cli.main(["--automatic"]) == 75
    assert "已停止启动" in capsys.readouterr().out


def test_manifest_rejects_case_collisions_and_duplicate_json_keys(tmp_path):
    _, manifest, _ = bundle(tmp_path)
    manifest["files"]["web/App.js"] = record(b"conflict")
    with pytest.raises(update.UpdateError, match="大小写冲突"):
        update.validate_manifest(manifest)
    with pytest.raises(update.UpdateError, match="重复键"):
        update._json('{"schema":1,"schema":2}')


def test_download_rejects_bad_size_and_insecure_redirect_before_writing(portable, monkeypatch):
    with pytest.raises(update.UpdateError, match="HTTPS"):
        update._download("http://github.com/file", limit=10, deadline=time.monotonic() + 2, socket_timeout=1)
    class Response(io.BytesIO):
        headers = {"Content-Length": "100"}
    monkeypatch.setattr(update, "build_opener", lambda *a: SimpleNamespace(open=lambda *a, **kw: Response(b"small")))
    with pytest.raises(update.UpdateError, match="大小不匹配"):
        update._download("https://github.com/file", limit=10, deadline=time.monotonic() + 2, socket_timeout=1)


def test_official_same_version_does_not_download_or_change_code(portable, tmp_path, monkeypatch):
    _, manifest, _ = bundle(tmp_path)
    manifest["version"] = "0.1.6"
    manifest["asset"]["name"] = "Drama3-T8-Portable-0.1.6-code.zip"
    monkeypatch.setattr(update, "_release", lambda p: (manifest, "unused"))
    monkeypatch.setattr(update, "_download", lambda *a, **k: pytest.fail("same version downloaded code"))
    assert update.check_and_update(portable)["status"] == "current"
    assert update._read_version(portable) == "0.1.6"


def test_updater_imports_without_torch_or_comfy():
    code = "import sys; import drama3_t8.updater; assert 'torch' not in sys.modules; assert 'comfy' not in sys.modules; print('stdlib-only')"
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and "stdlib-only" in result.stdout


def test_windows_launcher_finishes_preparsed_block_even_when_updater_replaces_cmd(portable, tmp_path):
    if os.name != "nt":
        pytest.skip("Windows CMD block semantics")
    project = Path(__file__).resolve().parents[1]
    junction = portable / "runtime/python"
    assert junction.resolve().is_relative_to(tmp_path.resolve())
    (junction / "python.exe").unlink()
    junction.rmdir()
    result = subprocess.run(["cmd", "/d", "/c", "mklink", "/J", str(junction), str(Path(sys.executable).parent)], capture_output=True)
    assert result.returncode == 0, result.stderr
    try:
        (portable / "Start-Drama3.cmd").write_bytes((project / "Start-Drama3.cmd").read_bytes())
        (portable / "tools").mkdir()
        (portable / "tools/update_portable.py").write_text(
            "from pathlib import Path\nroot=Path(__file__).resolve().parents[1]\n"
            "(root/'Start-Drama3.cmd').write_text('@echo off\\necho Wrong new body\\nexit /b 4\\n')\n", encoding="utf-8")
        (portable / "app.py").write_text("from pathlib import Path\nPath(__file__).with_name('app-started.txt').write_text('started')\n", encoding="utf-8")
        launched = subprocess.run(["cmd", "/d", "/c", str(portable / "Start-Drama3.cmd")], capture_output=True, timeout=30)
        assert launched.returncode == 0, (launched.stdout, launched.stderr)
        assert (portable / "app-started.txt").read_text() == "started"
        assert b"Wrong new body" not in launched.stdout
    finally:
        assert getattr(junction.lstat(), "st_file_attributes", 0) & 0x400
        junction.rmdir()
