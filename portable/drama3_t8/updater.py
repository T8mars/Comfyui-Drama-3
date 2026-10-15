"""Standard-library-only code updates for an existing Windows portable bundle."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid
import zipfile

REPOSITORY = "T8mars/Comfyui-Drama-3"
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
MANIFEST_NAME = "portable-update.json"
MAX_DOWNLOAD = 256 * 1024 * 1024
MAX_UNPACKED = 512 * 1024 * 1024
MAX_FILES = 30000
ENTRYPOINT = "app.py"
VERSION_FILE = "drama3_t8/__init__.py"
ROOT_FILES = frozenset({"app.py", "Start-Drama3.cmd", "Update-Drama3.cmd", "README.md",
    "README.zh-CN.md", "LICENSE", "requirements.txt", "upstream-lock.json",
    "acceleration-manifest.json", "acceleration-requirements.txt", "acceptance-report.json"})
TOOLS = frozenset({"check_runtime.py", "download_models.py", "convert_model.py",
    "safetensors_stream.py", "extract_text_head.py", "update_portable.py"})
STATIC_SUFFIXES = frozenset({".html", ".css", ".js", ".json", ".svg", ".png", ".ico",
    ".webp", ".jpg", ".jpeg", ".gif", ".woff", ".woff2", ".ttf"})
SOURCE_SUFFIXES = STATIC_SUFFIXES | {".py", ".md", ".txt", ".toml", ".yaml", ".yml",
    ".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"}
PACKAGE_SUFFIXES = frozenset({".py", ".json", ".txt", ".jinja", ".jinja2", ".dict"})
FORBIDDEN_PARTS = frozenset({"models", "runtime", "userdata", "outputs", "output", "input",
    "temp", "user", "custom_nodes", "__pycache__", "roadmap.md", "node_modules"})
HEX = re.compile(r"[0-9a-f]{64}")
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
BACKUP_NAME = re.compile(r"backup-[0-9a-f]{32}")


class UpdateError(RuntimeError):
    pass


class UpdateBusy(UpdateError):
    pass


class WorkspaceInUse(UpdateError):
    pass


class RecoveryError(UpdateError):
    pass


def version_tuple(value):
    if not isinstance(value, str) or VERSION.fullmatch(value) is None:
        raise UpdateError(f"版本号格式错误：{value!r}")
    return tuple(map(int, value.split(".")))


def _relative(value):
    if not isinstance(value, str) or not value or len(value) > 240 or "\\" in value or ":" in value:
        raise UpdateError(f"非法更新路径：{value!r}")
    parts = value.split("/")
    for part in parts:
        if (not part or part in {".", ".."} or part.startswith(".") or part.endswith((".", " "))
                or any(ord(char) < 32 or char in '<>"|?*' for char in part)
                or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])", part.split(".")[0])):
            raise UpdateError(f"非法更新路径：{value!r}")
    if PurePosixPath(value).is_absolute():
        raise UpdateError(f"更新路径不能是绝对路径：{value}")
    return parts


def is_update_path(value):
    """The publisher and installer share this exact code-only allowlist."""
    try:
        parts = _relative(value)
    except UpdateError:
        return False
    if any(part.casefold() in {'__pycache__', 'roadmap.md', 'node_modules'} for part in parts):
        return False
    if len(parts) == 1:
        return value in ROOT_FILES
    suffix = PurePosixPath(value).suffix.lower()
    if parts[0] == "drama3_t8":
        return suffix in PACKAGE_SUFFIXES
    if parts[0] == "web":
        return suffix in STATIC_SUFFIXES
    if parts[0] == "tools":
        return len(parts) == 2 and parts[1] in TOOLS
    if parts[0] == "licenses":
        return (suffix in {"", ".txt", ".md", ".license", ".json"}
            or suffix[1:].isdecimal() and bool(re.fullmatch(r"(?i)(LICENSE|NOTICE|COPYING)[-_.][A-Za-z0-9_. -]+", parts[-1])))
    if parts[:2] == ["vendor", "ComfyUI"]:
        # ComfyUI's runtime data folders are at its root. Names such as
        # comfy_api/input and ldm/*/models are legitimate Python packages.
        return (len(parts) > 2 and parts[2].casefold() not in FORBIDDEN_PARTS
                and (suffix in SOURCE_SUFFIXES or parts[-1] in {"LICENSE", "COPYING"}))
    return False


def _no_reparse(path):
    """Check every existing component, including a dangling symlink/junction."""
    path = Path(path)
    for candidate in [path, *path.parents]:
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise UpdateError(f"更新路径含符号链接或重解析点，已停止：{candidate}")


def _target(root, relative):
    _relative(relative)
    if not is_update_path(relative):
        raise UpdateError(f"更新清单包含未授权文件：{relative}")
    target = root.joinpath(*relative.split("/"))
    _no_reparse(target)
    if not target.resolve().is_relative_to(root):
        raise UpdateError(f"更新文件越过安装目录：{relative}")
    if target.exists() and not target.is_file():
        raise UpdateError(f"更新目标不是普通文件：{relative}")
    return target


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise UpdateError(f"JSON 存在重复键：{key}")
        value[key] = item
    return value


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_object)
    except (ValueError, UnicodeError) as exc:
        raise UpdateError(f"无法读取更新 JSON：{exc}") from exc


def _record(value):
    if not isinstance(value, dict) or set(value) != {"sha256", "bytes"}:
        raise UpdateError("文件校验记录必须包含 sha256 和 bytes")
    digest, size = value["sha256"], value["bytes"]
    if not isinstance(digest, str) or HEX.fullmatch(digest) is None or type(size) is not int or size < 0:
        raise UpdateError("文件校验和或大小不合法")
    return value


def validate_manifest(value, installed=None):
    if not isinstance(value, dict) or type(value.get("schema")) is not int or value["schema"] != 1:
        raise UpdateError("不支持的 portable-update.json 协议")
    target_version, minimum = version_tuple(value.get("version")), version_tuple(value.get("min_version"))
    if minimum < (0, 1, 3) or minimum > target_version:
        raise UpdateError("更新基础版本范围不合法")
    if installed is not None and version_tuple(installed) < minimum:
        raise UpdateError(f"本更新至少需要完整便携包 {value['min_version']}")
    asset = value.get("asset")
    if not isinstance(asset, dict) or set(asset) != {"name", "sha256", "bytes"}:
        raise UpdateError("缺少更新 ZIP 资产信息")
    _record({key: asset[key] for key in ("sha256", "bytes")})
    if asset["name"] != f"Drama3-T8-Portable-{value['version']}-code.zip" or not 0 < asset["bytes"] <= MAX_DOWNLOAD:
        raise UpdateError("代码 ZIP 名称或大小不合法")
    files = value.get("files")
    if not isinstance(files, dict) or not 2 <= len(files) <= MAX_FILES:
        raise UpdateError("更新文件清单为空或过大")
    folded = set()
    for name, record in files.items():
        if not is_update_path(name) or name.casefold() in folded:
            raise UpdateError(f"更新路径未授权或存在 Windows 大小写冲突：{name}")
        folded.add(name.casefold())
        _record(record)
    if not {ENTRYPOINT, VERSION_FILE}.issubset(files) or sum(v["bytes"] for v in files.values()) > MAX_UNPACKED:
        raise UpdateError("更新缺少应用入口/版本文件，或解压后过大")
    return value


def _digest(path):
    total, digest = 0, hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(block)
            digest.update(block)
    return {"bytes": total, "sha256": digest.hexdigest()}


def _check(path, record):
    if _digest(path) != record:
        raise UpdateError(f"文件 SHA256/大小不匹配：{path}")


def _read_version(root):
    path = _target(root, VERSION_FILE)
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise UpdateError("无法读取已安装版本")
    version_tuple(match[1])
    return match[1]


def _atomic_json(path, value):
    _no_reparse(path)
    temporary = path.with_name(path.name + ".tmp")
    _no_reparse(temporary)
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def _file_lock(path, *, create, busy_type):
    _no_reparse(path)
    if not create and not path.is_file():
        raise WorkspaceInUse("工作台尚未初始化，请先正常启动一次；本次保留现有程序")
    stream = path.open("a+b" if create else "r+b")
    locked = False
    try:
        if create:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise busy_type("工作台或更新进程正在运行，请结束后再更新") from exc
        locked = True
        yield
    finally:
        if locked:
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_UN)
            except OSError:
                # Closing the handle also releases its lock. An unlock error
                # must never replace a RecoveryError with a generic fallback.
                pass
        stream.close()


def _prepare_root(root):
    root = Path(root).absolute()
    _no_reparse(root)
    root = root.resolve()
    if not ((root / "package-manifest.json").is_file() and (root / "runtime/python/python.exe").is_file()
            and (root / ENTRYPOINT).is_file()):
        raise UpdateError("此目录不是完整便携整合包；请保留原包的 Python、模型和 package-manifest.json")
    updates = root / ".updates"
    _no_reparse(updates)
    updates.mkdir(exist_ok=True)
    return root, updates


def _trusted_url(url):
    parsed = urlparse(url)
    hosts = {"api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise UpdateError("更新下载必须来自 GitHub 官方 HTTPS 地址")
    return url


class _GitHubRedirect(HTTPRedirectHandler):
    max_redirections = 3
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _trusted_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, target=None, *, limit, deadline, socket_timeout, expected=None, progress=None):
    _trusted_url(url)
    request = Request(url, headers={"User-Agent": "Drama3-T8-Portable-Updater", "Accept": "application/vnd.github+json"})
    left = deadline - time.monotonic()
    if left <= 0:
        raise UpdateError("更新请求超时")
    output, total, last_percent = bytearray(), 0, -1
    stream = target.open("wb") if target is not None else None
    try:
        with build_opener(_GitHubRedirect()).open(request, timeout=min(socket_timeout, left)) as response:
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > limit
                    or expected is not None and int(length) != expected):
                raise UpdateError("更新下载声明大小不匹配")
            while True:
                if time.monotonic() > deadline:
                    raise UpdateError("更新下载超时，保留旧版本")
                block = getattr(response, "read1", response.read)(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > limit or expected is not None and total > expected:
                    raise UpdateError("更新下载超过清单大小")
                if stream is not None:
                    stream.write(block)
                else:
                    output.extend(block)
                if progress and expected:
                    percent = min(100, total * 100 // expected)
                    if percent // 10 > last_percent // 10:
                        progress(f"下载代码更新：{percent}%（{total // 1024}/{expected // 1024} KiB）")
                        last_percent = percent
        if expected is not None and total != expected:
            raise UpdateError("更新下载不完整")
        if stream is not None:
            stream.flush()
            os.fsync(stream.fileno())
        return bytes(output)
    finally:
        if stream is not None:
            stream.close()


def _release(progress):
    deadline = time.monotonic() + 8
    release = _json(_download(API_URL, limit=1024 * 1024, deadline=deadline, socket_timeout=2))
    if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
        raise UpdateError("未找到正式 GitHub Release")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise UpdateError("GitHub Release 缺少资产列表")
    by_name = {}
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("name"), str) or asset["name"] in by_name:
            raise UpdateError("GitHub Release 资产列表无效或重复")
        by_name[asset["name"]] = asset
    if MANIFEST_NAME not in by_name:
        raise UpdateError("最新 Release 尚未提供便携更新清单")
    manifest_url = by_name[MANIFEST_NAME].get("browser_download_url", "")
    manifest = validate_manifest(_json(_download(manifest_url, limit=4 * 1024 * 1024,
        deadline=deadline, socket_timeout=2)))
    if release.get("tag_name") not in {manifest["version"], "v" + manifest["version"]}:
        raise UpdateError("Release 标签与更新版本不一致")
    name = manifest["asset"]["name"]
    if name not in by_name or by_name[name].get("size") != manifest["asset"]["bytes"]:
        raise UpdateError("Release 中的代码 ZIP 大小或名称不匹配")
    url = by_name[name].get("browser_download_url", "")
    expected_prefix = f"https://github.com/{REPOSITORY}/releases/download/"
    if not manifest_url.startswith(expected_prefix) or not url.startswith(expected_prefix):
        raise UpdateError("更新资产不属于当前官方仓库")
    progress(f"已发现正式版本 {manifest['version']}")
    return manifest, url


def _stage(root, updates, archive, manifest):
    stage = updates / ("stage-" + uuid.uuid4().hex)
    _no_reparse(stage)
    stage.mkdir()
    files = manifest["files"]
    with zipfile.ZipFile(archive) as bundle:
        members, folded = {}, set()
        for member in bundle.infolist():
            name = member.filename
            if (name not in files or not is_update_path(name) or name.casefold() in folded or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16) or member.external_attr & 0x400
                    or member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise UpdateError(f"ZIP 含未授权、重复或链接成员：{name}")
            if member.file_size != files[name]["bytes"]:
                raise UpdateError(f"ZIP 成员大小不匹配：{name}")
            members[name] = member
            folded.add(name.casefold())
        if set(members) != set(files):
            raise UpdateError("ZIP 内容与更新文件清单不一致")
        for name, member in members.items():
            _target(root, name)
            destination = stage.joinpath(*name.split("/"))
            _no_reparse(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            _check(destination, files[name])
            if destination.suffix == ".py":
                try:
                    compile(destination.read_bytes(), name, "exec")
                except (ValueError, SyntaxError) as exc:
                    raise UpdateError(f"更新 Python 文件不能编译：{name}：{exc}") from exc
    if _read_version(stage) != manifest["version"]:
        raise UpdateError("ZIP 内版本文件与更新清单不一致")
    return stage


def _copy(source, target):
    _no_reparse(source)
    _no_reparse(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("wb") as output:
        shutil.copyfileobj(incoming, output, 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def _validated_journal(root, updates, journal):
    if not isinstance(journal, dict) or journal.get("schema") != 1 or not BACKUP_NAME.fullmatch(str(journal.get("backup", ""))):
        raise RecoveryError("更新恢复记录不合法，未修改应用文件")
    backup = updates / journal["backup"]
    _no_reparse(backup)
    entries = journal.get("files")
    if not isinstance(entries, list) or not 2 <= len(entries) <= MAX_FILES:
        raise RecoveryError("更新恢复文件列表不合法")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("existed")) is not bool:
            raise RecoveryError("更新恢复条目不合法")
        name = entry.get("path")
        _target(root, name)
        if name.casefold() in seen:
            raise RecoveryError("更新恢复条目重复")
        seen.add(name.casefold())
        if entry["existed"]:
            _record(entry.get("original"))
            original = backup.joinpath(*name.split("/"))
            _no_reparse(original)
            _check(original, entry["original"])
    return backup, entries


def _invalidate_bytecode(root, names):
    for name in names:
        if not name.endswith(".py"):
            continue
        source = _target(root, name)
        cache = source.parent / "__pycache__"
        _no_reparse(cache)
        if cache.is_dir():
            for candidate in cache.glob(source.stem + ".*.pyc"):
                _no_reparse(candidate)
                candidate.unlink()


def _rollback(root, updates, journal):
    try:
        backup, entries = _validated_journal(root, updates, journal)
        failures = []
        for entry in reversed(entries):
            name = entry["path"]
            destination = _target(root, name)
            try:
                if entry["existed"]:
                    if destination.is_file() and _digest(destination) == entry["original"]:
                        continue
                    temporary = destination.with_name(destination.name + ".restore-" + uuid.uuid4().hex)
                    try:
                        _copy(backup.joinpath(*name.split("/")), temporary)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                else:
                    destination.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(f"{name}：{exc}")
        if failures:
            raise RecoveryError("；".join(failures))
        _invalidate_bytecode(root, [entry["path"] for entry in entries])
        journal["state"] = "rolled_back"
        _atomic_json(updates / "transaction.json", journal)
    except Exception as exc:
        raise RecoveryError(f"自动回滚未完成，请保留 .updates 备份并停止启动：{exc}") from exc


def _recover(root, updates, progress):
    try:
        journal_path = updates / "transaction.json"
        _no_reparse(journal_path)
        if not journal_path.exists():
            return
        if journal_path.stat().st_size > 32 * 1024 * 1024:
            raise RecoveryError("更新恢复记录过大，已停止")
        journal = _json(journal_path.read_bytes())
        if not isinstance(journal, dict) or journal.get("state") not in {"applying", "committed", "rolled_back"}:
            raise RecoveryError("更新恢复状态不合法，已停止")
        if journal["state"] == "applying":
            progress("发现上次更新未完成，正在恢复原版本…")
            _rollback(root, updates, journal)
            progress("原版本已恢复。")
    except Exception as exc:
        raise RecoveryError(f"无法安全恢复上次更新，已停止启动：{exc}") from exc


def _required_space(root, manifest):
    old_sizes = [_target(root, name).stat().st_size for name in manifest["files"] if _target(root, name).is_file()]
    return (sum(item["bytes"] for item in manifest["files"].values()) + sum(old_sizes)
            + max(old_sizes, default=0) + 16 * 1024 * 1024)


def _apply(root, updates, stage, manifest):
    names = sorted(manifest["files"], key=lambda name: (name == ENTRYPOINT, name == VERSION_FILE, name))
    backup = updates / ("backup-" + uuid.uuid4().hex)
    backup.mkdir()
    entries = []
    for name in names:
        destination = _target(root, name)
        entry = {"path": name, "existed": destination.is_file()}
        if entry["existed"]:
            original = backup.joinpath(*name.split("/"))
            _copy(destination, original)
            entry["original"] = _digest(original)
        entries.append(entry)
    journal = {"schema": 1, "state": "applying", "backup": backup.name,
        "from_version": _read_version(root), "to_version": manifest["version"], "files": entries}
    _atomic_json(updates / "transaction.json", journal)
    try:
        for name in names:
            destination = _target(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage.joinpath(*name.split("/")), destination)
        _invalidate_bytecode(root, names)
        journal["state"] = "committed"
        _atomic_json(updates / "transaction.json", journal)
    except BaseException as exc:
        _rollback(root, updates, journal)
        if not isinstance(exc, Exception):
            raise
        raise UpdateError(f"更新失败，已恢复原版本：{exc}") from exc


def install_archive(root, archive, manifest, progress=print):
    """Install a verified local artifact; useful for offline updates and testing."""
    root, updates = _prepare_root(root)
    with _file_lock(updates / "update.lock", create=True, busy_type=UpdateBusy):
        with _file_lock(root / "userdata/studio.lock", create=False, busy_type=WorkspaceInUse):
            _recover(root, updates, progress)
            current = _read_version(root)
            validate_manifest(manifest, current)
            if version_tuple(manifest["version"]) <= version_tuple(current):
                raise UpdateError("本地更新版本必须高于当前版本")
            _check(archive, {key: manifest["asset"][key] for key in ("bytes", "sha256")})
            required = _required_space(root, manifest)
            if shutil.disk_usage(root).free < required:
                raise UpdateError("磁盘空间不足以同时保存代码更新和回滚备份")
            stage = _stage(root, updates, archive, manifest)
            _apply(root, updates, stage, manifest)
    return {"status": "updated", "version": manifest["version"]}


def check_and_update(root, progress=print):
    root, updates = _prepare_root(root)
    with _file_lock(updates / "update.lock", create=True, busy_type=UpdateBusy):
        # Recovery comes before networking or version checks. An interrupted
        # commit may already have replaced the version file.
        journal = updates / "transaction.json"
        try:
            _no_reparse(journal)
        except UpdateError as exc:
            raise RecoveryError(str(exc)) from exc
        if journal.exists():
            with _file_lock(root / "userdata/studio.lock", create=False, busy_type=WorkspaceInUse):
                _recover(root, updates, progress)
        current = _read_version(root)
        manifest, url = _release(progress)
        validate_manifest(manifest, current)
        if version_tuple(manifest["version"]) <= version_tuple(current):
            progress(f"当前版本 {current} 已是最新正式版。")
            return {"status": "current", "version": current}
        with _file_lock(root / "userdata/studio.lock", create=False, busy_type=WorkspaceInUse):
            archive = updates / ("download-" + uuid.uuid4().hex + ".zip.part")
            try:
                progress(f"正在更新 {current} → {manifest['version']}；模型与用户数据保持原位。")
                _no_reparse(archive)
                _download(url, archive, limit=manifest["asset"]["bytes"], expected=manifest["asset"]["bytes"],
                    deadline=time.monotonic() + 600, socket_timeout=20, progress=progress)
                _check(archive, {key: manifest["asset"][key] for key in ("sha256", "bytes")})
                required = _required_space(root, manifest)
                if shutil.disk_usage(root).free < required:
                    raise UpdateError("磁盘空间不足以保存代码和回滚备份")
                stage = _stage(root, updates, archive, manifest)
                _apply(root, updates, stage, manifest)
            finally:
                try:
                    archive.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    progress(f"下载缓存暂未清理：{archive}：{cleanup_error}")
        progress(f"更新完成：{manifest['version']}。")
        return {"status": "updated", "version": manifest["version"]}
