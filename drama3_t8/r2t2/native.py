"""Python-ABI independent, same-process Confucius Q8 inference. No worker or server."""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import re
import struct
import threading
from functools import lru_cache
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from ..asr_language import normalize_asr_language

SUPPORTED_LANGUAGES = ("Chinese", "English", "Cantonese", "Japanese", "Korean", "German",
                       "French", "Russian", "Portuguese", "Spanish", "Italian")
PAIR = {
    "model": (1834422208, "151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3"),
    "projector": (348336544, "8dc2c67e6a0484114928142d098db7ad94ae9f34c78948ef9d37a9678418cb65"),
}
_HANDLES = []
_LIBRARIES = {}
_LOAD_LOCK = threading.RLock()
CancelCallback = C.CFUNCTYPE(C.c_bool, C.c_void_p)


@lru_cache(maxsize=16)
def _verify(path, size, mtime, expected_hash):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        if struct.unpack("<4sI", f.read(8)) != (b"GGUF", 3):
            raise ValueError(f"无效 GGUF 模型：{path}")
        f.seek(0)
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"Confucius Q8 模型校验失败：{path}")


def verify_file(path, role):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    size, digest = PAIR[role]
    if stat.st_size != size:
        raise ValueError(f"请选择 Confucius4-R2T2 官方 Q8 {role} 文件：{path}")
    _verify(str(path), size, stat.st_mtime_ns, digest)
    return path


def runtime_directory():
    override = os.environ.get("DRAMA_R2T2_NATIVE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    for candidate in (root / "runtime/r2t2-native", root / "r2t2-runtime"):
        if (candidate / "lib/drama_r2t2.dll").is_file():
            return candidate
    raise FileNotFoundError("缺少 Confucius 原生运行库，请使用含 r2t2-runtime 的 Windows 节点包 / 整合包")


def _load_library():
    if os.name != "nt":
        raise RuntimeError("当前附带 Confucius 原生运行库支持 Windows x64 / CUDA 12.8")
    directory = runtime_directory()
    with _LOAD_LOCK:
        if directory in _LIBRARIES:
            return _LIBRARIES[directory]
        folders = [directory / "lib", directory / "backends"]
        torch_spec = find_spec("torch")
        if torch_spec and torch_spec.origin:
            folders.append(Path(torch_spec.origin).parent / "lib")
        for folder in folders:
            if folder.is_dir():
                _HANDLES.append(os.add_dll_directory(str(folder)))
        registry = C.CDLL(str(directory / "lib/ggml.dll"))
        registry.ggml_backend_load.argtypes = [C.c_char_p]
        registry.ggml_backend_load.restype = C.c_void_p
        registry.ggml_backend_dev_count.restype = C.c_size_t
        registry.ggml_backend_dev_by_type.argtypes = [C.c_int]
        registry.ggml_backend_dev_by_type.restype = C.c_void_p
        # Other native nodes may already have registered only a CPU backend.
        # Fill each missing device type; total device count cannot prove CUDA is available.
        for device_type, name in ((0, "ggml-cpu.dll"), (1, "ggml-cuda.dll")):
            if not registry.ggml_backend_dev_by_type(device_type):
                plugin = directory / "backends" / name
                if plugin.is_file():
                    registry.ggml_backend_load(str(plugin).encode("utf-8"))
        if not registry.ggml_backend_dev_by_type(1):
            raise RuntimeError("Confucius CUDA 后端未发现 GPU，请检查附带运行库与 NVIDIA 驱动")
        lib = C.CDLL(str(directory / "lib/drama_r2t2.dll"))
        lib.r2t2_api_version.restype = C.c_int
        if lib.r2t2_api_version() != 1:
            raise RuntimeError("Confucius 原生 C API 版本不匹配")
        lib.r2t2_create.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int]
        lib.r2t2_create.restype = C.c_void_p
        lib.r2t2_destroy.argtypes = [C.c_void_p]
        lib.r2t2_destroy.restype = None
        lib.r2t2_generate.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_size_t,
                                     C.c_char_p, C.c_int, CancelCallback]
        lib.r2t2_generate.restype = C.c_char_p
        lib.r2t2_last_error.restype = C.c_char_p
        lib.r2t2_last_truncated.restype = C.c_bool
        _HANDLES.append(registry)
        _LIBRARIES[directory] = lib
        return lib


def build_prompt(context="", language=None, prefix=""):
    language = normalize_asr_language(language, SUPPORTED_LANGUAGES, backend="Confucius")
    if not isinstance(context, str) or not isinstance(prefix, str):
        raise ValueError("Confucius context and prefix must be text")
    if "\0" in context or "\0" in prefix:
        raise ValueError("Confucius prompt cannot contain NUL characters")
    if len(context) > 8192:
        raise ValueError("CONTEXT_LIMIT: 提示词和热词总计不能超过 8192 字符")
    return ("<|im_start|>system\n" + context + "<|im_end|>\n"
            "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
            "<|im_end|>\n<|im_start|>assistant\n" +
            (f"language {language}<asr_text>" if language else "") + prefix)


def parse_raw(raw, language=None):
    # Consume complete visible control tokens rather than leaving a dangling '<'.
    raw = re.split(r"<\|[^<>]*\|>", raw, maxsplit=1)[0].replace("\ufffd", "")
    # Confucius may emit token 91 ('|') before EOG. The native ABI removes
    # EOG but leaves this terminal marker; embedded literal pipes are text.
    raw = raw.rstrip().removesuffix("|")
    if language:
        return language, raw.strip()
    if "<asr_text>" not in raw:
        if raw.strip():
            raise RuntimeError("ASR_PARSE_ERROR: 自动语言转写缺少 <asr_text> 标记，请指定语言后重试")
        return "", ""
    meta, text = raw.split("<asr_text>", 1)
    language = next((line.strip()[9:].strip() for line in meta.splitlines()
                     if line.strip().lower().startswith("language ")), "")
    return ("" if language.lower() == "none" else language), text.strip()


class NativeQ8Engine:
    def __init__(self, model, projector, *, n_ctx=8192, n_threads=8, check_cancel=None):
        model, projector = verify_file(model, "model"), verify_file(projector, "projector")
        self._lib = _load_library()
        self._lock = threading.RLock()
        self._handle = None
        self._cancel_error = None
        self.check_cancel = check_cancel or (lambda: None)
        self._callback = CancelCallback(self._cancelled)
        self.model_name, self.projector_name = model.name, projector.name
        self.check_cancel()
        self._handle = self._lib.r2t2_create(str(model).encode("utf-8"), str(projector).encode("utf-8"),
                                           n_ctx, 1024, n_threads, -1)
        if not self._handle:
            raise RuntimeError(self._lib.r2t2_last_error().decode("utf-8", "replace"))

    def _cancelled(self, _):
        try:
            self.check_cancel()
            return False
        except BaseException as exc:
            self._cancel_error = exc
            return True

    def transcribe(self, audio, *, context="", language=None, max_tokens=1024):
        language = normalize_asr_language(language, SUPPORTED_LANGUAGES, backend="Confucius")
        pcm = np.ascontiguousarray(audio, dtype=np.float32)
        if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
            raise ValueError("Expected nonempty finite mono PCM")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        prompt = build_prompt(context, language)
        with self._lock:
            if not self._handle:
                raise RuntimeError("Confucius 模型已释放")
            self.check_cancel()
            self._cancel_error = None
            raw = self._lib.r2t2_generate(self._handle, pcm.ctypes.data_as(C.POINTER(C.c_float)),
                                         len(pcm), prompt.encode("utf-8"), max_tokens, self._callback)
            if self._cancel_error is not None:
                raise self._cancel_error
            if raw is None:
                raise RuntimeError(self._lib.r2t2_last_error().decode("utf-8", "replace"))
            truncated = bool(self._lib.r2t2_last_truncated())
        detected, text = parse_raw(raw.decode("utf-8"), language)
        return {"text": text, "language": detected, "truncated": truncated,
                "finish_reason": "length" if truncated else "stop",
                "model": self.model_name, "projector": self.projector_name}

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.r2t2_destroy(self._handle)
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
