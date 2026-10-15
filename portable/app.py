"""Local white/pink Drama 3 workspace; all inference runs in this process."""
import argparse
import json
import threading
import uuid
import webbrowser
import importlib.util
import re
from dataclasses import replace
from pathlib import Path
from contextlib import asynccontextmanager
import socket
from urllib.request import urlopen
from urllib.error import URLError

from drama3_t8.bootstrap import standalone_engine

ROOT = Path(__file__).resolve().parent
standalone_engine(ROOT)

import torch
import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drama3_t8.studio import Studio, SessionNotFound
from drama3_t8.state import load_session
from drama3_t8 import __version__
from safetensors import SafetensorError

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

def create_app(root=ROOT):
    root = Path(root).resolve()
    @asynccontextmanager
    async def lifespan(app):
        with Studio(root) as studio:
            app.state.studio = studio
            yield
    app = FastAPI(title="Drama 3 Local Studio · T8", docs_url=None, redoc_url=None, lifespan=lifespan)


    @app.get("/api/status")
    def status():
        return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "没有 CUDA GPU",
                "torch": torch.__version__, "model_ready": app.state.studio.engine.model_path.is_file(),
                "codec_ready": app.state.studio.engine.codec_path.is_file(), "model_loaded": app.state.studio.engine.drama is not None,
                "references": app.state.studio.references(), "sessions": app.state.studio.sessions(), "version": __version__, "application": "drama3-t8",
                "settings": app.state.studio.settings, "models": [p.name for p in (root / "models").glob("*.safetensors")],
                "model_paths": {key: {"path": str(path), "ready": path.exists()}
                                for key, path in app.state.studio.model_paths.items()},
                "acceleration": {"fla_installed": importlib.util.find_spec("fla") is not None,
                                 "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
                                 "attention": "SDPA", "cuda_graph": False}}


    @app.get("/api/jobs")
    def jobs():
        return app.state.studio.jobs()


    @app.get("/api/jobs/{job_id}")
    def job_details(job_id: str):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404)
        try:
            return app.state.studio.get_job(job_id)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc


    @app.post("/api/jobs")
    def submit(payload: dict):
        allowed = {"tts", "dialogue", "scene", "revise", "infill", "long", "srt", "text_stream", "unload", "configure", "align", "preview"}
        if not isinstance(payload.get("mode", "tts"), str) or payload.get("mode", "tts") not in allowed:
            raise HTTPException(400, "不支持的任务类型")
        try:
            return {"job_id": app.state.studio.submit(payload)}
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc


    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try:
            app.state.studio.cancel(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"cancelled": True}


    async def store_upload(file, extensions):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in extensions:
            raise HTTPException(400, "不支持的文件类型")
        path = app.state.studio.data / "inputs" / (uuid.uuid4().hex + extension)
        try:
            size = 0
            with path.open("wb") as target:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "文件不能超过 512 MiB")
                    target.write(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path


    @app.post("/api/projects/import")
    async def import_project(file: UploadFile):
        path = await store_upload(file, {".safetensors"})
        try:
            session = replace(load_session(path), session_id=uuid.uuid4().hex)
            return {"session_id":app.state.studio.register_session(session)}
        except (ValueError, KeyError, TypeError, IndexError, RuntimeError, SafetensorError) as exc:
            raise HTTPException(400, f"无法读取工程：{exc}") from exc
        finally:
            path.unlink(missing_ok=True)


    @app.get("/api/sessions/{session_id}")
    def session_details(session_id: str):
        try:
            session = app.state.studio.get_session(session_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, "工程存档不存在，请重新导入工程") from exc
        except SessionNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except (SafetensorError, ValueError, KeyError, TypeError, IndexError) as exc:
            raise HTTPException(400, f"无法读取工程：{exc}") from exc
        return {"session_id":session.session_id,"direction":session.direction,
                "project":f"/api/files/session/{session.session_id}.safetensors",
                "takes":[{"id":t.take_id,"text":t.transcript,"metadata":t.metadata,
                          "duration":t.codes.shape[-1]*2048/44100,
                          "audio":f"/api/sessions/{session_id}/takes/{i}.wav"} for i,t in enumerate(session.takes)]}


    @app.get("/api/sessions/{session_id}/takes/{index}.wav")
    def take_preview(session_id: str, index: int):
        if not re.fullmatch(r"[0-9a-f]{32}", session_id) or index < 0:
            raise HTTPException(404)
        target = app.state.studio.output / "previews" / f"{session_id}_{index}.wav"
        if target.is_file():
            return FileResponse(target, media_type="audio/wav")
        raise HTTPException(409, "请先在任务队列中解码此 take")


    @app.post("/api/references")
    async def reference(file: UploadFile, transcript: str = Form(""), name: str = Form("")):
        path = await store_upload(file, AUDIO_SUFFIXES)
        input_id = path.name
        try:
            return {"job_id": app.state.studio.submit({"mode": "reference", "input_id": input_id,
                                            "transcript": transcript, "name": name})}
        except BaseException:
            path.unlink(missing_ok=True)
            raise


    @app.post("/api/transcribe")
    async def transcribe_file(file: UploadFile, language: str = Form("auto"),
                              context: str = Form(""), hotwords: str = Form("")):
        path = await store_upload(file, AUDIO_SUFFIXES)
        input_id = path.name
        try:
            return {"job_id": app.state.studio.submit({"mode": "transcribe", "input_id": input_id,
                "language": language, "context": context, "hotwords": hotwords, "name": Path(file.filename or "audio.wav").name})}
        except BaseException:
            path.unlink(missing_ok=True)
            raise


    @app.get("/api/files/{kind}/{name}")
    def files(kind: str, name: str):
        bases = {"output": app.state.studio.output, "session": app.state.studio.data / "sessions",
                 "input": app.state.studio.data / "inputs"}
        if kind not in bases:
            raise HTTPException(404)
        base = bases[kind].resolve()
        path = (base / name).resolve()
        if path.parent != base or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)


    @app.get("/api/jobs/{job_id}/chunks")
    def chunks(job_id: str):
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(400)
        folder = app.state.studio.output / "stream" / job_id
        return [f"/api/stream/{job_id}/{p.name}" for p in sorted(folder.glob("*.wav"))]


    @app.get("/api/stream/{job_id}/{name}")
    def stream_file(job_id: str, name: str):
        base = (app.state.studio.output / "stream").resolve()
        path = (base / job_id / name).resolve()
        if not path.is_relative_to(base) or path.suffix != ".wav" or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")


    app.mount("/", StaticFiles(directory=root / "web", html=True), name="workspace")
    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            listener.bind(("127.0.0.1", args.port))
        except OSError as exc:
            try:
                with urlopen(url + "/api/status", timeout=2) as response:
                    status = json.load(response)
                existing = status.get("application") == "drama3-t8" or all(
                    key in status for key in ("model_ready", "codec_ready", "references", "sessions", "version"))
            except (OSError, ValueError, URLError):
                existing = False
            if not existing:
                raise RuntimeError(f"端口 {args.port} 已被其他程序占用，请使用 --port 选择其他端口") from exc
            if not args.no_browser:
                webbrowser.open(url)
            print(f"工作台已在运行：{url}")
            return
        listener.listen(128)
        print(f"工作台地址：{url}", flush=True)
        if not args.no_browser:
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port))
        server.run(sockets=[listener])
        if not server.started:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
