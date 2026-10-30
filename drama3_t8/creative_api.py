"""HTTP presentation for local creative features; inference uses Studio's queue."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse


def public_json(value):
    # JavaScript numbers cannot round-trip the full uint64 seed range.
    if isinstance(value, dict):
        return {key: str(item) if key in {"seed", "base_seed", "requested_seed"} and type(item) is int
                else public_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_json(item) for item in value]
    return value


def register_creative_routes(app, root, store_upload):
    router = APIRouter(prefix="/api/creative")

    def desk(request):
        from .creative_studio import creative_manager
        return creative_manager(request.app.state.studio)

    def checked(function, *args, **kwargs):
        from .workspace_lock import WorkspaceBusy
        from .creative_workspace import ProjectConflict
        try:
            return public_json(function(*args, **kwargs))
        except (WorkspaceBusy, ProjectConflict) as exc:
            raise HTTPException(409, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except (ValueError, KeyError, TypeError, FileExistsError) as exc:
            raise HTTPException(400, str(exc)) from exc

    async def upload(request, file, extensions, *, limit_bytes=None):
        owner = desk(request)
        path = await store_upload(file, extensions, directory=owner.inputs, limit_bytes=limit_bytes)
        return owner.register_input(path)

    @router.get("/library")
    def library(request: Request, query: str = "", tags: str = "", favorites: bool = False, archived: bool = False):
        return checked(desk(request).library_view, query, tags, favorites, archived)

    @router.delete("/catalog/{kind}/{item_id}")
    def delete_catalog_item(kind: str, item_id: str, request: Request, payload: dict):
        return checked(desk(request).catalog_trash_item, kind, item_id, payload)

    @router.get("/trash")
    def trash(request: Request):
        return checked(desk(request).trash_view)

    @router.post("/trash/restore")
    def restore_trash(request: Request, payload: dict):
        return checked(desk(request).catalog_trash_item, payload.get("kind"), payload.get("id"), payload, restore=True)

    @router.get("/configuration")
    def configuration(request: Request):
        return checked(desk(request).configuration)

    @router.post("/configuration")
    def configure(request: Request, payload: dict):
        return checked(desk(request).configure, payload)

    @router.post("/voices")
    def create_voice(request: Request, payload: dict):
        return checked(desk(request).voices.create_voice, **payload)

    @router.patch("/voices/{voice_id}")
    def update_voice(voice_id: str, request: Request, payload: dict):
        return checked(desk(request).voices.update_voice, voice_id, **payload)

    @router.post("/reference")
    async def reference(request: Request, file: UploadFile, name: str = Form(""), voice_id: str = Form(""),
                        transcript: str = Form(""), language: str = Form(""), delivery: str = Form(""),
                        tags: str = Form(""), notes: str = Form(""), trim_silence: bool = Form(False),
                        normalize: bool = Form(False), start_seconds: float = Form(0), end_seconds: str = Form("")):
        path = await upload(request, file, {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"})
        return checked(desk(request).submit, {"mode": "creative_reference", "input_id": path.name,
            "name": name, "voice_id": voice_id, "transcript": transcript, "language": language,
            "delivery": delivery, "tags": tags, "notes": notes, "trim_silence": trim_silence,
            "normalize": normalize, "start_seconds": start_seconds,
            "end_seconds": end_seconds if end_seconds.strip() else None})

    @router.get("/versions/{version_id}/audio")
    def voice_audio(version_id: str, request: Request, original: bool = False):
        voices = desk(request).voices
        file = checked(voices.source_file if original else voices.preview_file, version_id)
        if file is None:
            raise HTTPException(404, "此版本没有保存可试听的原音频")
        return FileResponse(file)

    @router.post("/versions/{version_id}/reencode")
    def reencode(version_id: str, request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "version_id": version_id, "mode": "creative_reencode"})

    @router.post("/reference/from-input")
    def reference_from_input(request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "mode": "creative_reference"})

    @router.post("/casts")
    def create_cast(request: Request, payload: dict):
        return checked(desk(request).voices.create_cast, **payload)

    @router.post("/voices/export")
    def export_voices(request: Request, payload: dict):
        return checked(desk(request).export_voices, payload)

    @router.post("/voices/import")
    async def import_voices(request: Request, file: UploadFile, conflict: str = Form("rename")):
        path = await upload(request, file, {".zip"}, limit_bytes=2 * 1024**3)
        return checked(desk(request).voices.import_bundle, path, conflict=conflict)

    @router.get("/projects")
    def projects(request: Request):
        return checked(desk(request).workspace.list_projects)

    @router.post("/projects")
    def create_project(request: Request, payload: dict):
        return checked(desk(request).create_project, payload)

    @router.get("/projects/{project_id}")
    def get_project(project_id: str, request: Request):
        return checked(desk(request).project_view, project_id)

    @router.patch("/projects/{project_id}")
    def update_project(project_id: str, request: Request, payload: dict):
        return checked(desk(request).update_project, project_id, payload)

    @router.post("/projects/{project_id}/plan")
    def plan(project_id: str, request: Request, payload: dict):
        return checked(desk(request).plan, project_id, payload)

    @router.post("/projects/{project_id}/run")
    def run(project_id: str, request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "project_id": project_id, "mode": "creative_batch"})

    @router.post("/projects/{project_id}/review")
    def review(project_id: str, request: Request, payload: dict):
        return checked(desk(request).review, project_id, payload)

    @router.post("/projects/{project_id}/quality")
    def quality(project_id: str, request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "project_id": project_id, "mode": "creative_quality"})

    @router.post("/projects/{project_id}/variation")
    def variation(project_id: str, request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "project_id": project_id, "mode": "creative_variation"})

    @router.post("/projects/{project_id}/delivery")
    def delivery(project_id: str, request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "project_id": project_id, "mode": "creative_delivery"})

    @router.post("/projects/{project_id}/export")
    def export_project(project_id: str, request: Request, payload: dict):
        return checked(desk(request).export_project, project_id, payload)

    @router.post("/projects/import")
    async def import_project(request: Request, file: UploadFile):
        path = await upload(request, file, {".zip"}, limit_bytes=4 * 1024**3)
        return checked(desk(request).import_project, path)

    @router.get("/projects/{project_id}/files/{relative:path}")
    def project_file(project_id: str, relative: str, request: Request):
        return FileResponse(checked(desk(request).project_file, project_id, relative))

    @router.get("/projects/{project_id}/versions/{version_id}/audio")
    def project_voice_audio(project_id: str, version_id: str, request: Request, original: bool = False):
        from .creative_bundle import project_voice_library
        owner = desk(request)
        checked(owner.workspace.get_project, project_id)
        voices = project_voice_library(owner.workspace, project_id, owner.voices)
        file = checked(voices.source_file if original else voices.preview_file, version_id)
        if file is None:
            raise HTTPException(404, "此版本没有保存可试听的原音频")
        return FileResponse(file)

    @router.get("/downloads/{name}")
    def download(name: str, request: Request):
        return FileResponse(checked(desk(request).download_file, name))

    @router.get("/inputs/{name}")
    def input_audio(name: str, request: Request):
        return FileResponse(checked(desk(request).input_file, name))

    @router.post("/script/preview")
    def script_preview(request: Request, payload: dict):
        return checked(desk(request).script_preview, payload)

    @router.post("/recording")
    async def recording(request: Request, file: UploadFile, transcribe: bool = Form(True),
                        speakers: bool = Form(False), num_speakers: int = Form(0),
                        language: str = Form("auto"), name: str = Form("")):
        path = await upload(request, file, {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".webm"})
        return checked(desk(request).submit, {"mode": "creative_recording", "input_id": path.name,
            "name": name or Path(file.filename or "录音").stem, "transcribe": transcribe,
            "speakers": speakers, "num_speakers": num_speakers, "language": language})

    @router.get("/performance")
    def performance(request: Request):
        return checked(desk(request).performance_view)

    @router.post("/benchmark")
    def benchmark(request: Request, payload: dict):
        return checked(desk(request).submit, {**payload, "mode": "creative_benchmark"})

    @router.post("/benchmark/apply")
    def apply_profile(request: Request):
        return checked(desk(request).submit, {"mode": "creative_apply_profile"})

    app.include_router(router)
