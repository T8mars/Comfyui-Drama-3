"""Local workspace, serialized GPU queue and durable generation results."""
import json
import re
from dataclasses import replace
from contextlib import contextmanager
import queue
import sqlite3
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import av
import comfy.model_management as mm

from .engine import DramaEngine, Sampling
from .state import Session, load_session, save_session
from .live_audio import LiveAudio
from .batch import generate_srt
from .model_paths import DEFAULT_PATHS, resolve_models, require_models
from .workspace_lock import WorkspaceLock


class Cancelled(RuntimeError):
    pass


class SessionNotFound(ValueError):
    pass


def integer_option(value, label):
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]{1,20}", value):
        value = int(value)
    if type(value) is not int:
        raise ValueError(f"{label}必须是整数")
    return value


def playback_identity(take,audio,subtype='PCM_16'):
    """Bind the actual preview format to the immutable selected source view."""
    from .render_infill import playback_identity as identity
    from .editor import fingerprint
    if audio['sample_rate']!=44100:raise ValueError('Infill playback requires native 44.1kHz audio')
    return identity(audio['waveform'][0],subtype,source=fingerprint(take))


@contextmanager
def result_files(*paths):
    # Each job publishes new identities. Never roll back a pre-existing result.
    for path in paths:
        if path.exists():
            raise FileExistsError(f"拒绝覆盖已有结果文件：{path}")
    try:
        yield
    except BaseException as exc:
        failures = []
        for path in reversed(paths):
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                failures.append(f"{path}：{cleanup_error}")
        if failures:
            raise RuntimeError(f"{exc}；部分结果文件未能清理：{'；'.join(failures)}") from exc
        raise


def read_audio(path):
    try:
        waveform, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(waveform.T.copy()), rate
    except sf.LibsndfileError:
        parts = []
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=44100)
        with av.open(str(path)) as container:
            for frame in container.decode(audio=0):
                parts.extend(f.to_ndarray() for f in resampler.resample(frame))
            parts.extend(f.to_ndarray() for f in resampler.resample(None))
        if not parts:
            raise ValueError("The selected file contains no decoded audio")
        return torch.from_numpy(np.concatenate(parts, axis=-1).copy()), 44100


class Studio:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = self.root / "userdata"
        self.data.mkdir(parents=True, exist_ok=True)
        self._owner = WorkspaceLock(self.data / "studio.lock")
        self._activity_lock = threading.RLock()
        self._active_job = None
        self._closed = False
        try:
            self._initialize()
        except BaseException:
            self._owner.close()
            raise

    def _initialize(self):
        self.output = self.root / "outputs"
        for folder in (self.data, self.data / "inputs", self.data / "references",
                       self.data / "sessions", self.data / "logs", self.output):
            folder.mkdir(parents=True, exist_ok=True)
        self.database = self.data / "studio.sqlite3"
        self.engine = DramaEngine(self.root / "models" / "drama3-t8-fp8.safetensors",
                                  self.root / "models" / "drama3-codec.safetensors")
        self.settings_path = self.data / "runtime-settings.json"
        self.settings = {**DEFAULT_PATHS, "asr_backend": "r2t2",
                         "expert_cache_gb": 12, "gdn_backend": "torch","moe_backend":"auto","cuda_graph":True,
                         "expert_placement":"ram"}
        if self.settings_path.exists():
            self.settings.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
        self.apply_settings(self.settings, persist=False)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,status TEXT,payload TEXT,result TEXT,
                    error TEXT,created REAL,updated REAL,progress REAL);
                CREATE TABLE IF NOT EXISTS voice_references(id TEXT PRIMARY KEY,name TEXT,transcript TEXT,path TEXT);
                CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,path TEXT,summary TEXT,updated REAL);
            """)
            db.execute("UPDATE jobs SET status='interrupted',error='程序重启中断了此任务，请重新提交' WHERE status IN ('running','queued')")
            for table, folder in (("sessions", "sessions"), ("voice_references", "references")):
                for row in db.execute(f"SELECT id,path FROM {table}").fetchall():
                    relative = f"{folder}/{row['id']}.safetensors"
                    if (self.data / relative).is_file() and row["path"] != relative:
                        db.execute(f"UPDATE {table} SET path=? WHERE id=?", (relative, row["id"]))
        self.pending = queue.Queue()
        self.cancel_events = {}
        self.worker = threading.Thread(target=self.run_queue, name="Drama-GPU-queue", daemon=True)
        self.worker.start()

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def submit(self, payload):
        encoded_payload = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        payload = json.loads(encoded_payload)
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._activity_lock:
            if self._closed:
                raise RuntimeError("工作台正在关闭，不能提交新任务")
            with self.connect() as db:
                db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)",
                           (job_id, "queued", encoded_payload, None, None, now, now, 0))
            self.cancel_events[job_id] = threading.Event()
            self.pending.put((job_id, payload))
        return job_id

    def cancel(self, job_id):
        with self._activity_lock:
            event = self.cancel_events.get(job_id)
            if event is None:
                raise ValueError("The task is no longer active")
            event.set()
            if self._active_job == job_id:
                mm.interrupt_current_processing(True)

    def begin_commit(self, job_id, progress):
        with self._activity_lock:
            progress()
            # Saving a result or applying settings must finish with an accurate status.
            self.cancel_events.pop(job_id, None)

    def close(self):
        with self._activity_lock:
            if self._closed:
                return
            self._closed = True
            for event in self.cancel_events.values():
                event.set()
            if self._active_job is not None:
                mm.interrupt_current_processing(True)
            self.pending.put(None)
        self.worker.join()
        try:
            self.engine.unload()
        finally:
            self._owner.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def data_path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.data / path

    def jobs(self):
        with self.connect() as db:
            rows = db.execute("""SELECT * FROM jobs WHERE status IN ('running','queued')
                OR id IN (SELECT id FROM jobs ORDER BY created DESC LIMIT 100)
                ORDER BY created DESC""").fetchall()
        with self._activity_lock:
            cancelable = set(self.cancel_events)
        return [{**dict(r), "payload": json.loads(r["payload"]), "cancelable": r["id"] in cancelable,
                 "result": json.loads(r["result"]) if r["result"] else None} for r in rows]

    def get_job(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("The requested task does not exist")
        with self._activity_lock:
            cancelable = job_id in self.cancel_events
        return {**dict(row), "payload": json.loads(row["payload"]), "cancelable": cancelable,
                "result": json.loads(row["result"]) if row["result"] else None}

    def references(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,name,transcript FROM voice_references ORDER BY name")]

    def sessions(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,summary,updated FROM sessions ORDER BY updated DESC")]

    def asr_languages(self, backend=None):
        """Advertise the selected implementation's actual manual languages."""
        backend = self.settings['asr_backend'] if backend is None else backend
        if backend == 'qwen':
            from .qwen3_asr.asr import LANGUAGES
            languages = LANGUAGES
        elif backend == 'r2t2':
            from .r2t2.native import SUPPORTED_LANGUAGES
            languages = SUPPORTED_LANGUAGES
        else:
            raise ValueError('请选择 Confucius R2T2 或 Qwen3-ASR')
        return ['auto', *languages]

    def get_session(self, session_id):
        with self.connect() as db:
            row = db.execute("SELECT path FROM sessions WHERE id=?", (session_id,)).fetchone()
        if row is None:
            raise SessionNotFound("The requested session does not exist")
        return load_session(self.data_path(row["path"]))

    def apply_settings(self, settings, persist=True):
        allowed = {*DEFAULT_PATHS, "asr_backend", "expert_cache_gb", "gdn_backend", "scene_head_mode","moe_backend","cuda_graph","expert_placement"}
        chosen = {**self.settings, **{k: v for k, v in settings.items() if k in allowed}}
        if chosen["gdn_backend"] not in {"torch", "fla"} or not 0 <= float(chosen["expert_cache_gb"]) <= 128:
            raise ValueError("Invalid acceleration or expert cache setting")
        if chosen["asr_backend"] not in {"r2t2", "qwen"}:
            raise ValueError("请选择 Confucius R2T2 或 Qwen3-ASR")
        if chosen.get("scene_head_mode","compact") not in {"compact","full"}:
            raise ValueError("场景控制头应选择 compact 或 full")
        if chosen.get('moe_backend','auto') not in {'auto','torch','cuda'} or type(chosen.get('cuda_graph',True)) is not bool:
            raise ValueError('请选择有效的专家加速与 CUDA Graph 设置')
        if chosen.get('expert_placement','ram') not in {'ram','vram'}:
            raise ValueError('专家驻留位置应选择 ram 或 vram')
        paths = resolve_models(chosen, self.root / "models", check=persist)
        engine = DramaEngine(paths["model"], paths["codec"],
            expert_cache_gb=float(chosen["expert_cache_gb"]), gdn_backend=chosen["gdn_backend"],
            text_head_path=paths["text_head"], line_head_path=paths["line_head"],
            scene_head_mode=chosen.get("scene_head_mode","compact"),asr_path=paths["asr_model"],
            moe_backend=chosen.get('moe_backend','auto'),cuda_graph=chosen.get('cuda_graph',True),
            expert_placement=chosen.get('expert_placement','ram'),
            asr_backend=chosen["asr_backend"], r2t2_paths=paths)
        temporary = self.settings_path.with_suffix(".tmp") if persist else None
        try:
            if temporary is not None:
                temporary.write_text(json.dumps(chosen, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.engine.unload()
            if temporary is not None:
                temporary.replace(self.settings_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self.engine = engine
        self.settings = chosen
        self.model_paths = paths

    def register_session(self, session):
        target = self.data / "sessions" / f"{session.session_id}.safetensors"
        summary = session.takes[-1].transcript[:100] if session.takes else "导入的音色 / 空会话"
        with result_files(target):
            save_session(session, target)
            with self.connect() as db:
                db.execute("INSERT INTO sessions VALUES(?,?,?,?)", (session.session_id, target.relative_to(self.data).as_posix(), summary, time.time()))
        return session.session_id

    def session_for(self, payload):
        if payload.get("session_id"):
            return self.get_session(payload["session_id"])
        references = []
        with self.connect() as db:
            for ref_id in payload.get("reference_ids", []):
                row = db.execute("SELECT path FROM voice_references WHERE id=?", (ref_id,)).fetchone()
                if row is None:
                    raise ValueError(f"Unknown reference {ref_id}")
                for reference in load_session(self.data_path(row["path"])).references:
                    requested = payload.get("reference_speakers", {}).get(ref_id)
                    if requested in (None, ""):
                        references.append(reference)
                        continue
                    speaker = integer_option(requested, "角色编号 / Speaker")
                    if not 0 <= speaker < 16:
                        raise ValueError("Speaker must be between 0 and 15")
                    if len(set(re.findall(r"<\|speaker:(\d+)\|>",reference.transcript)))>1:
                        raise ValueError("此参考含多个角色，请保留原稿映射或拆分参考音频")
                    transcript = re.sub(r"<\|speaker:\d+\|>\s*", "", reference.transcript)
                    references.append(replace(reference, transcript=f"<|speaker:{speaker}|> {transcript}"))
        return Session(tuple(references), direction=payload.get("direction", ""))

    def run_queue(self):
        while True:
            item = self.pending.get()
            if item is None:
                self.pending.task_done()
                return
            job_id, payload = item
            cancel = self.cancel_events[job_id]
            result_json = None
            last_update = 0

            def progress(step=0, total=1):
                nonlocal last_update
                if cancel.is_set():
                    raise Cancelled("任务已取消")
                if time.monotonic() - last_update > 0.25:
                    last_update = time.monotonic()
                    with self.connect() as db:
                        db.execute("UPDATE jobs SET progress=?,updated=? WHERE id=?",
                                   (min(0.99, step / max(1, total)), time.time(), job_id))

            try:
                progress()
                with self._activity_lock:
                    if cancel.is_set():
                        raise Cancelled("任务已取消")
                    self._active_job = job_id
                    mm.interrupt_current_processing(False)
                with self.connect() as db:
                    db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?", (time.time(), job_id))
                with torch.inference_mode():
                    result = self.execute(job_id, payload, progress)
                result_json = json.dumps(result, ensure_ascii=False, allow_nan=False)
                if cancel.is_set():
                    raise Cancelled("任务已取消")
                status, error = "done", None
            except (Cancelled, mm.InterruptProcessingException) as exc:
                result, status, error = None, "cancelled", str(exc) or "任务已取消"
            except BaseException as exc:
                result, status, error = None, "cancelled" if cancel.is_set() else "failed", str(exc)
                if status == "failed":
                    try:
                        (self.data / "logs" / f"{job_id}.log").write_text(traceback.format_exc(), encoding="utf-8")
                        if hasattr(exc,"generation_attempts"):
                            (self.data/"logs"/f"{job_id}.attempts.json").write_text(
                                json.dumps(exc.generation_attempts,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
                    except OSError as log_error:
                        error += f"（错误日志未能保存：{log_error}）"
            with self._activity_lock:
                if cancel.is_set():
                    result, status, error = None, "cancelled", "任务已取消"
                self._active_job = None
                mm.interrupt_current_processing(False)
                self.cancel_events.pop(job_id, None)
            with self.connect() as db:
                db.execute("UPDATE jobs SET status=?,result=?,error=?,updated=?,progress=? WHERE id=?",
                           (status, result_json if status == "done" else None, error, time.time(),
                            1 if status == "done" else 0, job_id))
            self.pending.task_done()

    def execute(self, job_id, payload, progress):
        mode = payload.get("mode", "tts")
        if mode == 'editor_prepare':
            if type(payload.get('align_words',False)) is not bool:
                raise ValueError('align_words must be a boolean')
            if payload.get('align_words',False):require_models(self.model_paths,'aligner')
            from .studio_editor import prepare_job
            return prepare_job(self,job_id,payload,progress)
        if mode == "configure":
            self.begin_commit(job_id, progress)
            self.apply_settings(payload["settings"])
            return {"message": "运行设置已保存", "settings": self.settings}
        if mode == "unload":
            self.begin_commit(job_id, progress)
            self.engine.unload()
            return {"message": "模型已释放"}
        if mode in {"reference", "transcribe"}:
            path = self.data / "inputs" / payload["input_id"]
            if path.resolve().parent != (self.data / "inputs").resolve():
                raise ValueError("Invalid input path")
            waveform, rate = read_audio(path)
            transcript = payload.get("transcript", "")
            if mode == "transcribe":
                result = self.engine.transcribe_detailed(waveform, rate, language=payload.get("language"),
                    context=payload.get("context", ""), hotwords=payload.get("hotwords", ""), check_cancel=progress)
                progress()
                self.begin_commit(job_id, progress)
                text_file, metadata = self.output / f"{job_id}.txt", self.output / f"{job_id}.asr.json"
                with result_files(text_file, metadata):
                    text_file.write_text(result["text"], encoding="utf-8")
                    metadata.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                return {**result, "transcript": result["text"],
                        "source_audio": f"/api/files/input/{path.name}",
                        "source_name": payload.get("name", path.name),
                        "text_file": f"/api/files/output/{text_file.name}",
                        "metadata": f"/api/files/output/{metadata.name}"}
            if not transcript.strip():
                result = self.engine.transcribe_detailed(waveform, rate, check_cancel=progress)
                transcript = result["text"]
                if result.get("truncated") or not transcript.strip():
                    raise ValueError("转写结果为空或被截断，请检查音频并填写准确稿件后再保存音色")
            options={key:payload[key] for key in ("trim_silence","normalize") if payload.get(key,False)}
            ref = self.engine.encode_reference(waveform, rate, transcript, payload.get("name", ""),**options)
            progress()
            self.begin_commit(job_id, progress)
            target = self.data / "references" / f"{job_id}.safetensors"
            with result_files(target):
                save_session(Session((ref,)), target)
                with self.connect() as db:
                    db.execute("INSERT INTO voice_references VALUES(?,?,?,?)",
                               (job_id, ref.name, ref.transcript, target.relative_to(self.data).as_posix()))
            return {"reference_id": job_id, "name": ref.name, "frames": ref.codes.shape[-1]}
        sampling_options = dict(payload.get("sampling", {}))
        seed = sampling_options.get("seed", 42)
        if isinstance(seed, str):
            if not re.fullmatch(r"[0-9]{1,20}", seed):
                raise ValueError("随机种子应为0到18446744073709551615的十进制整数")
            sampling_options["seed"] = int(seed)
        sampling = Sampling(**sampling_options).with_preset(payload.get("sampling_preset","custom"))
        sampling.validate()
        session = self.session_for(payload)
        if mode in {"infill", "revise", "preview", "align"}:
            index = integer_option(payload.get("take_index", -1), "演出版本 / take")
            if not session.takes:
                raise ValueError("请选择含音频的会话")
            if index < -1 or index >= len(session.takes):
                raise ValueError("The requested take does not exist")
            if mode in {"infill", "revise"} and index >= 0:
                session = session.branch(index + 1)
        if mode == "preview":
            if index < 0:
                index += len(session.takes)
            take = session.takes[index]
            from .editor import decode_output
            audio = decode_output(self.engine,take)
            subtype='FLOAT' if take.metadata.get('audio_editor') else take.metadata.get('render_infill',{}).get('subtype','PCM_16')
            view=playback_identity(take,audio,subtype)
            self.begin_commit(job_id, progress)
            path = self.output / "previews" / f"{session.session_id}_{index}.wav"
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name(f"{path.stem}.{job_id}.part.wav")
            try:
                sf.write(temporary, audio["waveform"][0].T.numpy(), audio["sample_rate"], subtype=subtype)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return {"audio":f"/api/sessions/{session.session_id}/takes/{index}.wav", "session_id":session.session_id,
                    "take_id":take.take_id,"take_index":index,"take_count":len(session.takes),
                    "playback_view":view,
                    "transcript":take.transcript, "metrics":take.metadata,
                    "project":f"/api/files/session/{session.session_id}.safetensors"}
        if mode == "align":
            from .alignment import align_audio
            take = session.takes[index]
            stored=take.metadata.get('audio_editor')
            if stored and stored.get('source'):
                from .editor import fingerprint
                if stored['source']!=fingerprint(take):
                    raise ValueError('Editor source no longer matches the selected take')
            if stored and stored.get('aligned'):
                # Reuse validated word timing after the actual cuts and gaps.
                from .editor import prepare,plan,captions
                _,analysis,edits=prepare(self.engine,take)
                mapped=captions(analysis,plan(analysis,edits)['pieces'],edits)
                words=[word for caption in mapped for word in caption['words']]
            else:
                if stored:
                    raise ValueError('此工程已有后期剪辑；请在多轨编辑器中启用词级字幕后重新准备，避免用完整原稿对齐已剪切音频')
                require_models(self.model_paths,'aligner')
                audio = self.engine.decode_take(take) if hasattr(self.engine,'decode_take') else self.engine.decode(take.codes)
                words = align_audio(self.model_paths["aligner"], audio["waveform"], audio["sample_rate"],
                                    take.transcript, payload.get("language", "Chinese"))
            self.begin_commit(job_id, progress)
            path = self.output / f"{job_id}.alignment.json"
            with result_files(path):
                path.write_text(json.dumps(words, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            return {"alignment": words, "session_id": session.session_id, "take_id":take.take_id,
                    "metadata": f"/api/files/output/{path.name}"}
        live = (LiveAudio(self.engine.codec_path, self.output / "stream" / job_id)
                if payload.get("audio_stream") and mode in {"tts", "dialogue", "scene", "revise", "long", "text_stream"} else None)
        on_codes = live.append if live is not None else None
        srt_audio, subtitle = None, None
        try:
            if mode == "srt":
                result, srt_audio, subtitle = generate_srt(self.engine, payload["text"], session, sampling, progress)
            else:
                result = self.generate_session(mode, payload, session, sampling, progress, on_codes)
        except BaseException:
            if live is not None:
                live.finish(abort=True)
            raise
        if live is not None:
            live.finish()
        progress()
        take = result.takes[-1]
        output_codes = (torch.cat([t.codes for t in result.takes[len(session.takes):]], dim=1)
                        if mode == "long" else take.codes)
        if srt_audio is not None:audio=srt_audio
        elif mode not in {'long','srt'} and hasattr(self.engine,'decode_take'):audio=self.engine.decode_take(take)
        else:audio=self.engine.decode(output_codes)
        subtype=take.metadata.get('render_infill',{}).get('subtype','PCM_16')
        view=playback_identity(take,audio,subtype) if mode not in {'long','srt'} else None
        self.begin_commit(job_id, progress)
        target = self.output / f"{job_id}.wav"
        preview = self.output / "previews" / f"{result.session_id}_{len(result.takes)-1}.wav"
        session_path = self.data / "sessions" / f"{result.session_id}.safetensors"
        metadata_path = self.output / f"{job_id}.json"
        subtitle_path = self.output / f"{job_id}.srt"
        paths = [target, session_path, metadata_path]
        if mode not in {"long", "srt"}:
            paths.append(preview)
        if subtitle is not None:
            paths.append(subtitle_path)
        with result_files(*paths):
            sf.write(target, audio["waveform"][0].T.numpy(), audio["sample_rate"], subtype=subtype)
            preview.parent.mkdir(exist_ok=True)
            if mode not in {"long", "srt"}:
                import shutil
                shutil.copyfile(target, preview)
            save_session(result, session_path)
            output_metadata = {**take.metadata, "output_duration_seconds": audio["waveform"].shape[-1] / audio["sample_rate"]}
            if mode in {"long", "srt"}:
                new_takes=result.takes[len(session.takes):]
                output_metadata.update(elapsed_seconds=sum(t.metadata["elapsed_seconds"] for t in new_takes),
                                       duration_seconds=audio["waveform"].shape[-1]/audio["sample_rate"],
                                       length_limit_reached=any(t.metadata["length_limit_reached"] for t in new_takes),
                                       chunk_count=len(new_takes),chunks=[t.metadata for t in new_takes])
            metadata_path.write_text(json.dumps(output_metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            subtitle_url = None
            if subtitle is not None:
                subtitle_path.write_text(subtitle, encoding="utf-8")
                subtitle_url = f"/api/files/output/{subtitle_path.name}"
            with self.connect() as db:
                db.execute("INSERT INTO sessions VALUES(?,?,?,?)",
                           (result.session_id, session_path.relative_to(self.data).as_posix(), take.transcript[:100], time.time()))
        return {"audio": f"/api/files/output/{target.name}", "metadata": f"/api/files/output/{metadata_path.name}",
                "session_id": result.session_id, "take_id": take.take_id,
                "take_index":len(result.takes)-1,"take_count":len(result.takes),
                "playback_view":view,
                "transcript": payload["text"] if mode in {"long","srt"} else take.transcript,
                "metrics": output_metadata,"subtitle":subtitle_url,
                "project":f"/api/files/session/{session_path.name}"}

    def generate_session(self, mode, payload, session, sampling, progress, on_codes=None):
        if mode == "tts":
            result = self.engine.speak(payload["text"], session, sampling, on_codes, progress)
        elif mode == "dialogue":
            result = self.engine.scene([s for s in payload["text"].splitlines() if s.strip()],
                                      references=session.references, sampling=sampling, progress=progress,
                                      session=session, on_codes=on_codes,generation_mode=payload.get('generation_mode','together'),
                                      split_long_text=payload.get('split_long_text',False),chunk_bytes=payload.get('chunk_bytes',300))
        elif mode == "revise":
            result = self.engine.revise(session, payload["note"], sampling, on_codes, progress)
        elif mode == "scene":
            result = self.engine.scene([s for s in payload["text"].splitlines() if s.strip()], payload.get("scene", ""),
                                       payload.get("characters", ""), session.references, sampling, progress,
                                       session=session, on_codes=on_codes,generation_mode=payload.get('generation_mode','together'),
                                       split_long_text=payload.get('split_long_text',False),chunk_bytes=payload.get('chunk_bytes',300))
        elif mode == "long":
            result = self.engine.long_form(payload["text"], session, sampling, progress, on_codes,
                                           payload.get("chunk_bytes",300))
        elif mode == "text_stream":
            result = self.engine.stream_text(payload["text"], payload.get("lookahead", ""),
                                             session, sampling, on_codes, progress)
        elif mode == "infill":
            if not session.takes:
                raise ValueError("请选择包含原始音频的会话")
            options={}
            if payload.get('target_speaker') is not None:
                options['target_speaker']=integer_option(payload['target_speaker'],'目标角色')
            if payload.get('expected_view') is not None:options['expected_view']=payload['expected_view']
            take = self.engine.infill(session.takes[-1], payload["text"], payload["start_seconds"],
                                      payload["end_seconds"], sampling, payload.get("duration_frames"), progress,**options)
            result = session.append(take)
        else:
            raise ValueError(f"Unknown generation mode {mode}")
        return result
