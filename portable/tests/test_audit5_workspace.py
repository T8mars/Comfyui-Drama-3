from pathlib import Path
import shutil

import pytest
import torch
import app as application

import drama3_t8.studio as module
from drama3_t8.state import Session, Take, Reference, save_session
from drama3_t8.studio import Studio


def take():
    frames = torch.ones(4, 11, dtype=torch.long)
    frames[:, 0] = 248100
    return Take('Original', 'Original', frames, metadata={'elapsed_seconds': 0, 'length_limit_reached': False})


@pytest.mark.parametrize('stage', ['audio', 'preview', 'session', 'metadata', 'database', 'subtitle'])
def test_failed_generation_removes_only_its_new_result_files(tmp_path, monkeypatch, stage):
    with Studio(tmp_path) as studio:
        original = Session(takes=(take(),)); studio.register_session(original)
        archive = studio.data / 'sessions' / (original.session_id + '.safetensors')
        old_bytes = archive.read_bytes()
        old_preview = studio.output / 'previews' / (original.session_id + '_0.wav')
        old_preview.parent.mkdir(); old_preview.write_bytes(b'old complete preview')
        wave = {'waveform': torch.zeros(1, 1, 100), 'sample_rate': 44100}
        monkeypatch.setattr(studio, 'generate_session', lambda mode, payload, session, *a: session.append(take()))
        monkeypatch.setattr(studio.engine, 'decode', lambda *a: wave)
        monkeypatch.setattr(module, 'generate_srt', lambda engine, text, session, *a: (session.append(take()), wave, '1\n00:00:00,000 --> 00:00:01,000\nHello\n'))
        real_audio, real_copy, real_save, real_text, real_connect = module.sf.write, shutil.copyfile, module.save_session, Path.write_text, studio.connect
        def audio(path, *a, **k):
            if stage == 'audio':
                Path(path).write_bytes(b'partial audio'); raise OSError('injected audio failure')
            return real_audio(path, *a, **k)
        def copy(source, target, *a, **k):
            if stage == 'preview':
                Path(target).write_bytes(b'partial preview'); raise OSError('injected preview failure')
            return real_copy(source, target, *a, **k)
        def save(session, path):
            real_save(session, path)
            if stage == 'session': raise OSError('injected session failure')
        def text(path, *a, **k):
            if (stage == 'metadata' and path.suffix == '.json') or (stage == 'subtitle' and path.suffix == '.srt'):
                path.write_bytes(b'partial text'); raise OSError('injected text failure')
            return real_text(path, *a, **k)
        class Connection:
            def __init__(self): self.db = real_connect()
            def __enter__(self): self.db.__enter__(); return self
            def __exit__(self, *a): return self.db.__exit__(*a)
            def execute(self, sql, parameters=()):
                if stage == 'database' and sql.startswith('INSERT INTO sessions'): raise OSError('injected database failure')
                return self.db.execute(sql, parameters)
        monkeypatch.setattr(module.sf, 'write', audio); monkeypatch.setattr(shutil, 'copyfile', copy)
        monkeypatch.setattr(module, 'save_session', save); monkeypatch.setattr(Path, 'write_text', text); monkeypatch.setattr(studio, 'connect', Connection)
        job = studio.submit({'mode': 'srt' if stage == 'subtitle' else 'tts', 'text': 'Test', 'session_id': original.session_id})
        recovered = studio.submit({'mode': 'unload'}); studio.pending.join()
        assert studio.get_job(job)['status'] == 'failed'
        assert studio.get_job(recovered)['status'] == 'done'
        assert archive.read_bytes() == old_bytes
        assert old_preview.read_bytes() == b'old complete preview'
        assert {p for p in studio.output.rglob('*') if p.is_file()} == {old_preview}
        assert list((studio.data / 'sessions').iterdir()) == [archive]
        assert len(studio.sessions()) == 1


def test_failed_transcription_metadata_does_not_leave_orphan_text(tmp_path, monkeypatch):
    with Studio(tmp_path) as studio:
        monkeypatch.setattr(module, 'read_audio', lambda *a: (torch.zeros(1, 100), 16000))
        monkeypatch.setattr(studio.engine, 'transcribe_detailed', lambda *a, **k: {'text': 'Transcript'})
        original = Path.write_text
        def write(path, *a, **k):
            if path.name.endswith('.asr.json'):
                path.write_bytes(b'partial metadata'); raise OSError('metadata disk failure')
            return original(path, *a, **k)
        monkeypatch.setattr(Path, 'write_text', write)
        job = studio.submit({'mode': 'transcribe', 'input_id': 'input.wav'}); studio.pending.join()
        assert studio.get_job(job)['status'] == 'failed'
        assert not list(studio.output.iterdir())


def test_failed_reference_registration_does_not_leave_orphan_archive(tmp_path, monkeypatch):
    with Studio(tmp_path) as studio:
        monkeypatch.setattr(module, 'read_audio', lambda *a: (torch.zeros(1, 100), 16000))
        monkeypatch.setattr(studio.engine, 'encode_reference', lambda *a: Reference('Hello', torch.ones(10, 4, dtype=torch.long)))
        real_connect = studio.connect
        class Connection:
            def __init__(self): self.db = real_connect()
            def __enter__(self): self.db.__enter__(); return self
            def __exit__(self, *a): return self.db.__exit__(*a)
            def execute(self, sql, parameters=()):
                if sql.startswith('INSERT INTO voice_references'): raise OSError('reference database failure')
                return self.db.execute(sql, parameters)
        monkeypatch.setattr(studio, 'connect', Connection)
        job = studio.submit({'mode': 'reference', 'input_id': 'input.wav', 'transcript': 'Hello'}); studio.pending.join()
        assert studio.get_job(job)['status'] == 'failed'
        assert not list((studio.data / 'references').iterdir())
        assert studio.references() == []


def test_nonfinite_asr_result_is_rejected_before_publishing_result_files(tmp_path, monkeypatch):
    with Studio(tmp_path) as studio:
        monkeypatch.setattr(module, 'read_audio', lambda *a: (torch.zeros(1, 100), 16000))
        monkeypatch.setattr(studio.engine, 'transcribe_detailed', lambda *a, **k: {'text': 'Transcript', 'elapsed_seconds': float('nan')})
        job = studio.submit({'mode': 'transcribe', 'input_id': 'input.wav'}); studio.pending.join()
        assert studio.get_job(job)['status'] == 'failed'
        assert not list(studio.output.iterdir())


def test_failed_alignment_write_removes_partial_metadata_and_preserves_session(tmp_path, monkeypatch):
    import drama3_t8.alignment as alignment
    with Studio(tmp_path) as studio:
        session = Session(takes=(take(),)); studio.register_session(session)
        monkeypatch.setattr(studio.engine, 'decode', lambda *a: {'waveform': torch.zeros(1, 1, 100), 'sample_rate': 44100})
        monkeypatch.setattr(alignment, 'align_audio', lambda *a: [{'text': 'Original', 'start': 0, 'end': 1}])
        original = Path.write_text
        def write(path, *a, **k):
            if path.name.endswith('.alignment.json'):
                path.write_bytes(b'partial alignment'); raise OSError('alignment disk failure')
            return original(path, *a, **k)
        monkeypatch.setattr(Path, 'write_text', write)
        job = studio.submit({'mode': 'align', 'session_id': session.session_id}); studio.pending.join()
        assert studio.get_job(job)['status'] == 'failed'
        assert studio.get_session(session.session_id).takes[0].transcript == 'Original'
        assert not list(studio.output.rglob('*.json'))
