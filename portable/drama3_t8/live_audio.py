"""CPU codec worker that emits real audio before GPU speech generation finishes."""
import queue
import copy
from dataclasses import replace
import threading
import traceback
from pathlib import Path

import soundfile as sf
import torch

from .native.codec import FRAME_LENGTH, SAMPLE_RATE
from .native.loader import load_codec


class LiveAudio:
    def __init__(self, codec_path, folder, chunk_frames=32, on_chunk=None):
        if type(chunk_frames) is not int or chunk_frames < 1:
            raise ValueError("Live chunk_frames must be a positive integer")
        self.codec_path = codec_path
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.on_chunk = on_chunk
        self.incoming = queue.Queue()
        self.error = None
        self.aborted = threading.Event()
        self._state_lock = threading.Lock()
        self._finished = False
        self.thread = threading.Thread(target=self.run, name="Drama-live-codec", daemon=True)
        self.thread.start()

    def append(self, frame):
        with self._state_lock:
            if self.error is not None:
                raise self.error
            if self._finished:
                raise RuntimeError("Live audio worker is already finished")
            # The worker runs later; a caller may immediately reuse its tensor.
            self.incoming.put(frame.detach().to(device="cpu", copy=True))

    def append_scene(self,take):
        """Publish a completed isolated scene with its actual codec boundaries."""
        from .state import Take
        if not isinstance(take,Take) or not (take.metadata.get('decode_sources') or 'scene_chunking' in take.metadata):
            raise ValueError('Scene preview requires an isolated scene take')
        if 'scene_chunking' in take.metadata:
            from .scene_chunks import validate_scene
            validate_scene(take)
        snapshot=replace(take,frames=take.frames.detach().to(device='cpu',copy=True),
            metadata=copy.deepcopy(take.metadata))
        with self._state_lock:
            if self.error is not None:raise self.error
            if self._finished:raise RuntimeError('Live audio worker is already finished')
            self.incoming.put(('scene',snapshot))

    def finish(self, abort=False):
        with self._state_lock:
            if abort:
                self.aborted.set()
            if not self._finished:
                self._finished = True
                self.incoming.put(None)
        self.thread.join()
        # Abort is cleanup for an error already propagating in the caller. A
        # secondary decoder fault must not replace that original cancellation.
        if self.error is not None and not abort:
            raise self.error

    def run(self):
        codec = None
        try:
            with torch.inference_mode():
                codec = load_codec(str(self.codec_path)).model
                codes, emitted, index = [], 0, 0

                def write_chunk(chunk):
                    nonlocal index
                    target=self.folder/f'{index:05}.wav'
                    temporary=target.with_suffix('.part')
                    try:
                        sf.write(temporary,chunk,SAMPLE_RATE,format='WAV',subtype='PCM_16')
                        temporary.replace(target)
                    finally:temporary.unlink(missing_ok=True)
                    if self.on_chunk is not None:self.on_chunk(target)
                    index+=1

                def emit():
                    nonlocal emitted
                    if len(codes) * FRAME_LENGTH <= emitted:
                        return
                    # Decode the complete causal prefix so chunk boundaries cannot lose earlier context.
                    waveform = codec.decode(torch.stack(codes, dim=1)[None], torch.float32)[0]
                    chunk = waveform[emitted:].numpy()
                    write_chunk(chunk)
                    emitted = waveform.shape[-1]

                while True:
                    frame = self.incoming.get()
                    if self.aborted.is_set():
                        break
                    if frame is None:
                        emit()
                        break
                    if isinstance(frame,tuple) and frame[0]=='scene':
                        emit()
                        codes,emitted=[],0
                        from types import SimpleNamespace
                        if 'scene_chunking' in frame[1].metadata:
                            from .scene_chunks import decode_scene as decode_isolated
                        else:
                            from .dialogue import decode_isolated
                        def decode(source):
                            wave=codec.decode(source[None],torch.float32)
                            return {'waveform':wave[:,None],'sample_rate':SAMPLE_RATE}
                        audio,_=decode_isolated(SimpleNamespace(decode=decode),frame[1])
                        wave=audio['waveform'][0,0]
                        for start in range(0,len(wave),self.chunk_frames*FRAME_LENGTH):
                            if self.aborted.is_set():break
                            write_chunk(wave[start:start+self.chunk_frames*FRAME_LENGTH].numpy())
                        continue
                    codes.append(frame)
                    if len(codes) % self.chunk_frames == 0:
                        emit()
        except BaseException as exc:
            # Keep useful stack locations, without pinning codec weights via
            # locals in an exited decode/callback frame retained by self.error.
            traceback.clear_frames(exc.__traceback__)
            self.error = exc
        finally:
            codec = None
