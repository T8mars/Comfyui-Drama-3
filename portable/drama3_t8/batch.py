"""SRT workflow: keep speech context and export actual achieved subtitle timing."""
from dataclasses import replace

import torch

from .state import Session
from .subtitles import Cue, parse_srt, write_srt


def generate_srt(engine, text, session, sampling, progress=None):
    sampling.validate()
    cues = parse_srt(text)
    if not cues:
        raise ValueError("The SRT contains no subtitle cues")
    session = session or Session()
    timeline, actual, cursor, rate = [], [], 0, 44100
    for index, cue in enumerate(cues):
        history = session.takes if len(session.takes) < 3 else (session.takes[0], session.takes[-1])
        context = replace(session, takes=history)
        generated = engine.speak(cue.text, context, replace(sampling, seed=(sampling.seed+index) % 2**64), progress=progress)
        take = generated.takes[-1]
        session = session.append(take)
        audio = engine.decode(take.codes)
        wave = audio["waveform"][0,0]
        rate = audio["sample_rate"]
        start = max(cursor, round(cue.start * rate))
        if start > cursor:
            timeline.append(torch.zeros(start-cursor))
        timeline.append(wave)
        actual.append(Cue(start/rate,(start+len(wave))/rate,cue.text))
        cursor = start+len(wave)
    return session, {"waveform":torch.cat(timeline)[None,None],"sample_rate":rate}, write_srt(actual)
