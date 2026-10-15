"""SRT import, actual audio timing and subtitle export."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def parse_time(value):
    match = re.fullmatch(r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if match is None:
        raise ValueError(f"Invalid subtitle timestamp: {value}")
    hours, minutes, seconds, millis = map(int, match.groups())
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid subtitle minute or second")
    return hours*3600 + minutes*60 + seconds + millis/1000


def parse_srt(text):
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip().lstrip("\ufeff")):
        lines = block.splitlines()
        if lines and lines[0].strip().isdigit():
            lines = lines[1:]
        if len(lines) < 2 or "-->" not in lines[0]:
            raise ValueError("Each subtitle cue needs a timestamp and text")
        start, end = map(parse_time, lines[0].split("-->", 1))
        if end <= start or (cues and start < cues[-1].start):
            raise ValueError("Subtitle intervals must be ordered and have positive duration")
        cues.append(Cue(start, end, "\n".join(lines[1:])))
    return cues


def format_time(seconds):
    total = max(0, round(seconds*1000))
    return f"{total//3600000:02}:{total//60000%60:02}:{total//1000%60:02},{total%1000:03}"


def write_srt(cues):
    return "\n\n".join(f"{i}\n{format_time(c.start)} --> {format_time(c.end)}\n{c.text}" for i,c in enumerate(cues,1)) + "\n"
