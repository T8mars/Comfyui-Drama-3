import pytest
from drama3_t8.subtitles import Cue, parse_srt, write_srt


def test_srt_unicode_multiline_and_millisecond_roundtrip():
    cues=[Cue(0,1.25,"你好，世界。"),Cue(2.5,4.125,"<|speaker:1|> Hello.\n第二行")]
    assert parse_srt(write_srt(cues))==cues


def test_invalid_subtitle_interval_does_not_become_silent_success():
    with pytest.raises(ValueError,match="positive duration"):
        parse_srt("1\n00:00:03,000 --> 00:00:01,000\nHello")
