import numpy as np
import pytest

from drama3_t8.creative_transcript import asr_rows, cluster_speakers, mono_16k


def test_speaker_suggestions_stable_by_first_appearance():
    vectors = [[1, 0, 0], [.99, .01, 0], [0, 1, 0], [.01, .99, 0], [1, 0, 0]]
    assert cluster_speakers(vectors, threshold=.2) == [0, 0, 1, 1, 0]
    assert cluster_speakers([[1, 0], [0, 1]], num_speakers=2) == [0, 1]
    assert cluster_speakers(vectors, num_speakers=1) == [0] * 5
    assert cluster_speakers([[1, 0]]) == [0]


def test_bad_speaker_vectors_do_not_create_fake_roles():
    with pytest.raises(ValueError):
        cluster_speakers([[0, 0], [1, 0]])
    with pytest.raises(ValueError):
        cluster_speakers([[np.nan, 0]])
    with pytest.raises(ValueError):
        cluster_speakers([[1, 0]], num_speakers=17)


def test_transcript_import_preserves_recognizer_precision_scope():
    rows = asr_rows({"text": "整段", "segments": [], "duration_seconds": 10})
    assert rows[0]["timing_method"] == "whole_recording"
    assert rows[0]["end_seconds"] == 10
    rows = asr_rows({"text": "一二", "segments": [
        {"text": "一", "start_seconds": .3, "end_seconds": 1.2},
        {"text": "二", "start_seconds": 2, "end_seconds": 3},
    ], "timing_method": "segment_estimate"})
    assert [row["original"] for row in rows] == ["一", "二"]
    assert rows[0]["start_seconds"] == .3
    assert rows[1]["role"] == "待分配"


def test_recording_resampling_does_not_modify_source_or_guess_audio_shape():
    source = np.ones((2, 4800), dtype=np.float32) * .1
    result = mono_16k(source, 48000)
    assert result.shape == (1600,)
    assert np.all(source == np.float32(.1))
    with pytest.raises(ValueError):
        mono_16k(np.array([np.inf]), 16000)
