import numpy as np
import pytest

from drama3_t8.asr_reference import reference_asr_audio, reference_interval


def test_complete_and_selected_audio_have_exact_reference_sample_boundaries():
    audio=np.arange(2000).reshape(2,1000)
    whole,interval=reference_asr_audio(audio,100)
    assert np.array_equal(whole,audio) and interval=={'start_seconds':0.,'end_seconds':10.}
    selected,interval=reference_asr_audio(audio,100,1.23,4.56)
    assert np.array_equal(selected,audio[:,123:456])
    assert interval=={'start_seconds':1.23,'end_seconds':4.56}


@pytest.mark.parametrize('start,end',[(float('nan'),None),(0,float('inf')),(-1,None),(2,1),(1,1)])
def test_invalid_intervals_rejected_before_upload(start,end):
    with pytest.raises(ValueError):reference_interval(start,end)


def test_empty_and_out_of_bounds_intervals_rejected():
    audio=np.ones((1,1000))
    for start,end in [(10,None),(0,11),(.0001,.0002)]:
        with pytest.raises(ValueError):reference_asr_audio(audio,100,start,end)
