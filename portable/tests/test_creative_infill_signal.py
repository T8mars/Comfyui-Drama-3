import numpy as np
import pytest
from drama3_t8 import creative_infill_quality as guard


@pytest.mark.parametrize("source,target,expected",[
    ("导演，我的感情戏刚酝酿好，你怎么开始算加班费了？","导演，我的感情戏刚酝酿好，你怎么开始发红包了？",True),
    ("今天回家","今天准时回家",True),("开会","摸鱼",True),("123","132",True),
    ("abcdef","bdf",False),("今天准时回家","今天回家",False),("今天回家","",False),
    ("Hello world.","Hello, world!",False),("HELLO","hello",False),("ＡＢＣ１２３","ABC123",False),
    ("今天回家","今天 回家！",False),("abc","",False),("", "新词",True),
])
def test_new_alnum_requires_insert_replace_not_delete_or_punctuation(source,target,expected):
    assert guard.needs_new_spoken_alnum(source,target) is expected


@pytest.mark.parametrize("kind",["zero","near-silent"])
def test_new_speech_extreme_quiet_nonempty_insert_is_rejected(kind):
    pcm=np.full((1,44100),0 if kind=="zero" else 2e-5,dtype=np.float32)
    result=guard.analyze_infill_insert("原词","新词",pcm,0,44100)
    assert result["rejected"] and result["rms_dbfs"]<-80 and result["peak_dbfs"]<-60


@pytest.mark.parametrize("source,target",[("原来多两个词","原来两个词"),("abcd","bd"),("你好","你好！"),("ABC","abc"),("你好","")])
def test_delete_only_punctuation_case_and_legitimate_empty_text_not_rejected(source,target):
    result=guard.analyze_infill_insert(source,target,np.zeros((1,1000)),0,1000)
    assert not result["requires_new_spoken_alnum"] and not result["rejected"]


@pytest.mark.parametrize("amplitude",[1e-4,2e-4,.01])
def test_audible_low_level_or_exact_rms_threshold_is_not_rejected(amplitude):
    result=guard.analyze_infill_insert("原词","新词",np.full((1,1),amplitude,dtype=np.float64),0,1)
    assert not result["rejected"]


def test_peak_at_or_above_threshold_prevents_signal_guard_rejection():
    for peak in (.001,.002):
        pcm=np.zeros((1,44100),dtype=np.float64);pcm[0,0]=peak
        result=guard.analyze_infill_insert("原词","新词",pcm,0,44100)
        assert result["rms_dbfs"]<-80 and not result["rejected"]


def test_nonempty_prefix_cannot_mask_silent_insert_and_empty_insert_remains_allowed():
    pcm=np.concatenate([np.full((1,1000),.2),np.zeros((1,2000))],axis=1)
    assert guard.analyze_infill_insert("原词","新词",pcm,1000,2000)["rejected"]
    result=guard.analyze_infill_insert("原来多两个词","原来两个词",pcm,1000,0)
    assert not result["rejected"] and result["inserted_samples"]==0


def test_new_required_speech_with_empty_insert_is_rejected():
    result=guard.analyze_infill_insert("原词","新词",np.zeros((1,1000)),1000,0)
    assert result["requires_new_spoken_alnum"] and result["rejected"]
    assert result["reason"]=="new_spoken_material_but_empty_insert"


def test_no_full_insert_array_and_no_input_mutation(monkeypatch):
    pcm=np.full((1,3*guard.BLOCK_SAMPLES+77),2e-5,dtype=np.float32);before=pcm.copy();sizes=[]
    square=guard.np.square
    def bounded(block,*args,**kwargs):
        sizes.append(block.size);assert block.size<=guard.BLOCK_SAMPLES
        return square(block,*args,**kwargs)
    monkeypatch.setattr(guard.np,"square",bounded)
    assert guard.analyze_infill_insert("原词","新词",pcm,0,pcm.shape[-1])["rejected"]
    assert sizes and np.array_equal(pcm,before)


@pytest.mark.parametrize("start,count",[(-1,1),(0,-1),(1,10),(True,1),(0,True)])
def test_invalid_or_boolean_ranges_are_not_silently_accepted(start,count):
    with pytest.raises(ValueError):guard.analyze_infill_insert("旧","新",np.zeros((1,10)),start,count)


