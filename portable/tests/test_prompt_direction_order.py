"""Reference-conditioned direction and completed-history row contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

import drama3_t8
from drama3_t8 import prompts
from drama3_t8.native.prompt import build_prompt
from drama3_t8.state import Reference, Take


@pytest.fixture
def tokenizer():
    package = Path(drama3_t8.__file__).resolve().parent
    return Tokenizer.from_file(str(package / "native/fish/tokenizer.json"))


@pytest.fixture
def config():
    return SimpleNamespace(tower=SimpleNamespace(num_codebooks=10, vocab_size=4096),
                           semantic_start_token_id=248099, semantic_end_token_id=252194)


@pytest.fixture
def references():
    first = torch.arange(30, dtype=torch.int16, device="cpu").reshape(10, 3)
    second = (torch.arange(70, dtype=torch.int64, device="cpu").reshape(10, 7) + 101)[:, 1:6]
    return (Reference("<|speaker:0|> 第一参考。", first),
            Reference("<|speaker:1|> 第二参考。", second))


@pytest.fixture
def history():
    codes = torch.arange(40, dtype=torch.int64, device="cpu").reshape(10, 4) + 300
    frames = torch.cat((codes[:1] + 248099, codes)).T.contiguous()
    return (Take("<|speaker:0|> [deadpan] 我把会议改", "我把会议改", frames),)


def _bytes(rows):
    assert rows.device.type == "cpu" and rows.dtype == torch.int64
    return bytes(rows.contiguous().view(torch.uint8).reshape(-1).tolist())


def _text_spans(rows, tokenizer, text):
    """Locate exact separately encoded text, including zero audio-code columns."""
    expected = tokenizer.encode(text).ids
    ids = rows[:, 0].tolist()
    return [(i, i + len(expected)) for i in range(len(ids) - len(expected) + 1)
            if ids[i:i + len(expected)] == expected
            and not bool(torch.count_nonzero(rows[i:i + len(expected), 1:]))]


def _user_bodies(rows, tokenizer):
    starts = _text_spans(rows, tokenizer, "<|im_start|>user\n")
    end_id = tokenizer.token_to_id("<|im_end|>")
    ids = rows[:, 0].tolist()
    result = []
    for _, body_start in starts:
        body_end = next(i for i in range(body_start, len(ids)) if ids[i] == end_id)
        result.append(rows[body_start:body_end])
    return result


@pytest.mark.parametrize("protocol", ("tts", "line", "stream"))
@pytest.mark.parametrize("reference_count", (0, 1, 2))
def test_direction_reference_and_history_contract(
        tokenizer, config, references, history, protocol, reference_count):
    refs = references[:reference_count]
    original_codes = [ref.codes.clone() for ref in refs]
    original_history = history[0].frames.clone()
    direction = "  [calm]\n 保留句尾笑意。\t "
    text = "<|speaker:0|> [deadpan] 成了午睡。"
    lookahead = "<|speaker:0|> [curious] 下一句。"
    lines = [text, "<|speaker:1|> [excited] 全国推广。"]
    instruction = {"tts": prompts.TTS_INSTRUCTION, "line": prompts.LINE_INSTRUCTION,
                   "stream": prompts.STREAM_INSTRUCTION}[protocol]

    segments = prompts.system_segments(refs, instruction)
    literals = [item for item in segments if isinstance(item, str) and item == direction]
    assert not literals
    code_objects = [item for item in segments if torch.is_tensor(item)]
    assert len(code_objects) == len(refs)
    assert all(item is ref.codes for item, ref in zip(code_objects, refs))

    options = dict(references=refs, takes=history, direction=direction)
    if protocol == "tts":
        rows = prompts.tts_prompt(tokenizer, config, text, **options)
        expected_current = tokenizer.encode(prompts.directed_text(text,direction)).ids
    elif protocol == "line":
        rows = prompts.scene_prompt(tokenizer, config, lines, **options)
        expected_current = tokenizer.encode(("\n" + prompts.SEPARATOR + "\n").join(prompts.directed_text(line,direction,scene=True) for line in lines)).ids
    else:
        rows = prompts.streaming_prompt(tokenizer, config, text, lookahead, **options)
        expected_current = (tokenizer.encode(prompts.directed_text(text,direction)).ids + tokenizer.encode(prompts.END_OF_TURN).ids
                            + tokenizer.encode(lookahead).ids)

    # Delivery belongs to the exact current user body, never ahead of refs.
    system_end = rows[:, 0].tolist().index(tokenizer.token_to_id("<|im_end|>"))
    system = tokenizer.decode(rows[:system_end,0].tolist(),skip_special_tokens=False)
    assert prompts.direction_tags(direction) not in system
    semantic = ((rows[:, 0] >= config.semantic_start_token_id)
                & (rows[:, 0] <= config.semantic_end_token_id))
    audio_positions = torch.nonzero(semantic, as_tuple=False)[:, 0].tolist()
    first_audio = audio_positions[0]
    header = prompts.REFERENCE_INSTRUCTION if refs else instruction
    header_spans = _text_spans(rows, tokenizer, header)
    assert len(header_spans) == 1 and header_spans[0][1] <= system_end
    offset = 0
    for number, ref in enumerate(refs, 1):
        text_spans = _text_spans(rows, tokenizer,
            f"\n\n[Reference {number}]\nText: {ref.transcript}\nSpeech: ")
        assert len(text_spans) == 1 and text_spans[0][1] <= audio_positions[offset]
        offset += ref.codes.shape[1]
    if refs and protocol != "tts":
        task_spans = _text_spans(rows, tokenizer, instruction)
        assert len(task_spans) == 1
        assert audio_positions[offset - 1] < task_spans[0][0] < system_end

    # The complete audio-row stream preserves every reference followed by the
    # actual completed take, with no reordering, replacement, duplication or loss.
    expected_audio = [torch.cat((ref.codes[:1].long() + config.semantic_start_token_id,
                                 ref.codes.long())).T for ref in refs]
    expected_audio.append(history[0].frames)
    assert _bytes(rows[semantic]) == _bytes(torch.cat(expected_audio))
    assert all(torch.equal(ref.codes, old) for ref, old in zip(refs, original_codes))
    assert _bytes(history[0].frames) == _bytes(original_history)

    bodies = _user_bodies(rows, tokenizer)
    assert len(bodies) == 2  # one completed history turn and the actual request.
    assert bodies[0][:, 0].tolist() == tokenizer.encode(history[0].prompt_text).ids
    assert bodies[1][:, 0].tolist() == expected_current
    assert all(not bool(torch.count_nonzero(body[:, 1:])) for body in bodies)
    ending = tokenizer.encode(prompts.MESSAGE_END).ids + tokenizer.encode(prompts.GENERATION_START).ids
    assert rows[-len(ending):, 0].tolist() == ending
    assert not bool(torch.count_nonzero(rows[-len(ending):, 1:]))


def test_no_direction_tts_matches_native_reference_prompt(tokenizer, config, references, history):
    text = "<|speaker:0|> 成了午睡。"
    expected = build_prompt(tokenizer, config, text,
        [(ref.transcript, ref.codes) for ref in references],
        [(take.prompt_text, take.frames[:, 1:].T) for take in history])
    assert _bytes(prompts.tts_prompt(tokenizer, config, text, references, history)) == _bytes(expected)
