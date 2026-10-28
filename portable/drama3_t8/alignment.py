"""Local word alignment; load the auxiliary model only for the requested operation."""
from pathlib import Path
import gc
import torch
import comfy.model_management as mm
from .audio import audio_waveform


def align_audio(model_path, waveform, sample_rate, transcript, language="Chinese"):
    wave = audio_waveform(waveform, sample_rate)
    from .prompts import spoken_text
    transcript=spoken_text(transcript)
    if not transcript.strip():
        raise ValueError("Word alignment requires a nonempty spoken transcript")
    path=Path(model_path)
    if path.is_file():
        if path.suffix.lower()!=".safetensors":
            raise ValueError("单文件对齐模型应为转换后的 .safetensors")
        from .qwen3_asr.asr import load_qwen3_asr
        from .qwen3_asr.align import align
        model=load_qwen3_asr(str(path))
        if not getattr(model.model,"aligner",False):
            raise ValueError("请选择 Qwen3-ForcedAligner 对齐模型，而不是 ASR 转写模型")
        items=align(model,wave,sample_rate,transcript)
        return [{"text":text,"start":float(start),"end":float(end)} for text,start,end in items]
    if not Path(model_path).is_dir():
        raise FileNotFoundError("缺少 Qwen3-ForcedAligner-0.6B 模型目录")
    from .vendor_qwen import Qwen3ForcedAligner
    mm.unload_all_models()
    mm.soft_empty_cache()
    device = mm.get_torch_device()
    model = Qwen3ForcedAligner.from_pretrained(str(model_path), dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                                             device_map=str(device), local_files_only=True)
    try:
        wave = wave.detach().cpu().mean(0)
        result = model.align(audio=(wave.numpy(), sample_rate), text=transcript, language=language)[0]
        return [{"text":i.text,"start":i.start_time,"end":i.end_time} for i in result.items]
    finally:
        del model
        gc.collect()
        mm.soft_empty_cache()
