"""Local recording segmentation and optional, editable speaker suggestions."""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy.cluster.hierarchy import cut_tree, fcluster, linkage
from scipy.spatial.distance import pdist
import soxr

from .creative_audio import analyze_audio, audio_array, file_digest


def mono_16k(waveform, sample_rate):
    wave = audio_array(waveform).mean(0)
    analyze_audio(wave, sample_rate)
    if sample_rate != 16000:
        wave = soxr.resample(wave, sample_rate, 16000)
    return np.asarray(wave, dtype=np.float32)


def speech_regions(waveform, sample_rate, vad_path, *, check_cancel=None, padding_seconds=.08):
    """Collect every FireRed interval before its rolling history expires."""
    from .r2t2.vad import FireRedOnnxVAD
    check_cancel = check_cancel or (lambda: None)
    pcm = mono_16k(waveform, sample_rate)
    detector = FireRedOnnxVAD(Path(vad_path))
    intervals, seen = [], set()
    # Trailing silence closes a final utterance; clamp all suggestions to input.
    padded = np.pad(pcm, (0, 9600))
    for start in range(0, len(padded), 640):
        check_cancel()
        detector.feed(padded[start:start + 640])
        for first, last in detector.speech_intervals:
            key = (int(first), int(last))
            if key not in seen:
                seen.add(key)
                intervals.append(key)
    if detector.speech_start_sample is not None:
        intervals.append((detector.speech_start_sample, len(pcm)))
    result = []
    for first, last in sorted(intervals):
        first = max(0.0, first / 16000 - padding_seconds)
        last = min(len(pcm) / 16000, last / 16000 + padding_seconds)
        if last - first < .15:
            continue
        if result and first <= result[-1]["end"]:
            result[-1]["end"] = max(last, result[-1]["end"])
        else:
            result.append({"start": first, "end": last, "text": "", "role": "待分配"})
    return {"segments": result, "timing_method": "firered_acoustic_estimate",
            "duration_seconds": len(pcm) / 16000, "manual_review_required": True}


class SpeakerEmbedder:
    """3D-Speaker CAM++ ONNX, using the documented sherpa feature contract.

    No Sherpa native extension or GPU model is required. Global-mean fbank uses
    80 bins, high-frequency -400Hz and non-snipped edges. Inputs are normalized
    float PCM. This reproduces the existing Sherpa reference extractor.
    """
    def __init__(self, model_path, threads=2):
        import onnxruntime as ort
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError("请选择 CAM++ 中英说话人 ONNX 模型")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        self.metadata = self.session.get_modelmeta().custom_metadata_map
        input_info = self.session.get_inputs()
        if (len(input_info) != 1 or input_info[0].shape[-1] != 80
                or self.metadata.get("framework") != "3d-speaker"
                or self.metadata.get("feature_normalize_type") != "global-mean"
                or self.metadata.get("sample_rate") != "16000"
                or self.metadata.get("normalize_samples") != "1"):
            raise ValueError("说话人模型不是受支持的 16k / 80维 / global-mean CAM++ 格式")
        self.input_name = input_info[0].name
        self.path, self.sha256 = str(path.resolve()), file_digest(path)

    def embedding(self, waveform, sample_rate):
        import kaldi_native_fbank as knf
        pcm = mono_16k(waveform, sample_rate)
        if len(pcm) < 3200 or float(np.max(np.abs(pcm))) < 1e-6:
            raise ValueError("说话人参考太短或为静音，请选择清晰语音")
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = 16000
        options.frame_opts.dither = 0
        options.frame_opts.snip_edges = False
        options.mel_opts.num_bins = 80
        options.mel_opts.high_freq = -400
        bank = knf.OnlineFbank(options)
        bank.accept_waveform(16000, pcm.tolist())
        bank.input_finished()
        features = np.asarray([bank.get_frame(i) for i in range(bank.num_frames_ready)], dtype=np.float32)
        features -= features.mean(0, keepdims=True)
        value = self.session.run(None, {self.input_name: features[None]})[0].reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or not math.isfinite(norm) or norm < 1e-8:
            raise RuntimeError("说话人模型返回无效向量")
        return value / norm


def cluster_speakers(embeddings, *, num_speakers=0, threshold=.45, max_speakers=16):
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("无效说话人向量")
    if type(num_speakers) is not int or not 0 <= num_speakers <= max_speakers or not 0 < threshold < 2:
        raise ValueError("说话人数或聚类阈值无效")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("说话人向量不能为空")
    values = values / norms
    if len(values) == 1 or num_speakers == 1:
        return [0] * len(values)
    distances = np.clip(pdist(values.astype(np.float64), metric="cosine"), 0, 2)
    tree = linkage(distances, method="average")
    if num_speakers:
        raw = cut_tree(tree, n_clusters=[min(num_speakers, len(values))])[:, 0]
    else:
        raw = fcluster(tree, t=threshold, criterion="distance")
        if len(set(raw)) > max_speakers:
            raw = fcluster(tree, t=max_speakers, criterion="maxclust")
    # Stable names follow the first audible appearance, not scipy's label order.
    identity, result = {}, []
    for label in raw:
        if int(label) not in identity:
            identity[int(label)] = len(identity)
        result.append(identity[int(label)])
    return result


def suggest_speakers(waveform, sample_rate, regions, model_path, *, num_speakers=0,
                     threshold=.45, window_seconds=2.0, check_cancel=None):
    if not 1 <= float(window_seconds) <= 10:
        raise ValueError("说话人分析窗口应为1到10秒")
    check_cancel = check_cancel or (lambda: None)
    started = time.perf_counter()
    pcm = mono_16k(waveform, sample_rate)
    extractor = SpeakerEmbedder(model_path)
    windows, vectors = [], []
    for source_index, region in enumerate(regions):
        first, last = float(region["start"]), float(region["end"])
        if not math.isfinite(first + last) or not 0 <= first < last <= len(pcm) / 16000 + 1e-5:
            raise ValueError("语音选区超出原录音")
        count = max(1, math.ceil((last - first) / window_seconds))
        for index in range(count):
            start = first + (last - first) * index / count
            end = first + (last - first) * (index + 1) / count
            if end - start < .2:
                continue
            check_cancel()
            vectors.append(extractor.embedding(pcm[round(start * 16000):round(end * 16000)], 16000))
            windows.append({"start": start, "end": end, "source_index": source_index})
    if not windows:
        return {"segments": [], "manual_review_required": True, "timing_method": "speaker_window_estimate"}
    labels = cluster_speakers(vectors, num_speakers=num_speakers, threshold=threshold)
    result = []
    for window, label in zip(windows, labels):
        role = f"说话人 {label + 1}"
        if result and result[-1]["role"] == role and window["start"] <= result[-1]["end"] + .1:
            result[-1]["end"] = window["end"]
        else:
            result.append({**window, "role": role, "speaker": label, "text": ""})
    return {"segments": result, "speakers": len(set(labels)), "timing_method": "speaker_window_estimate",
            "manual_review_required": True, "overlap_resolution": "not_performed",
            "model_path": extractor.path, "model_sha256": extractor.sha256,
            "elapsed_seconds": time.perf_counter() - started,
            "note": "自动角色为聚类建议，换人边界为窗口估计；请校正角色和边界后生成剧本"}


def asr_rows(result, *, source_duration=None):
    """Keep recognizer boundaries honest; no invented per-word timestamps."""
    rows = []
    for index, segment in enumerate(result.get("segments") or []):
        text = str(segment.get("text", segment.get("stable_text", ""))).strip()
        if not text:
            continue
        start = segment.get("start", segment.get("start_seconds", 0))
        end = segment.get("end", segment.get("end_seconds", source_duration))
        rows.append({"id": f"L{index + 1:02}", "role": "待分配", "original": text, "original_text": text,
                     "subtitle": text, "readout_override": None, "subtitle_override": None,
                     "start_seconds": start, "end_seconds": end,
                     "timing_method": result.get("timing_method", "segment_estimate")})
    if not rows and result.get("text", "").strip():
        rows.append({"id": "L01", "role": "待分配", "original": result["text"].strip(), "original_text": result["text"].strip(),
                     "subtitle": result["text"].strip(), "readout_override": None, "subtitle_override": None, "start_seconds": 0,
                     "end_seconds": source_duration or result.get("duration_seconds"),
                     "timing_method": "whole_recording"})
    return rows
