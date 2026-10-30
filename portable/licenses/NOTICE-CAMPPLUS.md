# Optional CAM++ speaker embedding

The optional `campplus-zh-en.onnx` model is used on CPU for speaker grouping and
reference-similarity assistance. It is not a calibrated identity probability or
a perceptual speech-quality score.

- Upstream: Alibaba DAMO Academy / 3D-Speaker contributors.
- Model: `iic/speech_campplus_sv_zh_en_16k-common_advanced`, revision `v1.0.0`.
- Original model card: https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced
- Code: https://github.com/modelscope/3D-Speaker
- Upstream model-card license: Apache License 2.0. See `LICENSE-CAMPPLUS`.
- ONNX distribution: https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
- Artifact SHA-256: `aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2`
- Bytes: 28,281,164. The distributed ONNX file is unmodified.

The T8 CPU adapter reads the model's ONNX metadata and follows the sherpa-onnx
16 kHz, 80-bin Kaldi fbank configuration (including `high_freq=-400`, no dither,
`snip_edges=False`, and global mean normalization). The adapter uses
`kaldi-native-fbank` and `onnxruntime`; it does not require a sherpa-onnx wheel.

Speech boundaries and speaker groups are editable suggestions. Overlapping
speech is not separated. Chinese and English are the upstream model's stated
languages; other languages and short/noisy recordings are not guaranteed.
