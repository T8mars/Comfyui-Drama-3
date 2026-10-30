# Audio loudness processing

LUFS processing invokes FFmpeg as a subprocess. Pitch-preserving duration
adjustment uses librosa. The resolver uses an explicit `DRAMA_FFMPEG_PATH`, the
local `imageio-ffmpeg` executable, or an installed FFmpeg on PATH; it does not
install packages or download binaries at runtime.

The complete Windows environment includes the unmodified official
`imageio_ffmpeg-0.6.0-py3-none-win_amd64.whl` contents:

- Python wrapper: imageio contributors, Copyright 2019–2025 imageio,
  BSD-2-Clause; see `LICENSE-IMAGEIO-FFMPEG` and the original dist-info license.
- Windows executable: FFmpeg 7.1 essentials build by gyan.dev,
  Copyright 2000–2024 the FFmpeg developers, GPL version 3 or later as reported
  by its actual `-L` output; see `LICENSE-FFMPEG`.
- Wheel SHA-256: `02fa47c83703c37df6bfe4896aab339013f62bf02c5ebf2dce6da56af04ffc0a`.
- Executable SHA-256: `2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3`.
- Executable: `imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`.

Upstream material and source/build references:

- https://pypi.org/project/imageio-ffmpeg/0.6.0/
- https://github.com/imageio/imageio-ffmpeg/tree/v0.6.0
- https://github.com/imageio/imageio-binaries/tree/master/ffmpeg
- https://github.com/GyanD/codexffmpeg/releases/tag/7.1
- https://github.com/FFmpeg/FFmpeg/commit/b08d7969c5
- https://www.gyan.dev/ffmpeg/builds/
- https://ffmpeg.org/legal.html

The executable's configuration is preserved in `FFMPEG-BUILDCONF.txt`.
Its GPL license is distinct from the Python wrapper's BSD license. Upstream
notices for the selected FFmpeg build and its included libraries continue to
apply. No FFmpeg executable enters the GitHub code-only update ZIP.
