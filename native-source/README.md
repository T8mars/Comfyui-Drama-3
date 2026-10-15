# R2T2 native C API

Same-process Windows x64 / CUDA 12.8 backend used by the DRAMA nodes and studio. No Python worker, HTTP inference service or Python ABI dependency.

The adapted source and notices are in this directory. Use MSVC 2022, CMake 3.26+, CUDA 12.8 and the pinned llama.cpp checkout:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout ad6c66839af3c5646fba8c6c2e2087a1e4e38948
# Build llama.cpp shared libraries with GGML_CUDA=ON for your GPU architecture.
cmake -S native-source -B build -A x64 -DLLAMA_CPP_DIR="<llama.cpp source>" -DLLAMA_BUILD_DIR="<matching llama.cpp build>"
cmake --build build --config Release
```

`CMakeLists.txt` expects the matching build's `llama_cpp/src/Release`, `llama_cpp/tools/mtmd/Release` and `llama_cpp/ggml/src/Release` layout. Set the paths for your build accordingly. Place `drama_r2t2.dll`, `llama.dll`, `mtmd.dll`, `ggml.dll`, `ggml-base.dll` and required CUDA DLLs under `r2t2-runtime/lib/`; CUDA and CPU ggml backend plugins belong in `r2t2-runtime/backends/`. Alternatively point `DRAMA_R2T2_NATIVE_DIR` at this layout.

The released binary was validated on an RTX 5090 Laptop. Rebuild the CUDA backend for GPU architectures not supported by that build. Preserve Apache-2.0, MIT and NVIDIA CUDA distribution notices in `licenses/`. The Confucius model weights have a separate NetEase Youdao model license.
