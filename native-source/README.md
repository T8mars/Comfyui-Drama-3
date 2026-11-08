# Windows R2T2 CUDA runtime

The bridge and all ggml libraries must use the pinned llama.cpp revision
`ad6c66839af3c5646fba8c6c2e2087a1e4e38948`. See [provenance](NOTICE-UPSTREAM.md)
and the included licenses. The release runtime contains no Python or weights.
The C API runs in the same process, without a Python worker, HTTP inference
service or Python ABI dependency.

The CUDA backend uses Visual Studio 2022 x64 and CUDA Toolkit 12.8. Its explicit
architecture list avoids a build that only works on the developer's GPU:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout ad6c66839af3c5646fba8c6c2e2087a1e4e38948
cmake -S llama.cpp -B llama-multi -G "Visual Studio 17 2022" -A x64 -T cuda=12.8 `
  -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CUDA=ON `
  "-DCMAKE_CUDA_ARCHITECTURES=50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real" `
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build llama-multi --config Release --target ggml-cuda --parallel 2
```

To rebuild the C API bridge against a matching shared llama/mtmd build:

```powershell
cmake -S native-source -B build -A x64 -DLLAMA_CPP_DIR="<llama.cpp source>" -DLLAMA_BUILD_DIR="<matching shared build>"
cmake --build build --config Release
```

`CMakeLists.txt` expects that build's `llama_cpp/src/Release`,
`llama_cpp/tools/mtmd/Release` and `llama_cpp/ggml/src/Release` library layout;
adapt those paths for a different build layout. Keep the pinned ABI consistent.

Keep the runtime layout `lib/drama_r2t2.dll`, its matching dependencies and
`backends/ggml-cuda.dll`. The DLL's actual images, rather than the CMake cache,
must be inspected with CUDA's `cuobjdump --list-elf` and `--list-ptx`.

`lib/runtime-compat.json` has schema `1`, the full `backend_sha256`, integer
`cuda_real_arches` and `cuda_virtual_arches` obtained from those images. For
example, `86` represents compute capability 8.6; the architecture-specific
`sm_120a` image is accepted only for 12.0. Custom builds need their own verified
manifest. Copying the release manifest beside a different DLL is rejected.

The Python boundary checks this identity and the selected GPU before loading
ggml, because a missing CUDA kernel can abort the entire host process. Image
inspection and simulated preflight checks are distinct from physical GPU
inference: release validation states which actual GPU was tested. A compatible
NVIDIA CUDA driver is required for PTX JIT.
Preserve the Apache-2.0, MIT and NVIDIA CUDA distribution notices in `licenses/`.
Confucius model weights retain their separate NetEase Youdao model license.
