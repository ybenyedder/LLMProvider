# Vendored llama.cpp

- **Version**: v0.5.0 (upstream tag, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`,
  ggml 0.25.1). Previous vendored version was b3621 (2024-07).
- **Source**: https://github.com/ggml-org/llama.cpp (shallow clone, files copied).
- **Why the bump**: 14 months of upstream fixes (arm64 kernel work, CPU-feature
  dispatch via getauxval, API cleanups). The tablet garbage-output episode that
  triggered it (512× `!` on the SM-P610 while x86_64 was fine) turned out to be a
  **corrupted model copy in the app's filesDir** (same size, different bytes —
  caught by the FNV-1a hash logged at load), NOT the old kernels; the b3621 build
  generated correctly on the tablet once fed the intact file. The bump stays:
  it is the current upstream and both devices validate on it.

## Pruned (not vendored, nothing references them with our options)

- Backends: `ggml/src/ggml-{cuda,vulkan,opencl,sycl,cann,hip,musa,metal,hexagon,openvino,virtgpu,webgpu,zdnn,zendnn,et,blas,rpc}` — CPU-only build (GGML_CUDA/... OFF, `n_gpu_layers = 0` in the JNI).
- Top-level dirs: `app, benches, ci, common, conversion, docs, examples, grammars,
  media, models, pocs, requirements, scripts, skills, tests, tools, vendor/vendor note` —
  `LLAMA_BUILD_COMMON/APP/EXAMPLES/TESTS/TOOLS/SERVER/MTMD` are all set OFF by
  `../CMakeLists.txt`; `add_subdirectory(vendor)` is unconditional so `vendor/` IS vendored.
- `common/` is no longer built or linked (the JNI only uses `llama.h` + `ggml.h`).

## Verify after changing the vendored tree

```bash
cmake -S llama.cpp -B /tmp/hostllama -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF \
  -DGGML_LLAMAFILE=OFF
cmake --build /tmp/hostllama -j   # must reach "Built target llama"
```
