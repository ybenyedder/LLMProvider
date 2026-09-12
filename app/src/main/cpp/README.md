# Native code

- `gguf_llm_jni.cpp` : our own JNI bridge to llama.cpp (lib `ggufllm`) — model loading,
  text generation, text→embedding (mean of token-embedding rows), input-embedding lookup
  and generation from injected embeddings (soft prompt).
- `llama.cpp/` : vendored copy of upstream **tag b3621** (commit f91fc5639be1e272194303ea26e1d4c226ec981b,
  2024-08-25, MIT license, see `llama.cpp/LICENSE`). Only `ggml/ src/ include/ common/ cmake/` and the
  root `CMakeLists.txt` are kept; examples/tests/tools were trimmed. Do not upgrade casually:
  the JNI code targets this exact API surface (legacy `llama_sample_*` functions, `llama_get_model_tensor`).
