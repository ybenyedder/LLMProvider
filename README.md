# Tree4Five LLM Provider 🌲5️⃣ — RustingStrata v2.0 Edition

**Tree4Five LLM Provider** is a high-performance, 100% offline local LLM inference server for Android, powered by a **pure-Rust inference engine derived from [RustingStrata](https://github.com/ybenyedder/RustingStrata)** (`libggufllm.so`). It exposes a backward-compatible cross-app AIDL service (`ILLMService`) **and** an optional embedded OpenAI/Ollama HTTP API server (`127.0.0.1:11435`).

## 🚀 What's New in v2.0.0 (Pure-Rust Engine)
- **Bye-bye `llama.cpp` — Hello Pure Rust (`RustingStrata`):** Complete replacement of the C++/CMake engine with a zero-C-dependency Rust engine (`app/src/main/rust/ggufllm`) compiled natively for `arm64-v8a` and `x86_64`.
- **SIMD Quantized Kernels (`Q8_K`):** Hand-tuned ARM64 NEON (`ARMv8.0-A` safe) and x86_64 AVX2+FMA integer dot products across 25 GGML formats (`Q4_0`, `Q4_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, `IQ4_XS`, `IQ2_*`, `IQ3_*`, `TQ1_0`, `TQ2_0`, `F16`, `BF16`, `F32`, etc.).
- **Batched Prompt Prefill:** Multi-token batched matrix projection (`rows_multi`) reads weights once per chunk instead of per token.
- **Prefix KV-Cache Reuse:** Automatically reuses shared KV prefixes across turns so multi-turn prompts and system prompts resume instantly.
- **N-gram Speculative Decoding:** Zero-memory `ngram-simple` + `ngram-cache` drafter proposes up to 4 candidate tokens verified in a single batched pass during greedy decoding.
- **Broad GGUF Architecture Support:** `qwen2`, `qwen2moe`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp`, `flint`, `llama`, `mistral`, `smollm`, `gemma`, `gemma2`, `gemma4`, `phi3`, and `nomic-bert`.
- **Embedded Local HTTP API (`127.0.0.1:11435`):** Optional one-tap toggle exposing `/v1/chat/completions` (SSE streaming), `/v1/embeddings`, `/v1/models`, `/api/generate`, `/api/chat`, `/api/tags`, and `/health`.
- **Redesigned Dark Telemetry UI:** Live Rust engine metrics (`arch`, `quant`, `SIMD`, `prefill tok/s`, `decode tok/s`, `N-gram hits`, `KV usage`), 6 model presets (`Qwen3-0.6B`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen3-1.7B`, `SmolLM2-360M`, `nomic-embed`), cancellable atomic `.part` downloads with `GGUF` magic validation, speed/ETA readout, model deletion/reloading, and interactive benchmark playground.

## Build Instructions

Prerequisites: JDK 17, Android SDK + NDK 27+, Rust (`aarch64-linux-android` & `x86_64-linux-android` targets).

```bash
# Run Rust engine unit & parity tests
cd app/src/main/rust/ggufllm && cargo test --release

# Run JVM + JNI integration tests and build APK / AAB
cd /path/to/LLMProvider
./gradlew testDebugUnitTest assembleDebug assembleRelease bundleRelease
```

Gradle automatically invokes `scripts/build-rust-android.sh` during `preBuild` to compile `libggufllm.so` with 16 KiB ELF alignment (`-Wl,-z,max-page-size=16384`).

## AIDL Integration (Backward Compatible v1.0 → v2.0)

Bind to `com.tree4five.gguf.ACTION_LLM_SERVICE` in package `com.tree4five.gguf`:

```kotlin
val intent = Intent("com.tree4five.gguf.ACTION_LLM_SERVICE").apply {
    setPackage("com.tree4five.gguf")
}
bindService(intent, connection, Context.BIND_AUTO_CREATE)

// Standard v1.x call (still 100% compatible)
llmService.generateTextStream("Explain Quantum Physics", callback)

// New v2.0 parameterized call + telemetry
llmService.generateTextStreamWithParams(prompt, 512, 0.0f, 40, 0.95f, 1.10f, true, callback)
val metricsJson = llmService.engineMetricsJson
```

## License
MIT
