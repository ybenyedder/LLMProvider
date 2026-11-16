# Tree4Five LLM Provider 🌲5️⃣ — Auralis + RustingStrata v2.1 Edition

**Tree4Five LLM Provider** is a high-performance, 100% offline local LLM inference server for Android, powered by a **pure-Rust inference engine derived from [RustingStrata](https://github.com/ybenyedder/RustingStrata)** (`libggufllm.so`) and dressed in a **minimal OLED Auralis UI**. It exposes a backward-compatible cross-app AIDL service (`ILLMService`) **and** an optional embedded OpenAI/Ollama HTTP API server (`127.0.0.1:11435`).

## ✨ What's New in v2.1.0 (Auralis UI) + v2.0.0 Engine
- **Auralis Pure-OLED Design System:** Pure `#000000` black canvas, `#111114` tonal cards with `1dp` hairline `#24242C` borders, `#F4F4F5` Liner pearl pill CTAs, crimson `#FA233B` resonance accents, custom waveform emblem, and Android 15/16 edge-to-edge inset handling.
- **4-Tile Bionic Telemetry + 3-Tab Segmented Switcher:** Live throughput (`tok/s`), native process RSS & model footprint, KV context / prefix cache hits, and speculative N-gram acceptance rate above animated `Studio`, `Modèles`, and `Moteur` panels.
- **Pure-Rust Engine (`libggufllm.so`):** Zero-C/C++ `RustingStrata` inference core (`arm64-v8a` NEON & `x86_64` AVX2+FMA `Q8_K` kernels across 25 GGML quant formats, 16 KiB ELF alignment).
- **Embedded Local HTTP API (`127.0.0.1:11435`):** One-tap OpenAI (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) & Ollama (`/api/generate`, `/api/chat`, `/api/tags`, `/health`) gateway with instant cURL copy button.

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
