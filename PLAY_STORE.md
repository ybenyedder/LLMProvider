# Google Play Store Listing Information

## App Name
LLM Provider: Local AI Engine (by tree4five)

## Short Description
Pure-Rust local AI engine (GGUF) on device. Fast NEON, embeddings & API.

## Full Description
LLM Provider is a powerful, on-device AI inference engine powered by our pure-Rust RustingStrata engine (`libggufllm.so`). All inference, embeddings, and prompt formatting run 100% locally on your Android phone or tablet with zero cloud dependencies and complete privacy.

**Key Features:**
* **100% Offline Pure-Rust Engine:** Built from the ground up in Rust (no C++ runtime bloat), featuring ARM64 NEON `Q8_K` quantized SIMD kernels, batched prefill, prefix KV-cache reuse, and N-gram speculative decoding.
* **Wide GGUF Compatibility:** Supports Qwen3, Qwen2.5, Qwen3.5, Llama, SmolLM2, Mistral, Gemma, Phi-3, and Nomic-BERT models across 25 GGML quantization formats.
* **Text Embeddings & Soft Prompts:** Turn any supported model into an ultra-fast embedding provider and continue generation directly from quantized int8 embedding vectors via AIDL.
* **Universal AIDL + Local HTTP API:** Other Android apps can bind via AIDL (`com.tree4five.gguf.ACTION_LLM_SERVICE`) or call the optional local OpenAI/Ollama-compatible HTTP server (`127.0.0.1:11435`).
* **One-Tap Model Presets & Telemetry Dashboard:** Download Qwen3-0.6B, Qwen2.5-0.5B/1.5B, SmolLM2-360M, or nomic-embed with live speed/ETA tracking, atomic verification, and real-time `tok/s` telemetry.

## What's New (2.0.0)
* Complete migration from `llama.cpp` to the pure-Rust RustingStrata v2.0 inference engine (`libggufllm.so`).
* Added ARM64 NEON `Q8_K` SIMD kernels, batched prompt prefill, prefix KV-cache reuse, and N-gram speculative decoding.
* Added optional local OpenAI/Ollama-compatible HTTP API server (`127.0.0.1:11435`) alongside backward-compatible AIDL.
* Completely redesigned Material Dark UI with real-time telemetry, 6 presets, download speed/ETA/cancel, model management, and benchmark playground.

## Tags
AI, Machine Learning, Offline Chat, Rust, GGUF, Embeddings, Privacy, Productivity

## Category
Productivity / Tools

## Content Rating
Everyone (No user-generated content shared online)
