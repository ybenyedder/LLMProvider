package com.tree4five.gguf

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Thin JNI facade over `libggufllm.so` — the pure-Rust RustingStrata v2.0 GGUF inference engine.
 *
 * Features:
 *  - Zero-copy mmap GGUF parser + Q8_K integer SIMD kernels (ARM64 NEON / x86_64 AVX2)
 *  - Batched prompt prefill + prefix KV-cache reuse
 *  - N-gram speculative decoding for greedy / low-temperature acceleration
 *  - Lossless token_embd lookup & soft-prompt generation parity
 */
object LlmNative {

    /** Generation listener; methods are invoked from the generating thread. */
    interface LlmCallback {
        fun onToken(token: String)
        fun onError(message: String)
    }

    private val loaded = AtomicBoolean(false)

    /** Loads libggufllm once; returns false if the native library is unavailable. */
    fun ensureLoaded(): Boolean {
        if (loaded.get()) return true
        return synchronized(this) {
            if (loaded.get()) {
                true
            } else {
                try {
                    System.loadLibrary("ggufllm")
                    loaded.set(true)
                    true
                } catch (_: Throwable) {
                    false
                }
            }
        }
    }

    /** Loads `path` (GGUF) with the given context size and thread count. Returns 0 on failure. */
    external fun loadModel(path: String, nCtx: Int, nThreads: Int): Long

    /** Frees the engine; the handle must never be used afterwards. */
    external fun freeModel(handle: Long)

    /** Dimension of the token-embedding table, or -1 if unavailable. */
    external fun getEmbeddingDim(handle: Long): Int

    /** Context window in tokens: the GGUF train context capped by the runtime window. */
    external fun getContextLength(handle: Long): Int

    external fun tokenize(handle: Long, text: String, addSpecial: Boolean, parseSpecial: Boolean): IntArray

    external fun tokenToPiece(handle: Long, token: Int): String

    /**
     * Input-embedding lookup: flat array of tokens.size * [dim] floats holding
     * the dequantized token_embd row of each id.
     */
    external fun embedTokenRows(handle: Long, tokens: IntArray): FloatArray

    /** mode 0 = mean of token rows, mode 1 = mean of hidden states (pooling MEAN). */
    external fun embedText(handle: Long, text: String, mode: Int): FloatArray

    /** Generates from `prompt`. Returns the produced token count. */
    external fun generate(
        handle: Long,
        prompt: String,
        nPredict: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        callback: LlmCallback?
    ): Int

    /** Generates from injected embeddings (soft prompt) plus optional text follow-up. */
    external fun generateFromEmbeddings(
        handle: Long,
        vectors: FloatArray,
        count: Int,
        followupPrompt: String,
        nPredict: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        callback: LlmCallback?
    ): Int

    /** Asks the running generation loop to stop at the next token boundary. */
    external fun stop(handle: Long)

    /** Configures speculative N-gram decoding, prefix KV-cache reuse, and repeat penalty. */
    external fun configureEngine(
        handle: Long,
        specDecoding: Boolean,
        prefixCache: Boolean,
        repeatPenalty: Float
    )

    /** Clears the active KV cache and prefix token history. */
    external fun clearKvCache(handle: Long)

    /** Returns live telemetry JSON from the Rust engine (architecture, quant, SIMD, tok/s, etc.). */
    external fun getEngineInfoJson(handle: Long): String
}
