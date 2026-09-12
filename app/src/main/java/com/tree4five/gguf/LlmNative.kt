package com.tree4five.gguf

import java.util.concurrent.atomic.AtomicBoolean

/**
 * Thin JNI facade over libggufllm.so (vendored llama.cpp, tag b3621).
 *
 * One handle per loaded model file. The native side serializes every call on
 * a per-engine mutex; the service layer additionally funnels all calls through
 * a single-threaded dispatcher so llama contexts are never touched concurrently.
 *
 * Embedding surfaces:
 *  - [getEmbeddingDim] returns -1 when the model's token_embd table is not
 *    accessible: callers must then fall back to text-based context handling.
 *  - [embedTokenRows] mean-of-rows lookup (input-embedding space, microseconds).
 *  - [embedText] mode 0 = mean of token rows; mode 1 = mean of hidden states
 *    (pooling MEAN forward pass, slower).
 *  - [generateFromEmbeddings] injects `count` rows (soft prompt) then, if
 *    `followupPrompt` is non-empty, decodes that text before sampling.
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

    external fun tokenize(handle: Long, text: String, addSpecial: Boolean, parseSpecial: Boolean): IntArray

    external fun tokenToPiece(handle: Long, token: Int): String

    /**
     * Input-embedding lookup: flat array of tokens.size * [dim] floats holding
     * the dequantized token_embd row of each id. Feeding these rows to
     * [generateFromEmbeddings] reproduces the text generation path exactly.
     */
    external fun embedTokenRows(handle: Long, tokens: IntArray): FloatArray

    /** mode 0 = mean of token rows, mode 1 = mean of hidden states (pooling MEAN). */
    external fun embedText(handle: Long, text: String, mode: Int): FloatArray

    /** Generates from `prompt` (ChatML-formatted by the caller). Returns the token count. */
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
}
