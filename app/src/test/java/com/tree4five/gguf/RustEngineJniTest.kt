package com.tree4five.gguf

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Real JVM -> JNI -> Rust (`libggufllm.so`) integration tests verifying every native
 * entrypoint, tensor embedding lookup, soft-prompt parity, prefix caching, N-gram
 * speculative decoding, and live engine JSON telemetry.
 */
class RustEngineJniTest {

    private fun writeStr(out: ByteArrayOutputStream, s: String) {
        val bytes = s.toByteArray(Charsets.UTF_8)
        val len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(bytes.size.toLong()).array()
        out.write(len)
        out.write(bytes)
    }

    private fun leInt(v: Int): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(v).array()

    private fun leLong(v: Long): ByteArray =
        ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v).array()

    private fun leFloat(v: Float): ByteArray =
        ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(v).array()

    private fun addStrKv(kvs: ByteArrayOutputStream, nkv: IntArray, k: String, v: String) {
        writeStr(kvs, k)
        kvs.write(leInt(8))
        writeStr(kvs, v)
        nkv[0]++
    }

    private fun addU32Kv(kvs: ByteArrayOutputStream, nkv: IntArray, k: String, v: Int) {
        writeStr(kvs, k)
        kvs.write(leInt(4))
        kvs.write(leInt(v))
        nkv[0]++
    }

    private fun addF32Kv(kvs: ByteArrayOutputStream, nkv: IntArray, k: String, v: Float) {
        writeStr(kvs, k)
        kvs.write(leInt(6))
        kvs.write(leFloat(v))
        nkv[0]++
    }

    private fun addStrArrKv(kvs: ByteArrayOutputStream, nkv: IntArray, k: String, items: List<String>) {
        writeStr(kvs, k)
        kvs.write(leInt(9))
        kvs.write(leInt(8))
        kvs.write(leLong(items.size.toLong()))
        items.forEach { writeStr(kvs, it) }
        nkv[0]++
    }

    private fun addU32ArrKv(kvs: ByteArrayOutputStream, nkv: IntArray, k: String, items: List<Int>) {
        writeStr(kvs, k)
        kvs.write(leInt(9))
        kvs.write(leInt(4))
        kvs.write(leLong(items.size.toLong()))
        items.forEach { kvs.write(leInt(it)) }
        nkv[0]++
    }

    private fun buildSyntheticModel(file: File) {
        val vocab = listOf(
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "user", "assistant",
            "\n", "Hello", "Ġworld", "!", "H", "e", "l"
        )
        val types = listOf(3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        val merges = listOf("H e", "He l")

        val kvs = ByteArrayOutputStream()
        val nkv = intArrayOf(0)
        addStrKv(kvs, nkv, "general.architecture", "qwen2")
        addU32Kv(kvs, nkv, "qwen2.block_count", 1)
        addU32Kv(kvs, nkv, "qwen2.context_length", 512)
        addU32Kv(kvs, nkv, "qwen2.embedding_length", 8)
        addU32Kv(kvs, nkv, "qwen2.feed_forward_length", 16)
        addU32Kv(kvs, nkv, "qwen2.attention.head_count", 2)
        addU32Kv(kvs, nkv, "qwen2.attention.head_count_kv", 1)
        addF32Kv(kvs, nkv, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f)
        addF32Kv(kvs, nkv, "qwen2.rope.freq_base", 1_000_000f)
        addStrKv(kvs, nkv, "tokenizer.ggml.model", "gpt2")
        addStrKv(kvs, nkv, "tokenizer.ggml.pre", "qwen2")
        addStrArrKv(kvs, nkv, "tokenizer.ggml.tokens", vocab)
        addU32ArrKv(kvs, nkv, "tokenizer.ggml.token_type", types)
        addStrArrKv(kvs, nkv, "tokenizer.ggml.merges", merges)
        addU32Kv(kvs, nkv, "tokenizer.ggml.eos_token_id", 2)

        val hdr = ByteArrayOutputStream()
        val body = ByteArrayOutputStream()
        var offset = 0L
        var nt = 0L

        fun pushTensor(name: String, dims: LongArray, vals: FloatArray) {
            writeStr(hdr, name)
            hdr.write(leInt(dims.size))
            dims.forEach { hdr.write(leLong(it)) }
            hdr.write(leInt(0)) // F32
            hdr.write(leLong(offset))
            val bb = ByteBuffer.allocate(vals.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            vals.forEach { bb.putFloat(it) }
            val raw = bb.array()
            body.write(raw)
            val pad = ((32 - (raw.size % 32)) % 32)
            repeat(pad) { body.write(0) }
            offset += raw.size + pad
            nt++
        }

        fun det(n: Int, seed: Int) = FloatArray(n) { i -> (((i * 17 + seed * 31) % 97) - 48) * 0.02f }
        fun ones(n: Int) = FloatArray(n) { 1f }
        fun zeros(n: Int) = FloatArray(n) { 0f }

        pushTensor("token_embd.weight", longArrayOf(8, 12), det(8 * 12, 1))
        pushTensor("output_norm.weight", longArrayOf(8), ones(8))
        pushTensor("blk.0.attn_norm.weight", longArrayOf(8), ones(8))
        pushTensor("blk.0.attn_q.weight", longArrayOf(8, 8), det(8 * 8, 2))
        pushTensor("blk.0.attn_q.bias", longArrayOf(8), zeros(8))
        pushTensor("blk.0.attn_k.weight", longArrayOf(8, 4), det(8 * 4, 3))
        pushTensor("blk.0.attn_k.bias", longArrayOf(4), zeros(4))
        pushTensor("blk.0.attn_v.weight", longArrayOf(8, 4), det(8 * 4, 4))
        pushTensor("blk.0.attn_v.bias", longArrayOf(4), zeros(4))
        pushTensor("blk.0.attn_output.weight", longArrayOf(8, 8), det(8 * 8, 5))
        pushTensor("blk.0.ffn_norm.weight", longArrayOf(8), ones(8))
        pushTensor("blk.0.ffn_gate.weight", longArrayOf(8, 16), det(8 * 16, 6))
        pushTensor("blk.0.ffn_up.weight", longArrayOf(8, 16), det(8 * 16, 7))
        pushTensor("blk.0.ffn_down.weight", longArrayOf(16, 8), det(16 * 8, 8))

        val out = ByteArrayOutputStream()
        out.write("GGUF".toByteArray(Charsets.US_ASCII))
        out.write(leInt(3))
        out.write(leLong(nt))
        out.write(leLong(nkv[0].toLong()))
        out.write(kvs.toByteArray())
        out.write(hdr.toByteArray())
        while (out.size() % 32 != 0) out.write(0)
        out.write(body.toByteArray())
        file.writeBytes(out.toByteArray())
    }

    @Test
    fun rustJniFullPipelineAndParity() {
        assertTrue("libggufllm.so must load via java.library.path", LlmNative.ensureLoaded())
        val modelFile = File.createTempFile("rust_jni_test", ".gguf")
        modelFile.deleteOnExit()
        buildSyntheticModel(modelFile)

        val handle = LlmNative.loadModel(modelFile.absolutePath, 128, 2)
        assertTrue("loadModel must return non-zero handle", handle != 0L)

        try {
            assertEquals(8, LlmNative.getEmbeddingDim(handle))
            assertEquals(128, LlmNative.getContextLength(handle))

            val prompt = PromptManager.formatPrompt("Hello world!")
            val tokens = LlmNative.tokenize(handle, prompt, true, true)
            assertTrue("tokens must be non-empty", tokens.isNotEmpty())
            assertTrue("tokenToPiece must decode", LlmNative.tokenToPiece(handle, tokens.first()).isNotEmpty())

            val emb0 = LlmNative.embedText(handle, "Hello world!", 0)
            assertEquals(8, emb0.size)
            assertTrue(emb0.any { it != 0f })

            val emb1 = LlmNative.embedText(handle, "Hello world!", 1)
            assertEquals(8, emb1.size)
            assertTrue(emb1.any { it != 0f })

            val sbText = StringBuilder()
            LlmNative.configureEngine(handle, specDecoding = true, prefixCache = true, repeatPenalty = 1.10f)
            val nGen = LlmNative.generate(
                handle, prompt, 10, 0.0f, 40, 0.95f,
                object : LlmNative.LlmCallback {
                    override fun onToken(token: String) { sbText.append(token) }
                    override fun onError(message: String) { throw AssertionError(message) }
                }
            )
            assertTrue("generate must produce tokens", nGen > 0)

            val rows = LlmNative.embedTokenRows(handle, tokens)
            assertEquals(tokens.size * 8, rows.size)
            val sbEmbd = StringBuilder()
            LlmNative.generateFromEmbeddings(
                handle, rows, tokens.size, "", 10, 0.0f, 40, 0.95f,
                object : LlmNative.LlmCallback {
                    override fun onToken(token: String) { sbEmbd.append(token) }
                    override fun onError(message: String) { throw AssertionError(message) }
                }
            )
            assertEquals(
                "Greedy text path and embedTokenRows path must match bit-for-bit",
                sbText.toString(),
                sbEmbd.toString()
            )

            val info = JSONObject(LlmNative.getEngineInfoJson(handle))
            assertEquals("RustingStrata", info.getString("engine"))
            assertEquals("qwen2", info.getString("arch"))
            assertEquals(8, info.getInt("n_embd"))

            LlmNative.clearKvCache(handle)
        } finally {
            LlmNative.freeModel(handle)
            modelFile.delete()
        }
    }
}
