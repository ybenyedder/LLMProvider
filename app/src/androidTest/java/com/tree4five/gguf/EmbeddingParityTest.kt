package com.tree4five.gguf

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters

/**
 * On-device proof that the embeddings pipeline is lossless:
 *
 *  1. token_embd is accessible (dim == [EXPECTED_DIM] for Qwen2.5-0.5B).
 *  2. Greedy generation from a plain text prompt and from the corresponding
 *     injected token-embedding rows produce EXACTLY the same text: the soft
 *     prompt path never degrades the model output.
 *  3. embedText mode 0 (mean of token rows, no forward pass) is fast enough
 *     for per-turn context ingestion; mode 1 (pooling MEAN) is timed for info.
 *
 * The model is the one already present in the app's files/ directory; the
 * instrumentation targets the app package so it has direct file access.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class EmbeddingParityTest {

    companion object {
        /** Hidden sizes per known model family (null = accept any dim > 0). */
        private fun expectedDimFor(modelName: String): Int? = when {
            modelName.contains("0.5b", ignoreCase = true) -> 896
            modelName.contains("1.5b", ignoreCase = true) -> 1536
            else -> null
        }

        private const val N_CTX = 2048
        private const val N_PREDICT = 64

        private var handle: Long = 0L
        private var expectedDim: Int? = null

        /** Preferred: a model already in files/; fallback: /sdcard/Download. */
        private fun findModel(): File {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            context.filesDir.listFiles { _, name -> name.endsWith(".gguf", true) }
                ?.firstOrNull()?.let { return it }
            val download = File("/sdcard/Download")
            download.listFiles { _, name -> name.endsWith(".gguf", true) }
                ?.firstOrNull()?.let { return it }
            return File(context.filesDir, "no-model.gguf")
        }

        @JvmStatic
        @BeforeClass
        fun loadModel() {
            val model = findModel()
            assertTrue("no .gguf found in files/ or /sdcard/Download", model.exists())
            assertTrue(LlmNative.ensureLoaded())
            expectedDim = expectedDimFor(model.name)
            val threads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
            handle = LlmNative.loadModel(model.absolutePath, N_CTX, threads)
            assertTrue("loadModel failed", handle != 0L)
        }

        private fun currentDim(): Int = LlmNative.getEmbeddingDim(handle)

        @JvmStatic
        @AfterClass
        fun freeModel() {
            if (handle != 0L) {
                LlmNative.freeModel(handle)
                handle = 0L
            }
        }

        private fun generateText(prompt: String): String {
            val sb = StringBuilder()
            LlmNative.generate(
                handle = handle,
                prompt = PromptManager.formatPrompt(prompt),
                nPredict = N_PREDICT,
                temperature = 0f,
                topK = 40,
                topP = 0.95f,
                callback = object : LlmNative.LlmCallback {
                    override fun onToken(token: String) { sb.append(token) }
                    override fun onError(message: String) { sb.append("[error:").append(message).append("]") }
                }
            )
            return sb.toString()
        }

        private fun generateFromRows(prompt: String): String {
            val dim = currentDim()
            val tokens = LlmNative.tokenize(handle, PromptManager.formatPrompt(prompt), true, true)
            val rows = LlmNative.embedTokenRows(handle, tokens)
            assertEquals(tokens.size * dim, rows.size)
            val sb = StringBuilder()
            LlmNative.generateFromEmbeddings(
                handle = handle,
                vectors = rows,
                count = tokens.size,
                followupPrompt = "",
                nPredict = N_PREDICT,
                temperature = 0f,
                topK = 40,
                topP = 0.95f,
                callback = object : LlmNative.LlmCallback {
                    override fun onToken(token: String) { sb.append(token) }
                    override fun onError(message: String) { sb.append("[error:").append(message).append("]") }
                }
            )
            return sb.toString()
        }
    }

    @Test
    fun step1_embeddingDimIsExposed() {
        val dim = currentDim()
        val expected = expectedDim
        if (expected != null) {
            assertEquals("unexpected embedding dim", expected, dim)
        } else {
            assertTrue("embedding dim must be positive, got $dim", dim > 0)
        }
    }

    @Test
    fun step2_embedTokenRowsMatchesTextPathGreedy() {
        val prompts = listOf(
            "What is the OS info of this device?",
            "List the installed apps for me.",
            "Launch the Gmail app.",
            "Summarize what you can do in one short sentence.",
            "Remember that I prefer concise answers."
        )
        prompts.forEachIndexed { index, prompt ->
            val viaText = generateText(prompt)
            val viaEmbd = generateFromRows(prompt)
            assertEquals(
                "prompt #$index diverged\ntext: ${viaText.take(120)}\nembd: ${viaEmbd.take(120)}",
                viaText, viaEmbd
            )
        }
    }

    @Test
    fun step3_embedTextMode0IsFastAndShapeCorrect() {
        val long = ("The agent records every step of the task in the session log, " +
            "so a later turn can retrieve the relevant excerpt without replaying " +
            "the whole conversation. ").repeat(18)  // ~2800 chars
        assertTrue("length=${long.length}", long.length in 2500..3200)

        val t0 = System.nanoTime()
        val v = LlmNative.embedText(handle, long, 0)
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertEquals(currentDim(), v.size)
        assertTrue("mode 0 took ${ms}ms for ${long.length} chars (budget: 50ms)", ms < 50)
        // A mean of rows must not be identically zero.
        assertTrue(v.any { it != 0f })
    }

    @Test
    fun step4_embedTextMode1TimesHiddenStates() {
        val text = "Latent memory facts about the user for retrieval."
        val t0 = System.nanoTime()
        val v = LlmNative.embedText(handle, text, 1)
        val ms = (System.nanoTime() - t0) / 1_000_000
        assertEquals(currentDim(), v.size)
        // Informational budget: mode 1 pays a forward pass. Fails only if it is
        // wildly off scale (>60s means something is pathologically wrong).
        assertTrue("mode 1 took ${ms}ms", ms < 60_000)
    }

    @Test
    fun step5_generateFromEmbeddingsAcceptsTextFollowup() {
        // Soft prompt (rows of a short prefix) followed by a ChatML question:
        // this is the harness usage pattern (history as latent prefix, the
        // FSM question as text).
        val prefix = "The device is a Samsung tablet running Android."
        val tokens = LlmNative.tokenize(handle, prefix, true, true)
        val rows = LlmNative.embedTokenRows(handle, tokens)
        val question = PromptManager.formatPrompt("What kind of device is this? Answer in one word.")
        val sb = StringBuilder()
        val produced = LlmNative.generateFromEmbeddings(
            handle = handle,
            vectors = rows,
            count = tokens.size,
            followupPrompt = question,
            nPredict = 16,
            temperature = 0f,
            topK = 40,
            topP = 0.95f,
            callback = object : LlmNative.LlmCallback {
                override fun onToken(token: String) { sb.append(token) }
                override fun onError(message: String) { sb.append("[error:").append(message).append("]") }
            }
        )
        assertTrue("generation produced no tokens", produced > 0 || sb.isNotEmpty())
    }
}
