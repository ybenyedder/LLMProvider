package com.tree4five.gguf

import android.content.Context
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.ServiceTestRule
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.ClassRule
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * End-to-end exercise of the embeddings pipeline through the AIDL binder:
 * exactly what harnessDroid does in production.
 *
 *  - getEmbeddingDim exposes the model space;
 *  - embedText returns one packed [scale][q8 x dim] vector;
 *  - beginEmbeddingInput/setEmbeddingChunk/generateFromEmbeddings upload
 *    q8 chunks and generate with a text followup;
 *  - dimension mismatch and unknown handles are rejected cleanly.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class EmbeddingServiceBinderTest {

    companion object {
        @ClassRule
        @JvmField
        val serviceRule = ServiceTestRule()

        private var service: ILLMService? = null
        private var dim: Int = -1

        private fun findModel(): File? {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            context.filesDir.listFiles { _, name -> name.endsWith(".gguf", true) }
                ?.firstOrNull()?.let { return it }
            File("/sdcard/Download").listFiles { _, name -> name.endsWith(".gguf", true) }
                ?.firstOrNull()?.let { return it }
            return null
        }

        @JvmStatic
        @org.junit.BeforeClass
        fun setUp() {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val model = findModel()
            assertNotNull("no .gguf found in files/ or /sdcard/Download", model)

            // The service reads the active model at onCreate.
            context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                .edit().putString("active_model", model!!.absolutePath).commit()

            val binder = serviceRule.bindService(Intent(context, LLMInferenceService::class.java))
            service = ILLMService.Stub.asInterface(binder)
            assertNotNull(service)

            // Wait for the async model load.
            val deadline = System.currentTimeMillis() + 180_000
            while (System.currentTimeMillis() < deadline) {
                dim = try { service!!.embeddingDim } catch (e: Exception) { -1 }
                if (dim > 0) break
                Thread.sleep(1000)
            }
            assertTrue("model never loaded (dim=$dim)", dim > 0)
        }
    }

    private fun generateWithEmbd(count: Int, vectors: FloatArray, followup: String, nPredict: Int): String {
        val handle = service!!.beginEmbeddingInput(count, dim)
        assertTrue("beginEmbeddingInput rejected count=$count dim=$dim", handle > 0)
        try {
            // 8 vectors per chunk (~12 KB at dim=1536), like the production client.
            val perVector = 4 + dim
            var start = 0
            while (start < count) {
                val chunk = minOf(8, count - start)
                val bytes = ByteArray(chunk * perVector)
                val buf = java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                for (t in 0 until chunk) {
                    buf.put(EmbeddingCodec.encodeOne(vectors.copyOfRange((start + t) * dim, (start + t + 1) * dim)))
                }
                service!!.setEmbeddingChunk(handle, start, bytes)
                start += chunk
            }

            val latch = CountDownLatch(1)
            var fullText = ""
            val callback = object : ILLMCallback.Stub() {
                override fun onTokenReceived(token: String) {}
                override fun onGenerationComplete(text: String) {
                    fullText = text
                    latch.countDown()
                }
            }
            service!!.generateFromEmbeddings(handle, count, followup, nPredict, 0f, callback)
            assertTrue("generation timed out", latch.await(180, TimeUnit.SECONDS))
            return fullText
        } finally {
            service!!.releaseEmbeddings(handle)
        }
    }

    @Test
    fun step1_dimExposedThroughBinder() {
        assertTrue("dim must be positive, got $dim", dim > 0)
    }

    @Test
    fun step2_embedTextReturnsPackedVector() {
        val latch = CountDownLatch(1)
        var q8: ByteArray? = null
        var cbDim = -1
        val callback = object : IEmbedCallback.Stub() {
            override fun onEmbedding(packed: ByteArray?, vectorDim: Int) {
                q8 = packed
                cbDim = vectorDim
                latch.countDown()
            }
            override fun onError(message: String?) {
                latch.countDown()
            }
        }
        service!!.embedText("Binder embedText round trip.", callback)
        assertTrue("embedText timed out", latch.await(60, TimeUnit.SECONDS))
        assertEquals(dim, cbDim)
        assertEquals(4 + dim, q8!!.size)
    }

    @Test
    fun step3_uploadQ8ChunksAndGenerateWithFollowup() {
        // Token-level rows from a temporary engine (freed before generating:
        // the service already holds the model).
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelPath = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            .getString("active_model", null)
        assertNotNull(modelPath)
        assertTrue(LlmNative.ensureLoaded())
        val tmpHandle = LlmNative.loadModel(modelPath!!, 512, 2)
        assertTrue("temporary engine failed to load", tmpHandle != 0L)
        val tokens = LlmNative.tokenize(tmpHandle, "The device is a Samsung tablet running Android.", true, true)
        val rows = LlmNative.embedTokenRows(tmpHandle, tokens)
        LlmNative.freeModel(tmpHandle)
        assertEquals(tokens.size * dim, rows.size)

        val text = generateWithEmbd(
            count = tokens.size,
            vectors = rows,
            followup = PromptManager.formatPrompt("What kind of device is this? Answer in one word."),
            nPredict = 24
        )
        assertTrue(
            "generation produced no text (error='$text')",
            text.isNotEmpty() && !text.startsWith("Error")
        )
    }

    @Test
    fun step4_dimMismatchIsRejected() {
        val handle = service!!.beginEmbeddingInput(4, dim + 1)
        assertEquals(-1, handle)
    }

    @Test
    fun step5_unknownHandleCompletesWithError() {
        val latch = CountDownLatch(1)
        var fullText = ""
        val callback = object : ILLMCallback.Stub() {
            override fun onTokenReceived(token: String) {}
            override fun onGenerationComplete(text: String) {
                fullText = text
                latch.countDown()
            }
        }
        service!!.generateFromEmbeddings(999999, 1, "hello", 4, 0f, callback)
        assertTrue("call must complete", latch.await(60, TimeUnit.SECONDS))
        assertTrue(fullText.contains("unknown embedding input"))
    }
}
