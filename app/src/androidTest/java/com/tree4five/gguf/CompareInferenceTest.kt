package com.tree4five.gguf

import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.ServiceTestRule
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

@RunWith(AndroidJUnit4::class)
class CompareInferenceTest {

    @get:Rule
    val serviceRule = ServiceTestRule()

    @Test
    fun runPromptsAndCompare() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val url = "https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
        val modelFile = File(context.filesDir, "SmolLM2-135M-Instruct-Q4_K_M.gguf")
        
        Log.i("CompareTest", "Downloading model... This may take a while.")
        runBlocking {
            if (!modelFile.exists()) {
                val success = ModelDownloader.downloadOrCopyModel(url, modelFile, context) { progress ->
                    if (progress % 10 == 0) Log.i("CompareTest", "Download progress: $progress%")
                }
                assertTrue("Model download failed", success)
            }
        }
        
        val prefs = context.getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        prefs.edit().putString("active_model", modelFile.absolutePath).commit()

        val serviceIntent = Intent(context, LLMInferenceService::class.java)
        val binder: IBinder = serviceRule.bindService(serviceIntent)
        val service = ILLMService.Stub.asInterface(binder)
        assertNotNull(service)

        // Wait for model to load
        Thread.sleep(10000)

        val prompts = listOf(
            "Write a short, four-line poem about a robot learning how to paint a sunset.",
            "I have 3 apples. I give 1 apple to Alice. Then I go to the store and buy 5 more apples. Finally, I eat 2 apples. How many apples do I have left? Explain your math step-by-step.",
            "Write a simple Python function called `calculate_factorial` that takes an integer `n` and returns its factorial."
        )

        for ((index, prompt) in prompts.withIndex()) {
            val latch = CountDownLatch(1)
            var completeText = ""

            val callback = object : ILLMCallback.Stub() {
                override fun onTokenReceived(token: String) {
                }
                override fun onGenerationComplete(fullText: String) {
                    completeText = fullText
                    latch.countDown()
                }
            }

            Log.i("CompareTest", "======================================")
            Log.i("CompareTest", "PROMPT ${index + 1}: $prompt")
            
            service.generateTextStream("{\"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}]}", callback)
            
            val success = latch.await(60, TimeUnit.SECONDS)
            assertTrue("Generation timed out for prompt ${index + 1}", success)
            
            Log.i("CompareTest", "RESPONSE ${index + 1}:\n$completeText")
            Log.i("CompareTest", "======================================")
        }
    }
}
