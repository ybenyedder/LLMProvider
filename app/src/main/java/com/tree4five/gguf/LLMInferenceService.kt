package com.tree4five.gguf

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.withLock
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service binding harnessDroid to llama.cpp through [LlmNative].
 *
 * llama_context is not thread-safe: every generation runs on a
 * limitedParallelism(1) dispatcher AND under a Mutex, so requests queue
 * instead of corrupting the shared context (F10).
 */
class LLMInferenceService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.IO.limitedParallelism(1) + SupervisorJob())
    private val requestMutex = kotlinx.coroutines.sync.Mutex()
    private val generationCancelled = AtomicBoolean(false)

    @Volatile
    private var modelHandle: Long = 0L

    @Volatile
    private var embeddingDim: Int = -1

    /** One reserved embedding input (soft prompt); vectors are float32. */
    private class EmbdSlot(val count: Int, val dim: Int) {
        val vectors = FloatArray(count * dim)
    }

    // LRU of reserved embedding slots; access order guarded by this lock.
    private val embdSlots = object : LinkedHashMap<Int, EmbdSlot>(8, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Int, EmbdSlot>?): Boolean =
            size > MAX_EMBD_SLOTS
    }
    private var nextEmbdSlotId = 1

    private fun reserveEmbdSlotLocked(count: Int, dim: Int): Int {
        val id = nextEmbdSlotId++
        embdSlots[id] = EmbdSlot(count, dim)
        return id
    }

    companion object {
        private const val TAG = "LLMInferenceService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "LLM_SERVICE_CHANNEL"

        /** Context window; 2048 keeps the KV cache near 25 MB on Qwen2.5-0.5B (F7). */
        private const val N_CTX = 2048
        private const val N_PREDICT = 512

        /** Maximum reserved embedding slots (LRU-evicted). */
        private const val MAX_EMBD_SLOTS = 8

        /** Binder transfer ceiling per setEmbeddingChunk call, in vectors. */
        private const val MAX_CHUNK_VECTORS = 32

        /** Greedy by default: deterministic, best-behaved on small models. */
        private const val TEMPERATURE = 0.0f
        private const val TOP_K = 40
        private const val TOP_P = 0.95f
    }

    override fun onCreate() {
        super.onCreate()

        startForegroundService()
        initializeModel()
    }

    private fun startForegroundService() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "LLM Inference Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Keeps the LLM model active in the background for cross-app requests."
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("LLM Inference Service")
            .setContentText("Loading GGUF model...")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    private fun updateNotification(text: String) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("LLM Inference Service")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun initializeModel() {
        serviceScope.launch {
            try {
                val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                val modelPath = prefs.getString("active_model", null)

                if (modelPath == null) {
                    Log.e(TAG, "No model selected.")
                    updateNotification("Error: No model selected. Please select one in the app.")
                    return@launch
                }

                val modelFile = File(modelPath)
                if (!modelFile.exists()) {
                    Log.e(TAG, "Model file not found at path : ${modelFile.absolutePath}")
                    updateNotification("Error: Model not found. Please download it.")
                    return@launch
                }

                if (!LlmNative.ensureLoaded()) {
                    updateNotification("Error: native library missing on this device.")
                    return@launch
                }

                // CPU-only inference; half the cores for responsiveness (4G/4C SoC).
                val threads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
                val handle = LlmNative.loadModel(modelFile.absolutePath, N_CTX, threads)
                if (handle == 0L) {
                    updateNotification("Error during model initialization.")
                    return@launch
                }
                modelHandle = handle
                embeddingDim = try {
                    LlmNative.getEmbeddingDim(handle)
                } catch (_: Throwable) {
                    -1
                }
                Log.d(TAG, "Model loaded (handle=$handle, embeddingDim=$embeddingDim).")
                updateNotification("GGUF model is loaded and ready.")
            } catch (e: Exception) {
                Log.e(TAG, "Error during GGUF model initialization", e)
                updateNotification("Error during model initialization.")
            }
        }
    }

    private val binder = object : ILLMService.Stub() {

        override fun generateTextStream(prompt: String, callback: ILLMCallback) {
            serviceScope.launch {
                val handle = modelHandle
                if (handle == 0L) {
                    callback.onGenerationComplete("Error: Model not loaded or currently loading. Please wait.")
                    return@launch
                }

                try {
                    val formattedPrompt = PromptManager.formatPrompt(prompt)
                    Log.d(TAG, "Formatted prompt for inference: $formattedPrompt")

                    val fullText = StringBuilder()
                    requestMutex.withLock {
                        generationCancelled.set(false)
                        LlmNative.generate(
                            handle = handle,
                            prompt = formattedPrompt,
                            nPredict = N_PREDICT,
                            temperature = TEMPERATURE,
                            topK = TOP_K,
                            topP = TOP_P,
                            callback = object : LlmNative.LlmCallback {
                                override fun onToken(token: String) {
                                    fullText.append(token)
                                    callback.onTokenReceived(token)
                                }

                                override fun onError(message: String) {
                                    Log.e(TAG, "Generation error: $message")
                                }
                            }
                        )
                    }
                    val result = if (generationCancelled.get()) {
                        fullText.toString() + "\n[Generation Stopped]"
                    } else {
                        fullText.toString()
                    }
                    callback.onGenerationComplete(result)
                } catch (e: Exception) {
                    Log.e(TAG, "Error during generation", e)
                    callback.onGenerationComplete("Error during generation : ${e.message}")
                }
            }
        }

        override fun stopGeneration() {
            generationCancelled.set(true)
            val handle = modelHandle
            if (handle != 0L) {
                try {
                    LlmNative.stop(handle)
                } catch (e: Exception) {
                    Log.e(TAG, "Error stopping generation", e)
                }
            }
        }

        override fun getVersion(): String {
            return BuildConfig.VERSION_NAME
        }
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        val handle = modelHandle
        if (handle != 0L) {
            // Scope already cancelled: free the engine synchronously.
            runBlocking { requestMutex.withLock { LlmNative.freeModel(handle) } }
            modelHandle = 0L
        }
        Log.d(TAG, "Service destroyed, model released.")
    }
}
