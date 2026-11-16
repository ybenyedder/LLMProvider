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
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Foreground service exposing the pure-Rust RustingStrata v2.0 engine over both
 * Android AIDL ([ILLMService]) and an optional local OpenAI/Ollama HTTP API ([LocalHttpServer]).
 */
class LLMInferenceService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.IO.limitedParallelism(1) + SupervisorJob())
    private val embedScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private val requestMutex = kotlinx.coroutines.sync.Mutex()
    private val generationCancelled = AtomicBoolean(false)

    @Volatile
    private var modelHandle: Long = 0L

    @Volatile
    private var activeModelPath: String? = null

    @Volatile
    private var embeddingDim: Int = -1

    @Volatile
    private var activeNCtx: Int = DEFAULT_N_CTX

    @Volatile
    private var activeThreads: Int = 2

    @Volatile
    private var serviceState: String = "idle"

    @Volatile
    private var lastError: String = ""

    private var httpServer: LocalHttpServer? = null

    private class EmbdSlot(val count: Int, val dim: Int) {
        val vectors = FloatArray(count * dim)
    }

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

        const val ACTION_RELOAD_MODEL = "com.tree4five.gguf.ACTION_RELOAD_MODEL"
        const val ACTION_UPDATE_CONFIG = "com.tree4five.gguf.ACTION_UPDATE_CONFIG"

        const val DEFAULT_N_CTX = 2048
        const val DEFAULT_N_PREDICT = 512
        const val DEFAULT_TEMPERATURE = 0.0f
        const val DEFAULT_TOP_K = 40
        const val DEFAULT_TOP_P = 0.95f
        const val DEFAULT_REPEAT_PENALTY = 1.10f
        const val DEFAULT_HTTP_PORT = 11435

        private const val MAX_EMBD_SLOTS = 8
    }

    override fun onCreate() {
        super.onCreate()
        startForegroundServiceNotification()
        loadOrReloadModel()
        syncHttpServer()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_RELOAD_MODEL -> {
                loadOrReloadModel()
                syncHttpServer()
            }
            ACTION_UPDATE_CONFIG -> {
                applyRuntimePrefsToEngine()
                syncHttpServer()
            }
        }
        return START_STICKY
    }

    private fun startForegroundServiceNotification() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "RustingStrata LLM Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Runs local GGUF inference via the pure-Rust RustingStrata v2.0 engine."
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Tree4Five LLM Provider · Rust v2.0")
            .setContentText("Initializing Rust engine...")
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
            .setContentTitle("Tree4Five LLM Provider · Rust v2.0")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .setOngoing(true)
            .build()
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun applyRuntimePrefsToEngine() {
        val handle = modelHandle
        if (handle == 0L) return
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val spec = prefs.getBoolean("spec_decoding", true)
        val prefix = prefs.getBoolean("prefix_cache", true)
        val rep = prefs.getFloat("repeat_penalty", DEFAULT_REPEAT_PENALTY)
        try {
            LlmNative.configureEngine(handle, spec, prefix, rep)
        } catch (e: Exception) {
            Log.w(TAG, "configureEngine failed", e)
        }
    }

    private fun syncHttpServer() {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val enabled = prefs.getBoolean("http_server_enabled", false)
        val port = prefs.getInt("http_port", DEFAULT_HTTP_PORT).coerceIn(1024, 65535)
        if (!enabled) {
            httpServer?.stop()
            httpServer = null
            return
        }
        if (httpServer != null) return
        val server = LocalHttpServer(port, object : LocalHttpServer.EngineBridge {
            override fun isModelLoaded(): Boolean = modelHandle != 0L
            override fun modelName(): String = activeModelPath?.substringAfterLast('/') ?: "local-gguf"
            override fun getMetricsJson(): String = binder.engineMetricsJson
            override fun embedVector(text: String): FloatArray? {
                val h = modelHandle
                if (h == 0L || text.isEmpty()) return null
                return try {
                    LlmNative.embedText(h, text, 0)
                } catch (_: Exception) {
                    null
                }
            }
            override fun generateStreaming(
                prompt: String,
                nPredict: Int,
                temperature: Float,
                topK: Int,
                topP: Float,
                repeatPenalty: Float,
                specDecoding: Boolean,
                onToken: (String) -> Boolean
            ): String {
                val h = modelHandle
                if (h == 0L) return ""
                val sb = StringBuilder()
                runBlocking {
                    requestMutex.withLock {
                        val prefsInner = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                        val prefix = prefsInner.getBoolean("prefix_cache", true)
                        LlmNative.configureEngine(h, specDecoding, prefix, repeatPenalty)
                        generationCancelled.set(false)
                        serviceState = "generating"
                        LlmNative.generate(
                            handle = h,
                            prompt = prompt,
                            nPredict = nPredict,
                            temperature = temperature,
                            topK = topK,
                            topP = topP,
                            callback = object : LlmNative.LlmCallback {
                                override fun onToken(token: String) {
                                    sb.append(token)
                                    if (!onToken(token)) {
                                        LlmNative.stop(h)
                                    }
                                }
                                override fun onError(message: String) {
                                    Log.e(TAG, "HTTP gen error: $message")
                                }
                            }
                        )
                        serviceState = "ready"
                    }
                }
                return sb.toString()
            }
        })
        if (server.start()) {
            httpServer = server
        }
    }

    private fun loadOrReloadModel() {
        serviceScope.launch {
            requestMutex.withLock {
                try {
                    val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                    val modelPath = prefs.getString("active_model", null)
                    val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
                    val reqCtx = prefs.getInt("n_ctx", DEFAULT_N_CTX).coerceIn(512, 32768)
                    val reqThreads = prefs.getInt("n_threads", defaultThreads).coerceIn(1, 16)

                    if (modelPath == null) {
                        serviceState = "no_model"
                        lastError = "No model selected"
                        Log.e(TAG, "No model selected.")
                        updateNotification("No model selected — choose or download a GGUF in app.")
                        return@withLock
                    }

                    val modelFile = File(modelPath)
                    if (!modelFile.exists()) {
                        serviceState = "missing"
                        lastError = "Model file not found"
                        Log.e(TAG, "Model file not found at path: ${modelFile.absolutePath}")
                        updateNotification("Model file missing — please re-download.")
                        return@withLock
                    }

                    if (modelHandle != 0L && activeModelPath == modelFile.absolutePath && activeNCtx == reqCtx && activeThreads == reqThreads) {
                        applyRuntimePrefsToEngine()
                        serviceState = "ready"
                        return@withLock
                    }

                    serviceState = "loading"
                    lastError = ""
                    updateNotification("Loading ${modelFile.name} (RustingStrata)...")

                    val prev = modelHandle
                    if (prev != 0L) {
                        modelHandle = 0L
                        embeddingDim = -1
                        LlmNative.freeModel(prev)
                    }

                    if (!LlmNative.ensureLoaded()) {
                        serviceState = "error"
                        lastError = "libggufllm.so missing"
                        updateNotification("Error: Rust engine libggufllm.so missing.")
                        return@withLock
                    }

                    val handle = LlmNative.loadModel(modelFile.absolutePath, reqCtx, reqThreads)
                    if (handle == 0L) {
                        serviceState = "error"
                        lastError = "loadModel returned 0"
                        updateNotification("Error loading ${modelFile.name}.")
                        return@withLock
                    }

                    modelHandle = handle
                    activeModelPath = modelFile.absolutePath
                    activeNCtx = reqCtx
                    activeThreads = reqThreads
                    embeddingDim = try {
                        LlmNative.getEmbeddingDim(handle)
                    } catch (_: Throwable) {
                        -1
                    }
                    applyRuntimePrefsToEngine()
                    serviceState = "ready"
                    Log.i(TAG, "Rust engine ready: ${modelFile.name} (dim=$embeddingDim, ctx=$reqCtx, threads=$reqThreads)")
                    updateNotification("Ready · ${modelFile.name}")
                } catch (e: Exception) {
                    serviceState = "error"
                    lastError = e.message ?: "Initialization error"
                    Log.e(TAG, "Error initializing model", e)
                    updateNotification("Error: ${e.message}")
                }
            }
        }
    }

    private val binder = object : ILLMService.Stub() {

        override fun generateTextStream(prompt: String, callback: ILLMCallback) {
            val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            val nPredict = prefs.getInt("n_predict", DEFAULT_N_PREDICT)
            val temperature = prefs.getFloat("temperature", DEFAULT_TEMPERATURE)
            val topK = prefs.getInt("top_k", DEFAULT_TOP_K)
            val topP = prefs.getFloat("top_p", DEFAULT_TOP_P)
            val repeatPenalty = prefs.getFloat("repeat_penalty", DEFAULT_REPEAT_PENALTY)
            val specDecoding = prefs.getBoolean("spec_decoding", true)
            generateTextStreamWithParams(prompt, nPredict, temperature, topK, topP, repeatPenalty, specDecoding, callback)
        }

        override fun generateTextStreamWithParams(
            prompt: String,
            nPredict: Int,
            temperature: Float,
            topK: Int,
            topP: Float,
            repeatPenalty: Float,
            specDecoding: Boolean,
            callback: ILLMCallback?
        ) {
            if (callback == null) return
            serviceScope.launch {
                val handle = modelHandle
                if (handle == 0L) {
                    callback.onGenerationComplete("Error: Model not loaded or currently loading. Please wait.")
                    return@launch
                }
                try {
                    val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
                    val protocol = prefs.getString("api_protocol", "ChatML") ?: "ChatML"
                    val sysPrompt = prefs.getString("system_prompt", "") ?: ""
                    val prefixCache = prefs.getBoolean("prefix_cache", true)
                    val formattedPrompt = PromptManager.formatPrompt(prompt, protocol, sysPrompt)
                    val fullText = StringBuilder()

                    requestMutex.withLock {
                        serviceState = "generating"
                        generationCancelled.set(false)
                        LlmNative.configureEngine(handle, specDecoding, prefixCache, repeatPenalty)
                        LlmNative.generate(
                            handle = handle,
                            prompt = formattedPrompt,
                            nPredict = nPredict.coerceAtLeast(1),
                            temperature = temperature,
                            topK = topK,
                            topP = topP,
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
                        serviceState = "ready"
                    }
                    val result = if (generationCancelled.get()) {
                        fullText.toString() + "\n[Generation Stopped]"
                    } else {
                        fullText.toString()
                    }
                    callback.onGenerationComplete(result)
                } catch (e: Exception) {
                    serviceState = if (modelHandle != 0L) "ready" else "error"
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

        override fun getVersion(): String = BuildConfig.VERSION_NAME

        override fun getContextLength(): Int {
            val handle = modelHandle
            if (handle == 0L) return -1
            return try {
                LlmNative.getContextLength(handle)
            } catch (_: Exception) {
                -1
            }
        }

        override fun getEmbeddingDim(): Int = this@LLMInferenceService.embeddingDim

        override fun beginEmbeddingInput(count: Int, dim: Int): Int {
            val modelDim = this@LLMInferenceService.embeddingDim
            if (modelHandle == 0L || modelDim <= 0) return -1
            if (dim != modelDim) return -1
            if (count <= 0 || count > activeNCtx - 4) return -1
            synchronized(embdSlots) { return reserveEmbdSlotLocked(count, dim) }
        }

        override fun setEmbeddingChunk(handle: Int, startToken: Int, q8: ByteArray?) {
            val bytes = q8
            if (bytes == null || bytes.isEmpty()) return
            synchronized(embdSlots) {
                val slot = embdSlots[handle] ?: return
                val perVector = EmbeddingCodec.bytesPerVector(slot.dim)
                val chunkVectors = bytes.size / perVector
                if (chunkVectors == 0) return
                if (startToken < 0 || startToken + chunkVectors > slot.count) {
                    Log.e(TAG, "setEmbeddingChunk out of range: start=$startToken chunk=$chunkVectors slot=${slot.count}")
                    return
                }
                val decoded = EmbeddingCodec.decodeFlat(bytes, slot.dim)
                System.arraycopy(decoded, 0, slot.vectors, startToken * slot.dim, decoded.size)
            }
        }

        override fun generateFromEmbeddings(
            handle: Int,
            count: Int,
            followupPrompt: String?,
            nPredict: Int,
            temperature: Float,
            callback: ILLMCallback?
        ) {
            if (callback == null) return
            serviceScope.launch {
                val nativeHandle = modelHandle
                if (nativeHandle == 0L) {
                    callback.onGenerationComplete("Error: Model not loaded or currently loading. Please wait.")
                    return@launch
                }
                val slot = synchronized(embdSlots) { embdSlots[handle] }
                if (slot == null) {
                    callback.onGenerationComplete("Error: unknown embedding input (handle=$handle).")
                    return@launch
                }
                val vectors = synchronized(embdSlots) { slot.vectors.copyOf() }
                try {
                    val fullText = StringBuilder()
                    requestMutex.withLock {
                        serviceState = "generating"
                        generationCancelled.set(false)
                        LlmNative.generateFromEmbeddings(
                            handle = nativeHandle,
                            vectors = vectors,
                            count = count.coerceIn(1, slot.count),
                            followupPrompt = followupPrompt ?: "",
                            nPredict = nPredict.coerceAtMost(DEFAULT_N_PREDICT),
                            temperature = temperature,
                            topK = DEFAULT_TOP_K,
                            topP = DEFAULT_TOP_P,
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
                        serviceState = "ready"
                    }
                    val result = if (generationCancelled.get()) {
                        fullText.toString() + "\n[Generation Stopped]"
                    } else {
                        fullText.toString()
                    }
                    callback.onGenerationComplete(result)
                } catch (e: Exception) {
                    serviceState = if (modelHandle != 0L) "ready" else "error"
                    Log.e(TAG, "Error during embedding generation", e)
                    callback.onGenerationComplete("Error during generation : ${e.message}")
                }
            }
        }

        override fun releaseEmbeddings(handle: Int) {
            synchronized(embdSlots) { embdSlots.remove(handle) }
        }

        override fun embedText(text: String?, callback: IEmbedCallback?) {
            if (callback == null) return
            embedScope.launch {
                val nativeHandle = modelHandle
                if (nativeHandle == 0L || text.isNullOrEmpty()) {
                    callback.onError("Model not loaded or empty text.")
                    return@launch
                }
                try {
                    val vector = LlmNative.embedText(nativeHandle, text, 0)
                    callback.onEmbedding(EmbeddingCodec.encodeOne(vector), this@LLMInferenceService.embeddingDim)
                } catch (e: Exception) {
                    Log.e(TAG, "embedText failed", e)
                    callback.onError("embedText failed: ${e.message}")
                }
            }
        }

        override fun getEngineMetricsJson(): String {
            val h = modelHandle
            val json = if (h != 0L) {
                try {
                    JSONObject(LlmNative.getEngineInfoJson(h))
                } catch (_: Exception) {
                    JSONObject()
                }
            } else {
                JSONObject().put("engine", "RustingStrata").put("version", BuildConfig.VERSION_NAME)
            }
            json.put("service_state", serviceState)
            json.put("model_file", activeModelPath?.substringAfterLast('/') ?: "none")
            json.put("last_error", lastError)
            json.put("http_active", httpServer != null)
            return json.toString()
        }

        override fun clearKvCache() {
            val h = modelHandle
            if (h != 0L) {
                serviceScope.launch {
                    requestMutex.withLock {
                        LlmNative.clearKvCache(h)
                    }
                }
            }
        }

        override fun configureEngine(
            specDecoding: Boolean,
            prefixCache: Boolean,
            repeatPenalty: Float,
            nCtx: Int,
            nThreads: Int
        ) {
            val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            prefs.edit()
                .putBoolean("spec_decoding", specDecoding)
                .putBoolean("prefix_cache", prefixCache)
                .putFloat("repeat_penalty", repeatPenalty)
                .putInt("n_ctx", nCtx)
                .putInt("n_threads", nThreads)
                .apply()

            val h = modelHandle
            if (h != 0L) {
                LlmNative.configureEngine(h, specDecoding, prefixCache, repeatPenalty)
            }
            if (nCtx != activeNCtx || nThreads != activeThreads) {
                loadOrReloadModel()
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        super.onDestroy()
        httpServer?.stop()
        httpServer = null
        serviceScope.cancel()
        embedScope.cancel()
        val handle = modelHandle
        modelHandle = 0L
        if (handle != 0L) {
            runBlocking { requestMutex.withLock { LlmNative.freeModel(handle) } }
        }
        Log.d(TAG, "Service destroyed, Rust engine released.")
    }
}
