package com.tree4five.gguf

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.StatFs
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.chip.Chip
import com.tree4five.gguf.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger

class MainActivity : AppCompatActivity() {

    private var llmService: ILLMService? = null
    private var isBound = false
    private lateinit var binding: ActivityMainBinding
    private val uiScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var telemetryJob: Job? = null
    private val downloadCancelled = AtomicBoolean(false)

    private val modelFileNames = mutableListOf<String>()

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            isBound = true
            startTelemetryLoop()
            Toast.makeText(this@MainActivity, "Connected to RustingStrata v2.0 AIDL", Toast.LENGTH_SHORT).show()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
            telemetryJob?.cancel()
            binding.tvEngineBadge.text = "Disconnected"
        }
    }

    private val selectFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri?.let {
            binding.etModelUrl.setText(it.toString())
            Toast.makeText(this, "File selected — tap Fetch URL to import.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        checkPermissions()
        restorePreferences()
        updateModelList()
        updateStorageInfo()
        setupPresetsAndChips()
        setupActions()
    }

    private fun restorePreferences() {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val protocols = listOf("ChatML", "OpenAI", "Ollama", "Raw")
        val protocolAdapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, protocols)
        binding.spinnerProtocol.setAdapter(protocolAdapter)
        binding.spinnerProtocol.setText(prefs.getString("api_protocol", "ChatML"), false)

        binding.etSystemPrompt.setText(prefs.getString("system_prompt", ""))
        binding.switchSpecDecoding.isChecked = prefs.getBoolean("spec_decoding", true)
        binding.switchPrefixCache.isChecked = prefs.getBoolean("prefix_cache", true)
        binding.switchHttpServer.isChecked = prefs.getBoolean("http_server_enabled", false)
        binding.etTemp.setText(prefs.getFloat("temperature", LLMInferenceService.DEFAULT_TEMPERATURE).toString())
        binding.etMaxTokens.setText(prefs.getInt("n_predict", LLMInferenceService.DEFAULT_N_PREDICT).toString())
        binding.etContext.setText(prefs.getInt("n_ctx", LLMInferenceService.DEFAULT_N_CTX).toString())
        val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        binding.etThreads.setText(prefs.getInt("n_threads", defaultThreads).toString())
    }

    private fun savePreferencesFromUi(activeModelFile: String? = null) {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val edit = prefs.edit()
        if (activeModelFile != null) {
            edit.putString("active_model", File(filesDir, activeModelFile).absolutePath)
        }
        val temp = binding.etTemp.text?.toString()?.toFloatOrNull()?.coerceIn(0f, 2f)
            ?: LLMInferenceService.DEFAULT_TEMPERATURE
        val maxTok = binding.etMaxTokens.text?.toString()?.toIntOrNull()?.coerceIn(1, 4096)
            ?: LLMInferenceService.DEFAULT_N_PREDICT
        val nCtx = binding.etContext.text?.toString()?.toIntOrNull()?.coerceIn(512, 32768)
            ?: LLMInferenceService.DEFAULT_N_CTX
        val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        val threads = binding.etThreads.text?.toString()?.toIntOrNull()?.coerceIn(1, 16)
            ?: defaultThreads

        edit.putString("api_protocol", binding.spinnerProtocol.text.toString().ifBlank { "ChatML" })
            .putString("system_prompt", binding.etSystemPrompt.text?.toString() ?: "")
            .putBoolean("spec_decoding", binding.switchSpecDecoding.isChecked)
            .putBoolean("prefix_cache", binding.switchPrefixCache.isChecked)
            .putBoolean("http_server_enabled", binding.switchHttpServer.isChecked)
            .putFloat("temperature", temp)
            .putInt("n_predict", maxTok)
            .putInt("n_ctx", nCtx)
            .putInt("n_threads", threads)
            .apply()
    }

    private fun setupPresetsAndChips() {
        val presets = mapOf(
            binding.chipPresetQwen306b.id to
                "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
            binding.chipPreset05b.id to
                "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
            binding.chipPreset15b.id to
                "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            binding.chipPresetQwen317b.id to
                "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf",
            binding.chipPresetSmol.id to
                "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf",
            binding.chipPresetNomic.id to
                "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
        )
        presets.forEach { (chipId, url) ->
            findViewById<Chip>(chipId).setOnClickListener {
                binding.etModelUrl.setText(url)
            }
        }

        binding.chipPromptPoem.setOnClickListener {
            binding.etPrompt.setText("Write a short four-line poem about a robot watching a sunset.")
        }
        binding.chipPromptMath.setOnClickListener {
            binding.etPrompt.setText("I have 3 apples, buy 5 more, and eat 2. How many are left? Explain briefly.")
        }
        binding.chipPromptPython.setOnClickListener {
            binding.etPrompt.setText("Write a concise Python function fibonacci(n) with memoization.")
        }
        binding.chipPromptJson.setOnClickListener {
            binding.etPrompt.setText("Reply ONLY in valid JSON with keys \"status\" (\"ok\") and \"items\" (array of 3 fruits).")
        }
        binding.chipPromptEmbed.setOnClickListener {
            runEmbeddingSmokeTest()
        }
        binding.chipPromptBench.setOnClickListener {
            binding.etPrompt.setText("List the numbers 1 to 40 separated by commas, then summarize what local AI means.")
            binding.tvOutput.text = ""
            generateText(binding.etPrompt.text.toString())
        }
    }

    private fun setupActions() {
        binding.btnDownload.setOnClickListener {
            val url = binding.etModelUrl.text.toString().trim()
            if (url.isEmpty()) {
                Toast.makeText(this, "Enter a GGUF URL or pick a local file", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            var fileName = url.substringAfterLast("/").substringBefore("?")
            if (url.startsWith("content://")) {
                fileName = getFileNameFromUri(Uri.parse(url)) ?: "model_${System.currentTimeMillis()}.gguf"
            }
            if (!fileName.endsWith(".gguf", ignoreCase = true)) {
                fileName = "model_${System.currentTimeMillis()}.gguf"
            }
            val modelFile = File(filesDir, fileName)

            if (modelFile.exists()) {
                if (url.startsWith("content://")) {
                    modelFile.delete()
                } else {
                    Toast.makeText(this, "Model already installed — selected.", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    setSelectedModelText(fileName)
                    return@setOnClickListener
                }
            }

            downloadCancelled.set(false)
            binding.progressBar.visibility = View.VISIBLE
            binding.layoutDownloadMeta.visibility = View.VISIBLE
            binding.progressBar.progress = 0
            binding.tvProgress.text = "Connecting..."
            binding.btnDownload.isEnabled = false
            binding.btnSelectFile.isEnabled = false

            uiScope.launch {
                val success = ModelDownloader.downloadOrCopyModelDetailed(
                    sourceUri = url,
                    destinationFile = modelFile,
                    context = this@MainActivity,
                    cancelled = downloadCancelled
                ) { st ->
                    binding.progressBar.progress = st.percent
                    val dlMb = st.downloadedBytes / (1024.0 * 1024.0)
                    val totalStr = if (st.totalBytes > 0) {
                        String.format(Locale.US, "%.1f MB", st.totalBytes / (1024.0 * 1024.0))
                    } else {
                        "?"
                    }
                    binding.tvProgress.text = String.format(
                        Locale.US,
                        "%d%% · %.1f/%s · %.1f MB/s · ETA %ds",
                        st.percent, dlMb, totalStr, st.speedMbPerSec, st.etaSeconds
                    )
                }
                if (success) {
                    Toast.makeText(this@MainActivity, "Model ready: $fileName", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    setSelectedModelText(fileName)
                } else {
                    val msg = if (downloadCancelled.get()) "Download cancelled" else "Transfer failed (check URL or GGUF header)"
                    Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
                }
                updateStorageInfo()
                binding.btnDownload.isEnabled = true
                binding.btnSelectFile.isEnabled = true
                binding.progressBar.visibility = View.GONE
                binding.layoutDownloadMeta.visibility = View.GONE
            }
        }

        binding.btnCancelDownload.setOnClickListener {
            downloadCancelled.set(true)
        }

        binding.btnSelectFile.setOnClickListener {
            selectFileLauncher.launch(arrayOf("*/*"))
        }

        binding.btnStartService.setOnClickListener {
            startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
        }

        binding.btnReloadModel.setOnClickListener {
            startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
        }

        binding.btnDeleteModel.setOnClickListener {
            val selected = selectedModelFileName()
            if (selected.isNullOrEmpty()) {
                Toast.makeText(this, "No installed model selected", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val file = File(filesDir, selected)
            val sizeMb = file.length() / (1024.0 * 1024.0)
            AlertDialog.Builder(this)
                .setTitle("Delete model?")
                .setMessage(String.format(Locale.US, "Remove %s (%.1f MB)?", selected, sizeMb))
                .setPositiveButton("Delete") { _, _ ->
                    if (file.delete()) {
                        Toast.makeText(this, "Deleted $selected", Toast.LENGTH_SHORT).show()
                        updateModelList()
                        updateStorageInfo()
                    }
                }
                .setNegativeButton("Cancel", null)
                .show()
        }

        binding.btnClearKv.setOnClickListener {
            try {
                llmService?.clearKvCache()
                Toast.makeText(this, "KV & prefix cache cleared", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "clearKvCache failed", e)
            }
        }

        binding.btnStopService.setOnClickListener {
            try {
                if (isBound) {
                    unbindService(connection)
                    isBound = false
                }
                llmService = null
                telemetryJob?.cancel()
                stopService(Intent(this, LLMInferenceService::class.java))
                binding.tvEngineBadge.text = "Stopped"
                Toast.makeText(this, "Service stopped", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "stopService error", e)
            }
        }

        binding.btnTest.setOnClickListener {
            val prompt = binding.etPrompt.text.toString()
            if (prompt.isNotEmpty()) {
                binding.tvOutput.text = ""
                generateText(prompt)
            } else {
                Toast.makeText(this, "Enter a prompt first", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnStop.setOnClickListener {
            try {
                llmService?.stopGeneration()
            } catch (e: Exception) {
                Log.e(TAG, "Error stopping generation", e)
            }
        }

        binding.btnCopyOutput.setOnClickListener {
            val text = binding.tvOutput.text.toString()
            val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cm.setPrimaryClip(ClipData.newPlainText("LLM Output", text))
            Toast.makeText(this, "Copied to clipboard", Toast.LENGTH_SHORT).show()
        }

        binding.btnClearOutput.setOnClickListener {
            binding.tvOutput.text = ""
            binding.tvStreamStats.text = "TTFT: -- | Speed: -- | Tokens: 0"
        }
    }

    private fun startOrReloadService(action: String) {
        val selectedModel = selectedModelFileName()
        if (selectedModel.isNullOrEmpty()) {
            Toast.makeText(this, "Download or select a GGUF model first", Toast.LENGTH_SHORT).show()
            return
        }
        savePreferencesFromUi(selectedModel)
        val intent = Intent(this, LLMInferenceService::class.java).apply {
            this.action = action
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        if (!isBound) {
            bindService(Intent(this, LLMInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        }
    }

    private fun startTelemetryLoop() {
        telemetryJob?.cancel()
        telemetryJob = uiScope.launch {
            while (isActive && isBound) {
                refreshTelemetryOnce()
                delay(1000)
            }
        }
    }

    private fun refreshTelemetryOnce() {
        val svc = llmService ?: return
        try {
            val json = JSONObject(svc.engineMetricsJson)
            val state = json.optString("service_state", "unknown")
            val badgeText = when (state) {
                "ready" -> "READY · ${json.optString("simd", "Rust")}"
                "generating" -> "GENERATING"
                "loading" -> "LOADING..."
                "error" -> "ERROR"
                else -> state.uppercase(Locale.US)
            }
            binding.tvEngineBadge.text = badgeText
            val badgeColor = when (state) {
                "ready" -> R.color.colorPrimary
                "generating" -> R.color.colorSecondary
                "loading" -> R.color.colorWarning
                else -> R.color.colorDanger
            }
            binding.tvEngineBadge.setTextColor(ContextCompat.getColor(this, badgeColor))

            val modelFile = json.optString("model_file", "none")
            val arch = json.optString("arch", "-")
            val quant = json.optString("quant", "-")
            val embd = json.optInt("n_embd", -1)
            val ctx = json.optInt("n_ctx", -1)
            val kvUsed = json.optInt("kv_used", 0)
            val prefill = json.optDouble("prefill_tok_s", 0.0)
            val decode = json.optDouble("decode_tok_s", 0.0)
            val specAcc = json.optLong("spec_accepted", 0)
            val specDraft = json.optLong("spec_drafted", 0)
            val pHits = json.optLong("prefix_cache_hits", 0)
            val httpOn = json.optBoolean("http_active", false)

            binding.tvTelemetrySummary.text = buildString {
                append("Model: ").append(modelFile)
                append(" (").append(arch).append(" · ").append(quant).append(")\n")
                append("Embd: ").append(embd)
                append(" · Ctx: ").append(kvUsed).append("/").append(ctx)
                append(" · HTTP: ").append(if (httpOn) ":11435 ON" else "OFF").append("\n")
                append(String.format(Locale.US, "Speed: %.1f prefill | %.1f tok/s decode\n", prefill, decode))
                append("Spec N-gram: $specAcc/$specDraft · Prefix hits: $pHits")
            }
        } catch (_: Exception) {
        }
    }

    private fun selectedModelFileName(): String? {
        val raw = binding.spinnerModels.text?.toString()?.trim().orEmpty()
        if (raw.isEmpty()) return null
        return raw.substringBefore(" (").trim()
    }

    private fun setSelectedModelText(fileName: String) {
        val f = File(filesDir, fileName)
        val label = if (f.exists()) {
            String.format(Locale.US, "%s (%.0f MB)", fileName, f.length() / (1024.0 * 1024.0))
        } else {
            fileName
        }
        binding.spinnerModels.setText(label, false)
    }

    private fun updateModelList() {
        val files = filesDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) }
            ?.sortedBy { it.name } ?: emptyList()
        modelFileNames.clear()
        val display = files.map { f ->
            modelFileNames.add(f.name)
            String.format(Locale.US, "%s (%.0f MB)", f.name, f.length() / (1024.0 * 1024.0))
        }
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, display)
        binding.spinnerModels.setAdapter(adapter)

        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val activeName = prefs.getString("active_model", null)?.substringAfterLast("/")
        if (activeName != null && modelFileNames.contains(activeName)) {
            setSelectedModelText(activeName)
        } else if (modelFileNames.isNotEmpty()) {
            setSelectedModelText(modelFileNames[0])
        } else {
            binding.spinnerModels.setText("", false)
        }
    }

    private fun updateStorageInfo() {
        try {
            val stat = StatFs(filesDir.absolutePath)
            val freeGb = stat.availableBytes / (1024.0 * 1024.0 * 1024.0)
            val usedModelsMb = (filesDir.listFiles { _, n -> n.endsWith(".gguf", true) }
                ?.sumOf { it.length() } ?: 0L) / (1024.0 * 1024.0)
            binding.tvStorageInfo.text = String.format(
                Locale.US,
                "Models: %d installed (%.1f MB) · Free space: %.2f GB",
                modelFileNames.size, usedModelsMb, freeGb
            )
        } catch (_: Exception) {
        }
    }

    private fun getFileNameFromUri(uri: Uri): String? {
        var result: String? = null
        if (uri.scheme == "content") {
            contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                if (cursor.moveToFirst()) {
                    val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                    if (index != -1) {
                        result = cursor.getString(index)
                    }
                }
            }
        }
        if (result == null) {
            result = uri.path?.substringAfterLast('/')
        }
        return result
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 101)
            }
        }
    }

    private fun runEmbeddingSmokeTest() {
        val svc = llmService
        if (!isBound || svc == null) {
            Toast.makeText(this, "Start & bind service first!", Toast.LENGTH_SHORT).show()
            return
        }
        val t0 = System.nanoTime()
        svc.embedText("Tree4Five RustingStrata local embedding smoke test.", object : IEmbedCallback.Stub() {
            override fun onEmbedding(q8: ByteArray?, dim: Int) {
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                runOnUiThread {
                    binding.tvOutput.text = String.format(
                        Locale.US,
                        "Embedding test OK!\nDim: %d\nQ8 payload: %d bytes\nRound-trip latency: %.2f ms",
                        dim, q8?.size ?: 0, ms
                    )
                }
            }

            override fun onError(message: String?) {
                runOnUiThread {
                    binding.tvOutput.text = "Embedding error: $message"
                }
            }
        })
    }

    private fun generateText(prompt: String) {
        val svc = llmService
        if (!isBound || svc == null) {
            Toast.makeText(this, "Tap 'Start / Apply' first!", Toast.LENGTH_SHORT).show()
            return
        }
        savePreferencesFromUi(selectedModelFileName())

        val nPredict = binding.etMaxTokens.text?.toString()?.toIntOrNull()?.coerceIn(1, 4096)
            ?: LLMInferenceService.DEFAULT_N_PREDICT
        val temp = binding.etTemp.text?.toString()?.toFloatOrNull()?.coerceIn(0f, 2f)
            ?: LLMInferenceService.DEFAULT_TEMPERATURE
        val spec = binding.switchSpecDecoding.isChecked
        val startNs = System.nanoTime()
        var firstTokenNs = 0L
        val chunks = AtomicInteger(0)

        binding.tvStreamStats.text = "Prefilling..."

        val callback = object : ILLMCallback.Stub() {
            override fun onTokenReceived(token: String) {
                val now = System.nanoTime()
                if (firstTokenNs == 0L) {
                    firstTokenNs = now
                }
                val n = chunks.incrementAndGet()
                val ttftMs = (firstTokenNs - startNs) / 1_000_000.0
                val genSec = ((now - firstTokenNs) / 1_000_000_000.0).coerceAtLeast(0.01)
                val tokS = if (n > 1) (n - 1) / genSec else 0.0
                runOnUiThread {
                    binding.tvOutput.append(token)
                    binding.tvStreamStats.text = String.format(
                        Locale.US,
                        "TTFT: %.0f ms | Stream: %.1f tok/s | Chunks: %d",
                        ttftMs, tokS, n
                    )
                }
            }

            override fun onGenerationComplete(fullText: String) {
                runOnUiThread {
                    if (binding.tvOutput.text.isEmpty() && fullText.isNotEmpty()) {
                        binding.tvOutput.text = fullText
                    }
                    binding.tvOutput.append("\n\n[Complete]")
                    refreshTelemetryOnce()
                }
            }
        }

        try {
            svc.generateTextStreamWithParams(
                prompt,
                nPredict,
                temp,
                LLMInferenceService.DEFAULT_TOP_K,
                LLMInferenceService.DEFAULT_TOP_P,
                LLMInferenceService.DEFAULT_REPEAT_PENALTY,
                spec,
                callback
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error calling AIDL", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        telemetryJob?.cancel()
        uiScope.cancel()
        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
