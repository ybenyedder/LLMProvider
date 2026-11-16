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
import android.view.animation.DecelerateInterpolator
import android.widget.ArrayAdapter
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
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

    private data class ModelPreset(val label: String, val url: String)

    private var llmService: ILLMService? = null
    private var isBound = false
    private lateinit var binding: ActivityMainBinding
    private val uiScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var telemetryJob: Job? = null
    private val downloadCancelled = AtomicBoolean(false)

    private val modelFileNames = mutableListOf<String>()
    private var currentTab = 0

    private val modelPresets = listOf(
        ModelPreset(
            "Qwen3 0.6B (Q8_0 · 640 MB · Rapide)",
            "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf"
        ),
        ModelPreset(
            "Qwen2.5 0.5B Instruct (Q4_K_M · 400 MB)",
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        ModelPreset(
            "Qwen2.5 1.5B Instruct (Q4_K_M · 990 MB)",
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        ModelPreset(
            "Qwen3 1.7B (Q8_0 · 1.8 GB)",
            "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf"
        ),
        ModelPreset(
            "SmolLM2 360M Instruct (Q4_K_M · 258 MB)",
            "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
        ),
        ModelPreset(
            "Nomic Embed v1.5 (Q8_0 · 146 MB)",
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
        )
    )

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            isBound = true
            binding.btnStartBind.text = "Recharger"
            startTelemetryLoop()
            Toast.makeText(this@MainActivity, "Connecté au moteur RustingStrata", Toast.LENGTH_SHORT).show()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
            telemetryJob?.cancel()
            renderIdleState("OFFLINE")
        }
    }

    private val selectFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri?.let {
            val uriStr = it.toString()
            binding.etModelUrl.setText(uriStr)
            startDownloadOrImport(uriStr)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        configureSystemBars()
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        applyEdgeToEdgeInsets()

        checkPermissions()
        restorePreferences()
        setupDropdowns()
        updateModelList()
        updateStorageInfo()
        setupTabs()
        setupPromptChips()
        setupActions()

        if (modelFileNames.isEmpty()) {
            selectTab(1, animate = false)
        } else {
            selectTab(0, animate = false)
        }
        renderIdleState("IDLE")
    }

    private fun configureSystemBars() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.isAppearanceLightStatusBars = false
        controller.isAppearanceLightNavigationBars = false
    }

    private fun applyEdgeToEdgeInsets() {
        val baseTop = resources.getDimensionPixelSizeOrPx(14)
        val baseBottom = resources.getDimensionPixelSizeOrPx(28)
        val baseHoriz = resources.getDimensionPixelSizeOrPx(16)
        ViewCompat.setOnApplyWindowInsetsListener(binding.rootContainer) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                baseHoriz + bars.left,
                baseTop + bars.top,
                baseHoriz + bars.right,
                baseBottom + bars.bottom
            )
            insets
        }
    }

    private fun android.content.res.Resources.getDimensionPixelSizeOrPx(dp: Int): Int =
        (dp * displayMetrics.density).toInt()

    private fun setupDropdowns() {
        binding.spinnerModels.setDropDownBackgroundResource(R.drawable.bg_auralis_popup)
        binding.spinnerPresetModels.setDropDownBackgroundResource(R.drawable.bg_auralis_popup)
        binding.spinnerProtocol.setDropDownBackgroundResource(R.drawable.bg_auralis_popup)

        val presetLabels = modelPresets.map { it.label }
        binding.spinnerPresetModels.setAdapter(
            ArrayAdapter(this, R.layout.item_auralis_dropdown, presetLabels)
        )
        binding.spinnerPresetModels.setOnItemClickListener { _, _, position, _ ->
            if (position in modelPresets.indices) {
                binding.etModelUrl.setText(modelPresets[position].url)
            }
        }
        if (binding.etModelUrl.text.isNullOrBlank()) {
            binding.spinnerPresetModels.setText(modelPresets[0].label, false)
            binding.etModelUrl.setText(modelPresets[0].url)
        }

        binding.spinnerModels.setOnItemClickListener { _, _, _, _ ->
            selectedModelFileName()?.let { fileName ->
                savePreferencesFromUi(fileName)
                updateActiveModelSubtitle(fileName)
            }
        }
    }

    private fun setupTabs() {
        binding.tabStudio.setOnClickListener { selectTab(0, animate = true) }
        binding.tabModels.setOnClickListener { selectTab(1, animate = true) }
        binding.tabEngine.setOnClickListener { selectTab(2, animate = true) }
    }

    private fun selectTab(index: Int, animate: Boolean) {
        currentTab = index
        val tabs = listOf(binding.tabStudio, binding.tabModels, binding.tabEngine)
        val panels = listOf(binding.panelStudio, binding.panelModels, binding.panelEngine)

        val ink = ContextCompat.getColor(this, R.color.auralisInk)
        val muted = ContextCompat.getColor(this, R.color.auralisTextMuted)

        tabs.forEachIndexed { i, tab ->
            if (i == index) {
                tab.setBackgroundResource(R.drawable.bg_auralis_tab_active)
                tab.setTextColor(ink)
            } else {
                tab.background = null
                tab.setTextColor(muted)
            }
        }

        panels.forEachIndexed { i, panel ->
            if (i == index) {
                panel.animate().cancel()
                panel.visibility = View.VISIBLE
                if (animate) {
                    panel.alpha = 0f
                    panel.translationY = 18f
                    panel.animate()
                        .alpha(1f)
                        .translationY(0f)
                        .setDuration(180L)
                        .setInterpolator(DecelerateInterpolator())
                        .start()
                } else {
                    panel.alpha = 1f
                    panel.translationY = 0f
                }
            } else {
                panel.animate().cancel()
                panel.visibility = View.GONE
            }
        }
    }

    private fun restorePreferences() {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val protocols = listOf("ChatML", "OpenAI", "Ollama", "Raw")
        binding.spinnerProtocol.setAdapter(
            ArrayAdapter(this, R.layout.item_auralis_dropdown, protocols)
        )
        binding.spinnerProtocol.setText(prefs.getString("api_protocol", "ChatML"), false)

        binding.etSystemPrompt.setText(prefs.getString("system_prompt", ""))
        binding.switchSpecDecoding.isChecked = prefs.getBoolean("spec_decoding", true)
        binding.switchPrefixCache.isChecked = prefs.getBoolean("prefix_cache", true)
        binding.switchHttpServer.isChecked = prefs.getBoolean("http_server_enabled", false)
        binding.switchKeepAlive.isChecked = prefs.getBoolean("keep_alive", true)
        binding.etTemp.setText(prefs.getFloat("temperature", LLMInferenceService.DEFAULT_TEMPERATURE).toString())
        binding.etMaxTokens.setText(prefs.getInt("n_predict", LLMInferenceService.DEFAULT_N_PREDICT).toString())
        binding.etCtx.setText(prefs.getInt("n_ctx", LLMInferenceService.DEFAULT_N_CTX).toString())
        val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        binding.etThreads.setText(prefs.getInt("n_threads", defaultThreads).toString())
        updateHttpBadge(binding.switchHttpServer.isChecked)
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
        val nCtx = binding.etCtx.text?.toString()?.toIntOrNull()?.coerceIn(512, 32768)
            ?: LLMInferenceService.DEFAULT_N_CTX
        val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        val threads = binding.etThreads.text?.toString()?.toIntOrNull()?.coerceIn(1, 16)
            ?: defaultThreads

        edit.putString("api_protocol", binding.spinnerProtocol.text.toString().ifBlank { "ChatML" })
            .putString("system_prompt", binding.etSystemPrompt.text?.toString() ?: "")
            .putBoolean("spec_decoding", binding.switchSpecDecoding.isChecked)
            .putBoolean("prefix_cache", binding.switchPrefixCache.isChecked)
            .putBoolean("http_server_enabled", binding.switchHttpServer.isChecked)
            .putBoolean("keep_alive", binding.switchKeepAlive.isChecked)
            .putFloat("temperature", temp)
            .putInt("n_predict", maxTok)
            .putInt("n_ctx", nCtx)
            .putInt("n_threads", threads)
            .apply()
    }

    private fun setupPromptChips() {
        val chips = mapOf<TextView, String>(
            binding.chipPromptSummarize to
                "Résume en 3 points clairs les avantages d'un moteur LLM local 100% hors-ligne sur smartphone.",
            binding.chipPromptCode to
                "Write a concise Python function fibonacci(n) with memoization and type hints.",
            binding.chipPromptJson to
                "Reply ONLY in valid JSON with keys \"status\" (\"ok\"), \"engine\" (\"RustingStrata\"), and \"items\" (array of 3 fruits).",
            binding.chipPromptReason to
                "J'ai 3 pommes, j'en achète 5 et j'en mange 2. Combien en reste-t-il ? Explique brièvement."
        )
        chips.forEach { (view, text) ->
            view.setOnClickListener {
                binding.etPrompt.setText(text)
            }
        }
    }

    private fun setupActions() {
        binding.btnStartBind.setOnClickListener {
            startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
        }

        binding.btnStopService.setOnClickListener {
            stopInferenceService()
        }

        binding.btnDownload.setOnClickListener {
            val url = binding.etModelUrl.text.toString().trim()
            startDownloadOrImport(url)
        }

        binding.btnCancelDownload.setOnClickListener {
            downloadCancelled.set(true)
        }

        binding.btnImportModel.setOnClickListener {
            selectFileLauncher.launch(arrayOf("*/*"))
        }

        binding.btnReloadModel.setOnClickListener {
            startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
        }

        binding.btnDeleteModel.setOnClickListener {
            confirmDeleteSelectedModel()
        }

        binding.btnGenerate.setOnClickListener {
            val prompt = binding.etPrompt.text.toString()
            if (prompt.isNotEmpty()) {
                binding.tvOutput.text = ""
                generateText(prompt)
            } else {
                Toast.makeText(this, "Saisis un prompt d'abord", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnStop.setOnClickListener {
            try {
                llmService?.stopGeneration()
            } catch (e: Exception) {
                Log.e(TAG, "stopGeneration failed", e)
            }
        }

        binding.btnBenchmark.setOnClickListener {
            runFullBenchmark()
        }

        binding.btnTestEmbed.setOnClickListener {
            runEmbeddingSmokeTest()
        }

        binding.btnCopyOutput.setOnClickListener {
            copyToClipboard("LLMProvider Output", binding.tvOutput.text.toString(), "Sortie copiée")
        }

        binding.btnClearOutput.setOnClickListener {
            binding.tvOutput.text = ""
            binding.tvStreamStats.text = "TTFT -- · 0.0 tok/s · 0 chunks"
        }

        binding.btnClearKv.setOnClickListener {
            try {
                llmService?.clearKvCache()
                Toast.makeText(this, "Cache KV & préfixe vidé", Toast.LENGTH_SHORT).show()
                refreshTelemetryOnce()
            } catch (e: Exception) {
                Log.e(TAG, "clearKvCache failed", e)
            }
        }

        binding.btnApplyEngine.setOnClickListener {
            applyEngineConfiguration()
        }

        binding.btnCopyCurl.setOnClickListener {
            val modelName = selectedModelFileName() ?: "local-gguf"
            val curlCmd = """
                curl -s http://127.0.0.1:11435/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -d '{"model":"$modelName","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0.0}'
            """.trimIndent()
            copyToClipboard("LLMProvider cURL", curlCmd, "Requête cURL copiée")
        }
    }

    private fun startDownloadOrImport(url: String) {
        if (url.isEmpty()) {
            Toast.makeText(this, "Saisis une URL .gguf ou importe un fichier", Toast.LENGTH_SHORT).show()
            return
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
                Toast.makeText(this, "Modèle déjà installé et sélectionné", Toast.LENGTH_SHORT).show()
                updateModelList()
                setSelectedModelText(fileName)
                savePreferencesFromUi(fileName)
                return
            }
        }

        downloadCancelled.set(false)
        binding.progressBar.visibility = View.VISIBLE
        binding.layoutDownloadMeta.visibility = View.VISIBLE
        binding.progressBar.progress = 0
        binding.tvProgress.text = "Connexion..."
        binding.btnDownload.isEnabled = false
        binding.btnImportModel.isEnabled = false

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
                Toast.makeText(this@MainActivity, "Modèle prêt : $fileName", Toast.LENGTH_SHORT).show()
                updateModelList()
                setSelectedModelText(fileName)
                savePreferencesFromUi(fileName)
            } else {
                val msg = if (downloadCancelled.get()) {
                    "Téléchargement annulé"
                } else {
                    "Échec du transfert (vérifie l'URL ou l'en-tête GGUF)"
                }
                Toast.makeText(this@MainActivity, msg, Toast.LENGTH_SHORT).show()
            }
            updateStorageInfo()
            binding.btnDownload.isEnabled = true
            binding.btnImportModel.isEnabled = true
            binding.progressBar.visibility = View.GONE
            binding.layoutDownloadMeta.visibility = View.GONE
        }
    }

    private fun confirmDeleteSelectedModel() {
        val selected = selectedModelFileName()
        if (selected.isNullOrEmpty()) {
            Toast.makeText(this, "Aucun modèle sélectionné", Toast.LENGTH_SHORT).show()
            return
        }
        val file = File(filesDir, selected)
        val sizeMb = file.length() / (1024.0 * 1024.0)
        AlertDialog.Builder(this)
            .setTitle("Supprimer le modèle ?")
            .setMessage(String.format(Locale.US, "Supprimer %s (%.1f MB) ?", selected, sizeMb))
            .setPositiveButton("Supprimer") { _, _ ->
                if (file.delete()) {
                    Toast.makeText(this, "Supprimé : $selected", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    updateStorageInfo()
                }
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun applyEngineConfiguration() {
        savePreferencesFromUi(selectedModelFileName())
        updateHttpBadge(binding.switchHttpServer.isChecked)

        val svc = llmService
        if (!isBound || svc == null) {
            if (!selectedModelFileName().isNullOrEmpty()) {
                startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
            } else {
                Toast.makeText(this, "Paramètres enregistrés", Toast.LENGTH_SHORT).show()
            }
            return
        }

        val defaultThreads = maxOf(1, Runtime.getRuntime().availableProcessors() / 2)
        val nCtx = binding.etCtx.text?.toString()?.toIntOrNull()?.coerceIn(512, 32768)
            ?: LLMInferenceService.DEFAULT_N_CTX
        val threads = binding.etThreads.text?.toString()?.toIntOrNull()?.coerceIn(1, 16)
            ?: defaultThreads
        val spec = binding.switchSpecDecoding.isChecked
        val prefix = binding.switchPrefixCache.isChecked

        try {
            svc.configureEngine(spec, prefix, LLMInferenceService.DEFAULT_REPEAT_PENALTY, nCtx, threads)
            val cfgIntent = Intent(this, LLMInferenceService::class.java).apply {
                action = LLMInferenceService.ACTION_UPDATE_CONFIG
            }
            startService(cfgIntent)
            Toast.makeText(this, "Configuration RustingStrata appliquée", Toast.LENGTH_SHORT).show()
            refreshTelemetryOnce()
        } catch (e: Exception) {
            Log.e(TAG, "applyEngineConfiguration failed", e)
            Toast.makeText(this, "Erreur : ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startOrReloadService(action: String) {
        val selectedModel = selectedModelFileName()
        if (selectedModel.isNullOrEmpty()) {
            Toast.makeText(this, "Télécharge ou sélectionne un modèle GGUF", Toast.LENGTH_SHORT).show()
            selectTab(1, animate = true)
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

    private fun stopInferenceService() {
        try {
            if (isBound) {
                unbindService(connection)
                isBound = false
            }
            llmService = null
            telemetryJob?.cancel()
            stopService(Intent(this, LLMInferenceService::class.java))
            renderIdleState("STOPPED")
            updateHttpBadge(false)
            Toast.makeText(this, "Service arrêté", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "stopService error", e)
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
            val state = json.optString("service_state", "idle")
            val simd = json.optString("simd", "Rust")
            val arch = json.optString("arch", "-")
            val quant = json.optString("quant", "-")
            val nLayer = json.optInt("n_layer", 0)
            val ctx = json.optInt("n_ctx", 0)
            val kvUsed = json.optInt("kv_used", 0)
            val prefill = json.optDouble("prefill_tok_s", 0.0)
            val decode = json.optDouble("decode_tok_s", 0.0)
            val specAcc = json.optLong("spec_accepted", 0)
            val specDraft = json.optLong("spec_drafted", 0)
            val pHits = json.optLong("prefix_cache_hits", 0)
            val sizeMb = json.optDouble("size_mb", 0.0)
            val modelFile = json.optString("model_file", "none")
            val httpOn = json.optBoolean("http_active", false)

            applyStatusBadge(state, simd)
            if (modelFile != "none") {
                updateActiveModelSubtitle(modelFile)
            }
            updateHttpBadge(httpOn)

            binding.tvMetricThroughput.text = if (decode > 0.0) {
                String.format(Locale.US, "%.1f tok/s", decode)
            } else {
                "-- tok/s"
            }
            binding.tvMetricThroughputSub.text = if (prefill > 0.0) {
                String.format(Locale.US, "Prefill %.0f/s · %s", prefill, arch)
            } else {
                "$simd · $arch/$quant"
            }

            val rssMiB = readProcessRssMiB()
            binding.tvMetricMemory.text = if (rssMiB > 0) "$rssMiB MiB" else "-- MiB"
            binding.tvMetricMemorySub.text = if (sizeMb > 0.0) {
                String.format(Locale.US, "GGUF %.0f MB · %dL", sizeMb, nLayer)
            } else {
                "Quant: $quant"
            }

            binding.tvMetricCache.text = if (ctx > 0) "$kvUsed / $ctx" else "$kvUsed / --"
            binding.tvMetricCacheSub.text = "Prefix hits: $pHits"

            val accPct = if (specDraft > 0) (specAcc * 100.0) / specDraft else -1.0
            binding.tvMetricSpec.text = if (accPct >= 0.0) {
                String.format(Locale.US, "%.0f %%", accPct)
            } else {
                "-- %"
            }
            binding.tvMetricSpecSub.text = "Drafts: $specAcc/$specDraft"
        } catch (_: Exception) {
        }
    }

    private fun applyStatusBadge(state: String, simd: String) {
        val text: String
        val bgRes: Int
        val colorRes: Int
        when (state) {
            "ready" -> {
                text = "READY · $simd"
                bgRes = R.drawable.bg_auralis_badge_live
                colorRes = R.color.auralisLive
            }
            "generating" -> {
                text = "LIVE · GEN"
                bgRes = R.drawable.bg_auralis_badge_accent
                colorRes = R.color.auralisAccent
            }
            "loading" -> {
                text = "LOADING"
                bgRes = R.drawable.bg_auralis_badge_idle
                colorRes = R.color.auralisWarning
            }
            "error", "missing" -> {
                text = state.uppercase(Locale.US)
                bgRes = R.drawable.bg_auralis_badge_accent
                colorRes = R.color.auralisDanger
            }
            else -> {
                text = state.uppercase(Locale.US)
                bgRes = R.drawable.bg_auralis_badge_idle
                colorRes = R.color.auralisTextMuted
            }
        }
        binding.tvStatusBadge.text = text
        binding.tvStatusBadge.setBackgroundResource(bgRes)
        binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun renderIdleState(label: String) {
        binding.btnStartBind.text = "Démarrer"
        applyStatusBadge(label.lowercase(Locale.US), "Rust")
        val rss = readProcessRssMiB()
        binding.tvMetricMemory.text = if (rss > 0) "$rss MiB" else "-- MiB"
        updateActiveModelSubtitle(selectedModelFileName())
    }

    private fun updateHttpBadge(enabled: Boolean) {
        binding.tvHttpStatusBadge.text = if (enabled) "HTTP :11435 ON" else "HTTP OFF"
        binding.tvHttpStatusBadge.setBackgroundResource(
            if (enabled) R.drawable.bg_auralis_badge_live else R.drawable.bg_auralis_badge_idle
        )
        binding.tvHttpStatusBadge.setTextColor(
            ContextCompat.getColor(this, if (enabled) R.color.auralisLive else R.color.auralisTextMuted)
        )
    }

    private fun readProcessRssMiB(): Long {
        return try {
            File("/proc/self/status").useLines { lines ->
                val rssLine = lines.firstOrNull { it.startsWith("VmRSS:") } ?: return@useLines 0L
                val kb = rssLine.removePrefix("VmRSS:").trim().substringBefore(' ').toLongOrNull() ?: 0L
                kb / 1024L
            }
        } catch (_: Exception) {
            0L
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
        updateActiveModelSubtitle(fileName)
    }

    private fun updateActiveModelSubtitle(fileName: String?) {
        if (fileName.isNullOrBlank()) {
            binding.tvActiveModelSubtitle.text = "Aucun modèle sélectionné"
            return
        }
        val f = File(filesDir, fileName)
        binding.tvActiveModelSubtitle.text = if (f.exists()) {
            String.format(Locale.US, "%s · %.0f MiB", fileName, f.length() / (1024.0 * 1024.0))
        } else {
            fileName
        }
    }

    private fun updateModelList() {
        val files = filesDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) }
            ?.sortedBy { it.name } ?: emptyList()
        modelFileNames.clear()
        val display = files.map { f ->
            modelFileNames.add(f.name)
            String.format(Locale.US, "%s (%.0f MB)", f.name, f.length() / (1024.0 * 1024.0))
        }
        binding.spinnerModels.setAdapter(
            ArrayAdapter(this, R.layout.item_auralis_dropdown, display)
        )
        binding.tvInstalledModelsCount.text = "${modelFileNames.size} GGUF"

        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val activeName = prefs.getString("active_model", null)?.substringAfterLast("/")
        when {
            activeName != null && modelFileNames.contains(activeName) -> setSelectedModelText(activeName)
            modelFileNames.isNotEmpty() -> setSelectedModelText(modelFileNames[0])
            else -> {
                binding.spinnerModels.setText("", false)
                updateActiveModelSubtitle(null)
            }
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
                "%d modèle(s) (%.1f MB) · Espace libre : %.2f GB",
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
            Toast.makeText(this, "Démarre le moteur d'abord", Toast.LENGTH_SHORT).show()
            return
        }
        val t0 = System.nanoTime()
        svc.embedText("Tree4Five RustingStrata local embedding smoke test.", object : IEmbedCallback.Stub() {
            override fun onEmbedding(q8: ByteArray?, dim: Int) {
                val ms = (System.nanoTime() - t0) / 1_000_000.0
                runOnUiThread {
                    binding.tvOutput.text = String.format(
                        Locale.US,
                        "[Embed Q8 OK]\nDim: %d\nQ8 payload: %d bytes\nLatence: %.2f ms",
                        dim, q8?.size ?: 0, ms
                    )
                }
            }

            override fun onError(message: String?) {
                runOnUiThread {
                    binding.tvOutput.text = "Erreur embedding : $message"
                }
            }
        })
    }

    private fun runFullBenchmark() {
        val svc = llmService
        if (!isBound || svc == null) {
            Toast.makeText(this, "Démarre le moteur d'abord", Toast.LENGTH_SHORT).show()
            return
        }
        val benchPrompt = "List the numbers 1 to 40 separated by commas, then summarize local AI in one sentence."
        binding.etPrompt.setText(benchPrompt)
        binding.tvOutput.text = "[Benchmark RustingStrata]\nMesure embedding Q8...\n"

        val t0 = System.nanoTime()
        svc.embedText(benchPrompt, object : IEmbedCallback.Stub() {
            override fun onEmbedding(q8: ByteArray?, dim: Int) {
                val embedMs = (System.nanoTime() - t0) / 1_000_000.0
                runOnUiThread {
                    binding.tvOutput.append(
                        String.format(Locale.US, "Embed OK (dim=%d, %.2f ms)\nGénération greedy (temp=0)...\n\n", dim, embedMs)
                    )
                    generateText(benchPrompt, forceGreedy = true, maxTokensOverride = 160)
                }
            }

            override fun onError(message: String?) {
                runOnUiThread {
                    binding.tvOutput.append("Embed indisponible ($message)\nGénération greedy...\n\n")
                    generateText(benchPrompt, forceGreedy = true, maxTokensOverride = 160)
                }
            }
        })
    }

    private fun generateText(
        prompt: String,
        forceGreedy: Boolean = false,
        maxTokensOverride: Int? = null
    ) {
        val svc = llmService
        if (!isBound || svc == null) {
            Toast.makeText(this, "Appuie sur Démarrer en haut d'abord", Toast.LENGTH_SHORT).show()
            return
        }
        savePreferencesFromUi(selectedModelFileName())

        val nPredict = maxTokensOverride
            ?: binding.etMaxTokens.text?.toString()?.toIntOrNull()?.coerceIn(1, 4096)
            ?: LLMInferenceService.DEFAULT_N_PREDICT
        val temp = if (forceGreedy) 0.0f else (binding.etTemp.text?.toString()?.toFloatOrNull()?.coerceIn(0f, 2f)
            ?: LLMInferenceService.DEFAULT_TEMPERATURE)
        val spec = binding.switchSpecDecoding.isChecked
        val startNs = System.nanoTime()
        var firstTokenNs = 0L
        val chunks = AtomicInteger(0)

        binding.tvStreamStats.text = "Prefill..."

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
                        "TTFT %.0f ms · %.1f tok/s · %d chunks",
                        ttftMs, tokS, n
                    )
                }
            }

            override fun onGenerationComplete(fullText: String) {
                runOnUiThread {
                    if (binding.tvOutput.text.isEmpty() && fullText.isNotEmpty()) {
                        binding.tvOutput.text = fullText
                    }
                    binding.tvOutput.append("\n\n[Terminé]")
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

    private fun copyToClipboard(label: String, text: String, feedback: String) {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText(label, text))
        Toast.makeText(this, feedback, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        telemetryJob?.cancel()
        uiScope.cancel()
        val keepAlive = binding.switchKeepAlive.isChecked
        if (isBound) {
            unbindService(connection)
            isBound = false
        }
        if (isFinishing && !keepAlive) {
            stopService(Intent(this, LLMInferenceService::class.java))
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
