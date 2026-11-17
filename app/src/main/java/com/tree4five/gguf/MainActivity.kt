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
import android.view.LayoutInflater
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.view.inputmethod.InputMethodManager
import android.widget.ArrayAdapter
import android.widget.ImageView
import android.widget.LinearLayout
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
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.tree4five.gguf.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private data class ModelPreset(val title: String, val subtitle: String, val url: String)

    private var llmService: ILLMService? = null
    private var isBound = false
    private lateinit var binding: ActivityMainBinding
    private val uiScope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var telemetryJob: Job? = null
    private var lastHandledDownloadEventId = 0L
    private var wasGenerating = false
    private var editingIndex = -1

    private val modelFileNames = mutableListOf<String>()
    private var selectedModelName: String? = null
    private lateinit var chatAdapter: ChatAdapter
    private var currentTab = 0

    private val modelPresets = listOf(
        ModelPreset(
            "Qwen3 0.6B (Q4_K_M · Recommandé)",
            "378 MB · Rapide & raisonnement",
            "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
        ),
        ModelPreset(
            "SmolLM2 360M Instruct",
            "Q4_K_M · 258 MB · Débit max (tok/s)",
            "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
        ),
        ModelPreset(
            "Qwen2.5 0.5B Instruct",
            "Q4_K_M · 400 MB · Ultra fluide",
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ),
        ModelPreset(
            "Llama 3.2 1B Instruct",
            "Q4_K_M · 770 MB · Polyvalent",
            "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        ),
        ModelPreset(
            "Qwen2.5 1.5B Instruct",
            "Q4_K_M · 990 MB · Équilibré",
            "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        ),
        ModelPreset(
            "Qwen3 1.7B (Q4_K_M)",
            "1056 MB · Haute qualité",
            "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf"
        ),
        ModelPreset(
            "Nomic Embed v1.5",
            "Q8_0 · 146 MB · Embeddings",
            "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
        )
    )

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            isBound = true
            binding.btnStartBind.text = "Actif"
            startTelemetryLoop()
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
        setupChatRecycler()
        restorePreferences()
        setupDropdowns()
        renderPresetCatalogue()
        updateModelList()
        updateStorageInfo()
        setupNavigation()
        setupPromptChips()
        setupActions()
        observeBackgroundState()

        val initialTab = if (modelFileNames.isEmpty() && BackgroundTaskCoordinator.chatState.value.messages.isEmpty()) 1 else 0
        selectTab(initialTab, animate = false)
        renderIdleState("IDLE")
    }

    override fun onStart() {
        super.onStart()
        if (LLMInferenceService.isRunning && !isBound) {
            bindService(Intent(this, LLMInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        }
    }

    private fun configureSystemBars() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val controller = WindowInsetsControllerCompat(window, window.decorView)
        controller.isAppearanceLightStatusBars = false
        controller.isAppearanceLightNavigationBars = false
    }

    private fun applyEdgeToEdgeInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.rootContainer) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime())
            view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }
    }

    private fun setupChatRecycler() {
        chatAdapter = ChatAdapter(
            onCopyClick = { text ->
                copyToClipboard("LLMProvider", text, "Réponse copiée")
            },
            onUserClick = { pos, msg ->
                showUserMessageActions(pos, msg)
            },
            onRegenerateClick = { assistantPos ->
                regenerateFromAssistant(assistantPos)
            }
        )
        binding.rvChat.layoutManager = LinearLayoutManager(this).apply {
            stackFromEnd = true
        }
        binding.rvChat.itemAnimator = null
        binding.rvChat.adapter = chatAdapter
        refreshChatVisibility(BackgroundTaskCoordinator.chatState.value.messages.isEmpty())
    }

    private fun showUserMessageActions(position: Int, msg: ChatMessage) {
        val sheet = BottomSheetDialog(this)
        val view = layoutInflater.inflate(R.layout.sheet_message_actions, null)
        view.findViewById<TextView>(R.id.tvSheetPreview).text = msg.content

        view.findViewById<View>(R.id.actionEdit).setOnClickListener {
            sheet.dismiss()
            if (BackgroundTaskCoordinator.chatState.value.isGenerating) {
                stopActiveGeneration()
            }
            editingIndex = position
            binding.layoutEditingBanner.visibility = View.VISIBLE
            binding.etPrompt.setText(msg.content)
            binding.etPrompt.setSelection(msg.content.length)
            binding.etPrompt.requestFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.showSoftInput(binding.etPrompt, InputMethodManager.SHOW_IMPLICIT)
        }

        view.findViewById<View>(R.id.actionCopy).setOnClickListener {
            sheet.dismiss()
            copyToClipboard("LLMProvider", msg.content, "Message copié")
        }

        view.findViewById<View>(R.id.actionResend).setOnClickListener {
            sheet.dismiss()
            if (BackgroundTaskCoordinator.chatState.value.isGenerating) {
                stopActiveGeneration()
            }
            BackgroundTaskCoordinator.truncateFrom(position)
            sendUserPrompt(msg.content)
        }

        sheet.setContentView(view)
        sheet.show()
    }

    private fun regenerateFromAssistant(assistantPos: Int) {
        if (BackgroundTaskCoordinator.chatState.value.isGenerating) {
            stopActiveGeneration()
        }
        val messages = BackgroundTaskCoordinator.chatState.value.messages
        var userIdx = assistantPos - 1
        while (userIdx >= 0 && messages[userIdx].role != "user") {
            userIdx--
        }
        if (userIdx >= 0) {
            val prompt = messages[userIdx].content
            BackgroundTaskCoordinator.truncateFrom(userIdx)
            sendUserPrompt(prompt)
        }
    }

    private fun clearEditingMode() {
        editingIndex = -1
        binding.layoutEditingBanner.visibility = View.GONE
    }

    private fun refreshChatVisibility(empty: Boolean) {
        binding.layoutEmptyState.visibility = if (empty) View.VISIBLE else View.GONE
        binding.rvChat.visibility = if (empty) View.GONE else View.VISIBLE
    }

    private fun observeBackgroundState() {
        lastHandledDownloadEventId = BackgroundTaskCoordinator.downloadState.value.eventId
        uiScope.launch {
            BackgroundTaskCoordinator.downloadState.collect { st ->
                if (st.active) {
                    binding.progressBar.visibility = View.VISIBLE
                    binding.layoutDownloadMeta.visibility = View.VISIBLE
                    binding.progressBar.progress = st.percent
                    binding.tvProgress.text = st.statusText.ifBlank { "${st.percent}%" }
                    setDownloadControlsEnabled(false)
                } else {
                    binding.progressBar.visibility = View.GONE
                    binding.layoutDownloadMeta.visibility = View.GONE
                    setDownloadControlsEnabled(true)
                    if (st.eventId > lastHandledDownloadEventId && st.completedOk != null) {
                        lastHandledDownloadEventId = st.eventId
                        if (st.statusText.isNotBlank()) {
                            Toast.makeText(this@MainActivity, st.statusText, Toast.LENGTH_SHORT).show()
                        }
                        if (st.completedOk) {
                            selectedModelName = st.fileName
                            savePreferencesFromUi(st.fileName)
                            updateModelList()
                        }
                        updateStorageInfo()
                    }
                }
            }
        }

        uiScope.launch {
            BackgroundTaskCoordinator.chatState.collect { snap ->
                chatAdapter.submitMessages(snap.messages)
                refreshChatVisibility(snap.messages.isEmpty())
                if (snap.messages.isNotEmpty()) {
                    binding.rvChat.post {
                        val lm = binding.rvChat.layoutManager as? LinearLayoutManager
                        lm?.scrollToPositionWithOffset(snap.messages.size - 1, -100000)
                    }
                }
                setGeneratingUi(snap.isGenerating)
                binding.tvComposerStats.text = snap.composerStats
                if (wasGenerating && !snap.isGenerating) {
                    refreshTelemetryOnce()
                }
                wasGenerating = snap.isGenerating
            }
        }
    }

    private fun setupDropdowns() {
        binding.spinnerProtocol.setDropDownBackgroundResource(R.drawable.bg_dropdown_popup)
    }

    private fun renderPresetCatalogue() {
        val inflater = LayoutInflater.from(this)
        binding.llPresetModels.removeAllViews()
        val marginPx = (8 * resources.displayMetrics.density).toInt()

        modelPresets.forEachIndexed { idx, preset ->
            val row = inflater.inflate(R.layout.item_model_row, binding.llPresetModels, false)
            row.findViewById<TextView>(R.id.tvRowTitle).text = preset.title
            row.findViewById<TextView>(R.id.tvRowSubtitle).text = preset.subtitle
            row.findViewById<TextView>(R.id.tvRowBadge).visibility = View.GONE
            val action = row.findViewById<ImageView>(R.id.ivRowAction)
            action.setImageResource(R.drawable.ic_download)

            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            if (idx > 0) lp.topMargin = marginPx
            row.layoutParams = lp

            row.setOnClickListener {
                binding.etModelUrl.setText(preset.url)
                startDownloadOrImport(preset.url)
            }
            binding.llPresetModels.addView(row)
        }
    }

    private fun setupNavigation() {
        binding.navChat.setOnClickListener { selectTab(0, animate = true) }
        binding.navModels.setOnClickListener { selectTab(1, animate = true) }
        binding.navEngine.setOnClickListener { selectTab(2, animate = true) }
        binding.layoutModelPill.setOnClickListener { selectTab(1, animate = true) }
    }

    private fun selectTab(index: Int, animate: Boolean) {
        currentTab = index
        val panels = listOf(binding.panelChat, binding.panelModels, binding.panelEngine)
        val navIcons = listOf(binding.ivNavChat, binding.ivNavModels, binding.ivNavEngine)
        val navLabels = listOf(binding.tvNavChat, binding.tvNavModels, binding.tvNavEngine)

        val activeColor = ContextCompat.getColor(this, R.color.colorPrimary)
        val mutedColor = ContextCompat.getColor(this, R.color.colorMuted)

        navIcons.forEachIndexed { i, iv ->
            val tint = if (i == index) activeColor else mutedColor
            iv.setColorFilter(tint)
            navLabels[i].setTextColor(tint)
            navLabels[i].setTypeface(null, if (i == index) android.graphics.Typeface.BOLD else android.graphics.Typeface.NORMAL)
        }

        panels.forEachIndexed { i, panel ->
            if (i == index) {
                panel.animate().cancel()
                panel.visibility = View.VISIBLE
                if (animate) {
                    panel.alpha = 0f
                    panel.translationY = 14f
                    panel.animate()
                        .alpha(1f)
                        .translationY(0f)
                        .setDuration(160L)
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
            ArrayAdapter(this, R.layout.item_dropdown, protocols)
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

    private fun savePreferencesFromUi(activeModelFile: String? = selectedModelName) {
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val edit = prefs.edit()
        if (activeModelFile != null) {
            edit.putString("active_model", File(filesDir, activeModelFile).absolutePath)
        } else {
            edit.remove("active_model")
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
        val suggestions = mapOf(
            binding.chipPromptSummarize to
                "Résume en 3 points clairs les avantages d'un moteur LLM local 100% hors-ligne sur smartphone.",
            binding.chipPromptCode to
                "Write a concise Python function fibonacci(n) with memoization and type hints.",
            binding.chipPromptJson to
                "Reply ONLY in valid JSON with keys \"status\" (\"ok\"), \"engine\" (\"RustingStrata\"), and \"items\" (array of 3 fruits).",
            binding.chipPromptReason to
                "J'ai 3 pommes, j'en achète 5 et j'en mange 2. Combien en reste-t-il ? Explique brièvement."
        )
        suggestions.forEach { (view, prompt) ->
            view.setOnClickListener {
                binding.etPrompt.setText(prompt)
                binding.etPrompt.setSelection(prompt.length)
            }
        }
    }

    private fun setupActions() {
        binding.btnNewChat.setOnClickListener {
            if (BackgroundTaskCoordinator.chatState.value.isGenerating) {
                stopActiveGeneration()
            }
            clearEditingMode()
            BackgroundTaskCoordinator.clearChat()
            selectTab(0, animate = true)
        }

        binding.btnCancelEdit.setOnClickListener {
            clearEditingMode()
            binding.etPrompt.setText("")
        }

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
            BackgroundTaskCoordinator.downloadCancelled.set(true)
            BackgroundTaskCoordinator.updateDownloadState { it.copy(statusText = "Annulation...") }
            startService(Intent(this, LLMInferenceService::class.java).apply {
                action = LLMInferenceService.ACTION_CANCEL_DOWNLOAD
            })
        }

        binding.btnImportModel.setOnClickListener {
            selectFileLauncher.launch(arrayOf("*/*"))
        }

        binding.btnSend.setOnClickListener {
            val prompt = binding.etPrompt.text.toString().trim()
            if (prompt.isEmpty()) {
                Toast.makeText(this, "Écris un message d'abord", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (editingIndex >= 0) {
                BackgroundTaskCoordinator.truncateFrom(editingIndex)
                clearEditingMode()
            }
            binding.etPrompt.setText("")
            sendUserPrompt(prompt)
        }

        binding.btnStopGen.setOnClickListener {
            stopActiveGeneration()
        }

        binding.chipQuickBench.setOnClickListener {
            runFullBenchmark()
        }

        binding.chipQuickEmbed.setOnClickListener {
            runEmbeddingSmokeTest()
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
            val modelName = selectedModelName ?: "local-gguf"
            val curlCmd = """
                curl -s http://127.0.0.1:11435/v1/chat/completions \
                  -H "Content-Type: application/json" \
                  -d '{"model":"$modelName","messages":[{"role":"user","content":"Hello"}],"max_tokens":128,"temperature":0.0}'
            """.trimIndent()
            copyToClipboard("LLMProvider cURL", curlCmd, "Requête cURL copiée")
        }
    }

    private fun stopActiveGeneration() {
        BackgroundTaskCoordinator.cancelActiveTurn()
        try {
            llmService?.stopGeneration()
        } catch (_: Exception) {
        }
        if (LLMInferenceService.isRunning) {
            startService(Intent(this, LLMInferenceService::class.java).apply {
                action = LLMInferenceService.ACTION_STOP_GENERATION
            })
        }
    }

    private fun setGeneratingUi(active: Boolean) {
        binding.btnSend.visibility = if (active) View.GONE else View.VISIBLE
        binding.btnStopGen.visibility = if (active) View.VISIBLE else View.GONE
    }

    private fun setDownloadControlsEnabled(enabled: Boolean) {
        binding.btnDownload.isEnabled = enabled
        binding.btnImportModel.isEnabled = enabled
        for (i in 0 until binding.llPresetModels.childCount) {
            val child = binding.llPresetModels.getChildAt(i)
            child.isEnabled = enabled
            child.isClickable = enabled
            child.alpha = if (enabled) 1.0f else 0.45f
        }
    }

    private fun startForegroundCompat(intent: Intent) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
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

        if (modelFile.exists() && !url.startsWith("content://")) {
            Toast.makeText(this, "Modèle déjà installé et activé", Toast.LENGTH_SHORT).show()
            selectedModelName = fileName
            savePreferencesFromUi(fileName)
            updateModelList()
            return
        }

        if (BackgroundTaskCoordinator.isDownloading.get()) {
            Toast.makeText(this, "Téléchargement déjà en cours — annule d'abord", Toast.LENGTH_SHORT).show()
            return
        }

        val intent = Intent(this, LLMInferenceService::class.java).apply {
            action = LLMInferenceService.ACTION_START_DOWNLOAD
            putExtra(LLMInferenceService.EXTRA_URL, url)
            putExtra(LLMInferenceService.EXTRA_FILENAME, fileName)
        }
        startForegroundCompat(intent)
        if (!isBound) {
            bindService(Intent(this, LLMInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        }
    }

    private fun confirmDeleteModel(fileName: String) {
        val file = File(filesDir, fileName)
        val sizeMb = file.length() / (1024.0 * 1024.0)
        AlertDialog.Builder(this)
            .setTitle("Supprimer le modèle ?")
            .setMessage(String.format(Locale.US, "Supprimer %s (%.1f MB) ?", fileName, sizeMb))
            .setPositiveButton("Supprimer") { _, _ ->
                val wasSelected = selectedModelName == fileName
                if (file.delete()) {
                    if (wasSelected) selectedModelName = null
                    Toast.makeText(this, "Supprimé : $fileName", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    savePreferencesFromUi(selectedModelName)
                    updateStorageInfo()
                    if (wasSelected && LLMInferenceService.isRunning) {
                        if (selectedModelName != null) {
                            startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
                        } else {
                            stopInferenceService()
                        }
                    }
                }
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun applyEngineConfiguration() {
        savePreferencesFromUi(selectedModelName)
        updateHttpBadge(binding.switchHttpServer.isChecked)

        val svc = llmService
        if (!isBound || svc == null) {
            if (!selectedModelName.isNullOrEmpty()) {
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
            Toast.makeText(this, "Configuration appliquée", Toast.LENGTH_SHORT).show()
            refreshTelemetryOnce()
        } catch (e: Exception) {
            Log.e(TAG, "applyEngineConfiguration failed", e)
            Toast.makeText(this, "Erreur : ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startOrReloadService(action: String) {
        val model = selectedModelName
        if (model.isNullOrEmpty()) {
            Toast.makeText(this, "Télécharge ou sélectionne un modèle GGUF", Toast.LENGTH_SHORT).show()
            selectTab(1, animate = true)
            return
        }
        savePreferencesFromUi(model)
        val intent = Intent(this, LLMInferenceService::class.java).apply {
            this.action = action
        }
        startForegroundCompat(intent)
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
                updateTopModelPill(modelFile)
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
        val dotRes: Int
        when (state) {
            "ready" -> {
                text = "READY · $simd"
                bgRes = R.drawable.bg_badge_ready
                colorRes = R.color.colorPrimary
                dotRes = R.drawable.dot_status_ready
                binding.btnStartBind.text = "Prêt"
                binding.tvEmptySubtitle.text = "Moteur RustingStrata ($simd) · Prêt hors-ligne"
            }
            "generating" -> {
                text = "GENERATING"
                bgRes = R.drawable.bg_badge_gen
                colorRes = R.color.colorSecondary
                dotRes = R.drawable.dot_status_gen
                binding.btnStartBind.text = "Actif"
            }
            "loading" -> {
                text = "LOADING"
                bgRes = R.drawable.bg_badge_idle
                colorRes = R.color.colorWarning
                dotRes = R.drawable.dot_status_gen
                binding.btnStartBind.text = "Charge..."
                binding.tvEmptySubtitle.text = "Chargement du modèle GGUF..."
            }
            "error", "missing" -> {
                text = state.uppercase(Locale.US)
                bgRes = R.drawable.bg_badge_idle
                colorRes = R.color.colorDanger
                dotRes = R.drawable.dot_status_idle
                binding.btnStartBind.text = "Erreur"
            }
            else -> {
                text = state.uppercase(Locale.US)
                bgRes = R.drawable.bg_badge_idle
                colorRes = R.color.colorMuted
                dotRes = R.drawable.dot_status_idle
                binding.btnStartBind.text = "Démarrer"
                binding.tvEmptySubtitle.text = "Moteur RustingStrata · 100% hors-ligne"
            }
        }
        binding.tvStatusBadge.text = text
        binding.tvStatusBadge.setBackgroundResource(bgRes)
        binding.tvStatusBadge.setTextColor(ContextCompat.getColor(this, colorRes))
        binding.viewStatusDot.setBackgroundResource(dotRes)
    }

    private fun renderIdleState(label: String) {
        applyStatusBadge(label.lowercase(Locale.US), "Rust")
        val rss = readProcessRssMiB()
        binding.tvMetricMemory.text = if (rss > 0) "$rss MiB" else "-- MiB"
        updateTopModelPill(selectedModelName)
    }

    private fun updateHttpBadge(enabled: Boolean) {
        binding.tvHttpStatusBadge.text = if (enabled) "HTTP :11435 ON" else "HTTP OFF"
        binding.tvHttpStatusBadge.setBackgroundResource(
            if (enabled) R.drawable.bg_badge_ready else R.drawable.bg_badge_idle
        )
        binding.tvHttpStatusBadge.setTextColor(
            ContextCompat.getColor(this, if (enabled) R.color.colorPrimary else R.color.colorMuted)
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

    private fun updateTopModelPill(fileName: String?) {
        binding.tvTopModelName.text = if (fileName.isNullOrBlank()) {
            "Choisir un modèle"
        } else {
            fileName.removeSuffix(".gguf")
        }
    }

    private fun updateModelList() {
        val files = filesDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) }
            ?.sortedBy { it.name } ?: emptyList()
        modelFileNames.clear()
        files.forEach { modelFileNames.add(it.name) }

        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val activeName = prefs.getString("active_model", null)?.substringAfterLast("/")
        selectedModelName = when {
            activeName != null && modelFileNames.contains(activeName) -> activeName
            modelFileNames.isNotEmpty() -> modelFileNames[0]
            else -> null
        }
        updateTopModelPill(selectedModelName)

        binding.tvNoModelsHint.visibility = if (files.isEmpty()) View.VISIBLE else View.GONE
        binding.llInstalledModels.removeAllViews()
        val inflater = LayoutInflater.from(this)
        val marginPx = (8 * resources.displayMetrics.density).toInt()

        files.forEachIndexed { idx, file ->
            val row = inflater.inflate(R.layout.item_model_row, binding.llInstalledModels, false)
            val isActive = file.name == selectedModelName
            row.findViewById<TextView>(R.id.tvRowTitle).text = file.name
            row.findViewById<TextView>(R.id.tvRowSubtitle).text =
                String.format(Locale.US, "%.1f MB · GGUF", file.length() / (1024.0 * 1024.0))
            row.findViewById<TextView>(R.id.tvRowBadge).visibility =
                if (isActive) View.VISIBLE else View.GONE

            val deleteBtn = row.findViewById<ImageView>(R.id.ivRowAction)
            deleteBtn.setImageResource(R.drawable.ic_delete)
            deleteBtn.setOnClickListener { confirmDeleteModel(file.name) }

            val lp = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            if (idx > 0) lp.topMargin = marginPx
            row.layoutParams = lp

            row.setOnClickListener {
                selectedModelName = file.name
                savePreferencesFromUi(file.name)
                updateModelList()
                startOrReloadService(LLMInferenceService.ACTION_RELOAD_MODEL)
            }
            binding.llInstalledModels.addView(row)
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
                "%d GGUF (%.0f MB) · %.1f GB libre",
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

    private fun ensureServiceBootedForChat(): Boolean {
        if (LLMInferenceService.isRunning || isBound) return true
        val model = selectedModelName
        if (model.isNullOrEmpty()) {
            Toast.makeText(this, "Télécharge ou sélectionne un modèle GGUF", Toast.LENGTH_SHORT).show()
            selectTab(1, animate = true)
            return false
        }
        savePreferencesFromUi(model)
        startForegroundCompat(Intent(this, LLMInferenceService::class.java).apply {
            action = LLMInferenceService.ACTION_RELOAD_MODEL
        })
        bindService(Intent(this, LLMInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
        return true
    }

    private fun runEmbeddingSmokeTest() {
        if (!ensureServiceBootedForChat()) return
        val turnId = BackgroundTaskCoordinator.startChatTurn("Test Embedding Q8", "Mesure en cours...")
        startForegroundCompat(Intent(this, LLMInferenceService::class.java).apply {
            action = LLMInferenceService.ACTION_CHAT_EMBED
            putExtra(LLMInferenceService.EXTRA_TURN_ID, turnId)
        })
    }

    private fun runFullBenchmark() {
        val benchPrompt = "List the numbers 1 to 40 separated by commas, then summarize local AI in one sentence."
        sendUserPrompt(benchPrompt, forceGreedy = true, maxTokensOverride = 160)
    }

    private fun buildInferencePayload(latestPrompt: String): String {
        val protocol = binding.spinnerProtocol.text.toString().ifBlank { "ChatML" }
        if (protocol.equals("Raw", ignoreCase = true) || latestPrompt.contains("<|im_start|>")) {
            return latestPrompt
        }
        val arr = JSONArray()
        val recent = BackgroundTaskCoordinator.recentMessagesSnapshot(10)
        for (m in recent) {
            if (m.content.isNotBlank() && (m.role == "user" || m.role == "assistant")) {
                arr.put(JSONObject().put("role", m.role).put("content", m.content))
            }
        }
        return JSONObject().put("messages", arr).toString()
    }

    private fun sendUserPrompt(
        prompt: String,
        forceGreedy: Boolean = false,
        maxTokensOverride: Int? = null
    ) {
        if (!ensureServiceBootedForChat()) return
        savePreferencesFromUi(selectedModelName)

        val turnId = BackgroundTaskCoordinator.startChatTurn(prompt, "Réflexion...")
        val payload = buildInferencePayload(prompt)

        val nPredict = maxTokensOverride
            ?: binding.etMaxTokens.text?.toString()?.toIntOrNull()?.coerceIn(1, 4096)
            ?: LLMInferenceService.DEFAULT_N_PREDICT
        val temp = if (forceGreedy) 0.0f else (binding.etTemp.text?.toString()?.toFloatOrNull()?.coerceIn(0f, 2f)
            ?: LLMInferenceService.DEFAULT_TEMPERATURE)
        val spec = binding.switchSpecDecoding.isChecked

        val intent = Intent(this, LLMInferenceService::class.java).apply {
            action = LLMInferenceService.ACTION_CHAT_GENERATE
            putExtra(LLMInferenceService.EXTRA_PROMPT, payload)
            putExtra(LLMInferenceService.EXTRA_N_PREDICT, nPredict)
            putExtra(LLMInferenceService.EXTRA_TEMPERATURE, temp)
            putExtra(LLMInferenceService.EXTRA_SPEC_DECODING, spec)
            putExtra(LLMInferenceService.EXTRA_TURN_ID, turnId)
        }
        startForegroundCompat(intent)
        if (!isBound) {
            bindService(Intent(this, LLMInferenceService::class.java), connection, Context.BIND_AUTO_CREATE)
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
        if (isFinishing && !keepAlive && !BackgroundTaskCoordinator.isBusy()) {
            stopService(Intent(this, LLMInferenceService::class.java))
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
