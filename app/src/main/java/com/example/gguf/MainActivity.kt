package com.example.gguf

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.provider.OpenableColumns
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.gguf.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private var llmService: ILLMService? = null
    private var isBound = false
    private lateinit var binding: ActivityMainBinding

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            isBound = true
            Toast.makeText(this@MainActivity, "Connected to LLM AIDL service", Toast.LENGTH_SHORT).show()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
        }
    }

    private val selectFileLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        uri?.let {
            binding.etModelUrl.setText(it.toString())
            Toast.makeText(this, "File selected! Tap Fetch to copy it.", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        checkPermissions()
        updateModelList()

        binding.btnDownload.setOnClickListener {
            val url = binding.etModelUrl.text.toString().trim()
            if (url.isEmpty()) {
                Toast.makeText(this, "Please enter a URL or select a file", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            var fileName = url.substringAfterLast("/")
            if (url.startsWith("content://")) {
                fileName = getFileNameFromUri(Uri.parse(url)) ?: "model_${System.currentTimeMillis()}.gguf"
            }
            if (!fileName.endsWith(".gguf", ignoreCase = true)) {
                fileName = "model_${System.currentTimeMillis()}.gguf"
            }
            val modelFile = File(filesDir, fileName)

            if (modelFile.exists()) {
                Toast.makeText(this, "Model already exists! Selected it in the list.", Toast.LENGTH_SHORT).show()
                updateModelList()
                binding.spinnerModels.setText(fileName, false)
                return@setOnClickListener
            }

            binding.progressBar.visibility = View.VISIBLE
            binding.tvProgress.visibility = View.VISIBLE
            binding.btnDownload.isEnabled = false
            binding.btnSelectFile.isEnabled = false

            CoroutineScope(Dispatchers.Main).launch {
                val success = ModelDownloader.downloadOrCopyModel(url, modelFile, this@MainActivity) { progress ->
                    binding.progressBar.progress = progress
                    binding.tvProgress.text = "${progress}%"
                }
                if (success) {
                    Toast.makeText(this@MainActivity, "Model setup complete", Toast.LENGTH_SHORT).show()
                    updateModelList()
                    binding.spinnerModels.setText(fileName, false)
                } else {
                    Toast.makeText(this@MainActivity, "Failed to fetch/copy model", Toast.LENGTH_SHORT).show()
                }
                binding.btnDownload.isEnabled = true
                binding.btnSelectFile.isEnabled = true
                binding.progressBar.visibility = View.GONE
                binding.tvProgress.visibility = View.GONE
            }
        }

        binding.btnSelectFile.setOnClickListener {
            selectFileLauncher.launch(arrayOf("*/*"))
        }

        binding.btnStartService.setOnClickListener {
            val selectedModel = binding.spinnerModels.text.toString()
            if (selectedModel.isEmpty()) {
                Toast.makeText(this, "Please download or select a model first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
            prefs.edit().putString("active_model", File(filesDir, selectedModel).absolutePath).apply()

            val intent = Intent(this, LLMInferenceService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }

        binding.btnTest.setOnClickListener {
            val prompt = binding.etPrompt.text.toString()
            if (prompt.isNotEmpty()) {
                binding.tvOutput.text = ""
                generateText(prompt)
            } else {
                Toast.makeText(this, "Please enter a prompt", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnStop.setOnClickListener {
            if (isBound && llmService != null) {
                try {
                    llmService?.stopGeneration()
                } catch (e: Exception) {
                    Log.e("MainActivity", "Error stopping generation", e)
                }
            }
        }
    }

    private fun updateModelList() {
        val models = filesDir.listFiles { _, name -> name.endsWith(".gguf", ignoreCase = true) }
            ?.map { it.name } ?: emptyList()
        val adapter = ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, models)
        binding.spinnerModels.setAdapter(adapter)
        
        val prefs = getSharedPreferences("app_prefs", Context.MODE_PRIVATE)
        val activeModelPath = prefs.getString("active_model", null)
        val activeModelName = activeModelPath?.substringAfterLast("/")
        
        if (activeModelName != null && models.contains(activeModelName)) {
            binding.spinnerModels.setText(activeModelName, false)
        } else if (models.isNotEmpty()) {
            binding.spinnerModels.setText(models[0], false)
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
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 101)
            }
        }
    }

    private fun generateText(prompt: String) {
        if (!isBound || llmService == null) {
            Toast.makeText(this, "Service not bound!", Toast.LENGTH_SHORT).show()
            return
        }

        val callback = object : ILLMCallback.Stub() {
            override fun onTokenReceived(token: String) {
                runOnUiThread {
                    binding.tvOutput.append(token)
                }
            }

            override fun onGenerationComplete(fullText: String) {
                runOnUiThread {
                    binding.tvOutput.append("\n\n[Complete]")
                }
            }
        }

        try {
            llmService?.generateTextStream(prompt, callback)
        } catch (e: Exception) {
            Log.e("MainActivity", "Error calling AIDL", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }
}
