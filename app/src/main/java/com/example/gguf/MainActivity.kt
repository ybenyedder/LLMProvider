package com.example.gguf

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : AppCompatActivity() {

    private var llmService: ILLMService? = null
    private var isBound = false

    private lateinit var btnDownload: Button
    private lateinit var progressBar: ProgressBar
    private lateinit var tvProgress: TextView
    private lateinit var btnStartService: Button
    private lateinit var etPrompt: EditText
    private lateinit var btnTest: Button
    private lateinit var tvOutput: TextView

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            llmService = ILLMService.Stub.asInterface(service)
            isBound = true
            Toast.makeText(this@MainActivity, "Connecté au service LLM AIDL", Toast.LENGTH_SHORT).show()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            llmService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        checkPermissions()

        btnDownload = findViewById(R.id.btnDownload)
        progressBar = findViewById(R.id.progressBar)
        tvProgress = findViewById(R.id.tvProgress)
        btnStartService = findViewById(R.id.btnStartService)
        etPrompt = findViewById(R.id.etPrompt)
        btnTest = findViewById(R.id.btnTest)
        tvOutput = findViewById(R.id.tvOutput)

        val modelFile = File(filesDir, "model.gguf")

        btnDownload.setOnClickListener {
            // Using SmolLM 135M Instruct GGUF as requested
            val url = "https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct-GGUF/resolve/main/smollm-135m-instruct-q4_k_m.gguf"
            progressBar.visibility = View.VISIBLE
            tvProgress.visibility = View.VISIBLE
            btnDownload.isEnabled = false

            CoroutineScope(Dispatchers.Main).launch {
                val success = ModelDownloader.downloadModel(url, modelFile) { progress ->
                    progressBar.progress = progress
                    tvProgress.text = "${progress}%"
                }
                if (success) {
                    Toast.makeText(this@MainActivity, "Téléchargement terminé", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "Échec du téléchargement", Toast.LENGTH_SHORT).show()
                }
                btnDownload.isEnabled = true
                progressBar.visibility = View.GONE
                tvProgress.visibility = View.GONE
            }
        }

        btnStartService.setOnClickListener {
            val intent = Intent(this, LLMInferenceService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            bindService(intent, connection, Context.BIND_AUTO_CREATE)
        }

        btnTest.setOnClickListener {
            val prompt = etPrompt.text.toString()
            if (prompt.isNotEmpty()) {
                tvOutput.text = ""
                generateText(prompt)
            }
        }
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
            Toast.makeText(this, "Service non lié !", Toast.LENGTH_SHORT).show()
            return
        }

        val callback = object : ILLMCallback.Stub() {
            override fun onTokenReceived(token: String) {
                runOnUiThread {
                    tvOutput.append(token)
                }
            }

            override fun onGenerationComplete(fullText: String) {
                runOnUiThread {
                    tvOutput.append("\n\n[Terminé]")
                }
            }
        }

        try {
            llmService?.generateTextStream(prompt, callback)
        } catch (e: Exception) {
            Log.e("MainActivity", "Erreur lors de l'appel AIDL", e)
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
