package com.example.gguf

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import de.kherud.llama.LlamaModel
import de.kherud.llama.ModelParameters
import de.kherud.llama.InferenceParameters
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File

class LLMInferenceService : Service() {

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var llamaModel: LlamaModel? = null

    companion object {
        private const val TAG = "LLMInferenceService"
        private const val NOTIFICATION_ID = 1
        private const val CHANNEL_ID = "LLM_SERVICE_CHANNEL"
        private const val MODEL_PATH = "/data/local/tmp/model.gguf" // Mettre à jour avec le chemin réel
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
                description = "Maintient le modèle LLM actif en arrière-plan pour les requêtes inter-applications."
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Service d'Inférence LLM")
            .setContentText("Le modèle GGUF est chargé et prêt.")
            .setSmallIcon(android.R.drawable.ic_menu_info_details) // À remplacer par votre icône
            .setOngoing(true)
            .build()

        startForeground(NOTIFICATION_ID, notification)
    }

    private fun initializeModel() {
        serviceScope.launch {
            try {
                val modelFile = File(applicationContext.filesDir, "model.gguf")
                if (!modelFile.exists()) {
                    Log.e(TAG, "Le fichier modèle est introuvable au chemin : ${modelFile.absolutePath}")
                    return@launch
                }

                val params = ModelParameters()
                    .setModelFilePath(modelFile.absolutePath)
                    .setNGpuLayers(-1) // Décharger 100% des couches vers le GPU (Vulkan/CLBlast)
                
                Log.d(TAG, "Initialisation du modèle avec accélération matérielle...")
                llamaModel = LlamaModel(params)
                Log.d(TAG, "Modèle chargé avec succès.")
            } catch (e: Exception) {
                Log.e(TAG, "Erreur lors de l'initialisation du modèle GGUF", e)
            }
        }
    }

    private val binder = object : ILLMService.Stub() {
        override fun generateTextStream(prompt: String, callback: ILLMCallback) {
            // Assigner la tâche au pool Default pour ne pas bloquer le thread Binder
            serviceScope.launch {
                val model = llamaModel
                if (model == null) {
                    callback.onGenerationComplete("Erreur: Modèle non chargé.")
                    return@launch
                }

                try {
                    val inferenceParams = InferenceParameters(prompt)
                        .setNPredict(512) // Limite de tokens pour l'exemple

                    val fullTextBuilder = java.lang.StringBuilder()
                    
                    // Génération itérative
                    for (output in model.generate(inferenceParams)) {
                        val token = output.text
                        fullTextBuilder.append(token)
                        // Stream en temps réel vers le client
                        callback.onTokenReceived(token)
                    }
                    
                    // Indiquer la fin
                    callback.onGenerationComplete(fullTextBuilder.toString())
                } catch (e: Exception) {
                    Log.e(TAG, "Erreur pendant la génération", e)
                    callback.onGenerationComplete("Erreur pendant la génération : ${e.message}")
                }
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder {
        return binder
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        llamaModel?.close() // Libérer proprement la mémoire native
        llamaModel = null
        Log.d(TAG, "Service détruit, modèle libéré.")
    }
}
