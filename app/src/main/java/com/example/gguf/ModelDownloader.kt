package com.example.gguf

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {
    
    private const val TAG = "ModelDownloader"

    /**
     * Télécharge un modèle GGUF depuis HuggingFace (ou autre URL) vers le stockage local.
     * @param url L'URL directe du fichier .gguf (ex: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf)
     * @param destinationFile Le fichier de destination local
     * @param onProgress Callback de progression (0 à 100)
     */
    suspend fun downloadModel(
        url: String, 
        destinationFile: File, 
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val connection = URL(url).openConnection() as HttpURLConnection
            connection.connect()

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                Log.e(TAG, "Erreur serveur HTTP ${connection.responseCode}")
                return@withContext false
            }

            val fileLength = connection.contentLength
            val input = connection.inputStream
            val output = FileOutputStream(destinationFile)

            val data = ByteArray(4096)
            var total: Long = 0
            var count: Int
            
            var lastProgress = 0

            while (input.read(data).also { count = it } != -1) {
                total += count
                output.write(data, 0, count)
                
                if (fileLength > 0) {
                    val progress = ((total * 100) / fileLength).toInt()
                    if (progress != lastProgress) {
                        lastProgress = progress
                        withContext(Dispatchers.Main) {
                            onProgress(progress)
                        }
                    }
                }
            }

            output.flush()
            output.close()
            input.close()
            
            Log.d(TAG, "Téléchargement terminé : ${destinationFile.absolutePath}")
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "Erreur de téléchargement", e)
            if (destinationFile.exists()) {
                destinationFile.delete()
            }
            return@withContext false
        }
    }
}
