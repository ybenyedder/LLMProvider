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
     */
    suspend fun downloadModel(
        url: String, 
        destinationFile: File, 
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            var connection = URL(url).openConnection() as HttpURLConnection
            connection.instanceFollowRedirects = true // Important for HuggingFace
            connection.connect()
            
            // Handle redirect if needed (sometimes HttpURLConnection needs manual redirect handling)
            var redirectCount = 0
            while ((connection.responseCode == HttpURLConnection.HTTP_MOVED_TEMP || 
                    connection.responseCode == HttpURLConnection.HTTP_MOVED_PERM || 
                    connection.responseCode == HttpURLConnection.HTTP_SEE_OTHER) && redirectCount < 5) {
                val newUrl = connection.getHeaderField("Location")
                connection = URL(newUrl).openConnection() as HttpURLConnection
                connection.connect()
                redirectCount++
            }

            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                Log.e(TAG, "Erreur serveur HTTP ${connection.responseCode}")
                return@withContext false
            }

            val fileLength = connection.contentLengthLong // Handle > 2GB files
            val input = connection.inputStream
            val output = FileOutputStream(destinationFile)

            val data = ByteArray(65536) // Increased buffer size to 64KB for faster downloads
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
