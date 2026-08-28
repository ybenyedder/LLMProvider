package com.example.gguf

import android.content.Context
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {
    
    private const val TAG = "ModelDownloader"

    /**
     * Downloads a GGUF model from an HTTP URL, or copies it from a content:// or file:// URI.
     */
    suspend fun downloadOrCopyModel(
        sourceUri: String, 
        destinationFile: File, 
        context: Context,
        onProgress: (Int) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val input: InputStream
            var fileLength: Long = -1

            when {
                sourceUri.startsWith("http://") || sourceUri.startsWith("https://") -> {
                    var connection = URL(sourceUri).openConnection() as HttpURLConnection
                    connection.instanceFollowRedirects = true
                    connection.connect()
                    
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
                        Log.e(TAG, "HTTP Server Error ${connection.responseCode}")
                        return@withContext false
                    }
                    fileLength = connection.contentLengthLong
                    input = connection.inputStream
                }
                sourceUri.startsWith("content://") -> {
                    val uri = Uri.parse(sourceUri)
                    context.contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                        val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                        if (cursor.moveToFirst() && sizeIndex != -1) {
                            fileLength = cursor.getLong(sizeIndex)
                        }
                    }
                    input = context.contentResolver.openInputStream(uri) ?: throw Exception("Cannot open content URI")
                }
                sourceUri.startsWith("file://") -> {
                    val file = File(Uri.parse(sourceUri).path!!)
                    fileLength = file.length()
                    input = FileInputStream(file)
                }
                else -> {
                    Log.e(TAG, "Unsupported scheme: ${sourceUri}")
                    return@withContext false
                }
            }

            val output = FileOutputStream(destinationFile)
            val data = ByteArray(65536) // 64KB buffer
            var total: Long = 0
            var count: Int
            var lastProgress = 0

            input.use { inStream ->
                output.use { outStream ->
                    while (inStream.read(data).also { count = it } != -1) {
                        total += count
                        outStream.write(data, 0, count)
                        
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
                }
            }
            
            Log.d(TAG, "Transfer complete: ${destinationFile.absolutePath}")
            return@withContext true
            
        } catch (e: Exception) {
            Log.e(TAG, "Transfer error", e)
            if (destinationFile.exists()) {
                destinationFile.delete()
            }
            return@withContext false
        }
    }
}
