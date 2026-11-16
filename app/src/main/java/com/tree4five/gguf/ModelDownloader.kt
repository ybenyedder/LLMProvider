package com.tree4five.gguf

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
import java.util.concurrent.atomic.AtomicBoolean

data class DownloadStatus(
    val percent: Int,
    val downloadedBytes: Long,
    val totalBytes: Long,
    val speedMbPerSec: Double,
    val etaSeconds: Long
)

object ModelDownloader {

    private const val TAG = "ModelDownloader"

    /**
     * Backward-compatible helper used by existing callers and tests.
     */
    suspend fun downloadOrCopyModel(
        sourceUri: String,
        destinationFile: File,
        context: Context?,
        onProgress: (Int) -> Unit
    ): Boolean = downloadOrCopyModelDetailed(
        sourceUri = sourceUri,
        destinationFile = destinationFile,
        context = context,
        cancelled = null
    ) { status ->
        onProgress(status.percent)
    }

    /**
     * Downloads a GGUF model from HTTP/HTTPS or copies from `content://`/`file://` into a `.part`
     * file, verifies the 4-byte `"GGUF"` magic header, and atomically renames to `destinationFile`.
     */
    suspend fun downloadOrCopyModelDetailed(
        sourceUri: String,
        destinationFile: File,
        context: Context?,
        cancelled: AtomicBoolean? = null,
        onProgress: (DownloadStatus) -> Unit
    ): Boolean = withContext(Dispatchers.IO) {
        val partFile = File(destinationFile.parentFile, "${destinationFile.name}.part")
        var connection: HttpURLConnection? = null
        try {
            val input: InputStream
            var fileLength: Long = -1

            when {
                sourceUri.startsWith("http://") || sourceUri.startsWith("https://") -> {
                    var currentUrl = sourceUri
                    var redirectCount = 0
                    while (true) {
                        val conn = (URL(currentUrl).openConnection() as HttpURLConnection).apply {
                            instanceFollowRedirects = false
                            connectTimeout = 20_000
                            readTimeout = 30_000
                            setRequestProperty("User-Agent", "Tree4Five-LLMProvider/2.0 (RustingStrata)")
                        }
                        conn.connect()
                        val code = conn.responseCode
                        if (code in 300..399 && redirectCount < 8) {
                            val nextLocation = conn.getHeaderField("Location")
                            conn.disconnect()
                            if (nextLocation.isNullOrBlank()) {
                                Log.e(TAG, "Redirect $code without Location header")
                                return@withContext false
                            }
                            currentUrl = URL(URL(currentUrl), nextLocation).toString()
                            redirectCount++
                            continue
                        }
                        if (code != HttpURLConnection.HTTP_OK) {
                            Log.e(TAG, "HTTP Server Error $code for $currentUrl")
                            conn.disconnect()
                            return@withContext false
                        }
                        connection = conn
                        fileLength = conn.contentLengthLong
                        input = conn.inputStream
                        break
                    }
                }
                sourceUri.startsWith("content://") -> {
                    val uri = Uri.parse(sourceUri)
                    context?.contentResolver?.query(uri, null, null, null, null)?.use { cursor ->
                        val sizeIndex = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE)
                        if (cursor.moveToFirst() && sizeIndex != -1) {
                            fileLength = cursor.getLong(sizeIndex)
                        }
                    }
                    input = context?.contentResolver?.openInputStream(uri)
                        ?: throw IllegalStateException("Cannot open content URI")
                }
                sourceUri.startsWith("file://") -> {
                    val file = File(Uri.parse(sourceUri).path!!)
                    fileLength = file.length()
                    input = FileInputStream(file)
                }
                else -> {
                    Log.e(TAG, "Unsupported scheme: $sourceUri")
                    return@withContext false
                }
            }

            if (partFile.exists()) {
                partFile.delete()
            }
            val output = FileOutputStream(partFile)
            val data = ByteArray(131072) // 128 KB buffer
            var total: Long = 0
            var count: Int
            var lastUiMs = 0L
            val startNs = System.nanoTime()
            val header = ByteArray(4)
            var headerLen = 0

            input.use { inStream ->
                output.use { outStream ->
                    while (inStream.read(data).also { count = it } != -1) {
                        if (cancelled?.get() == true) {
                            Log.i(TAG, "Transfer cancelled by user")
                            throw InterruptedException("Cancelled")
                        }
                        if (headerLen < 4) {
                            val take = minOf(4 - headerLen, count)
                            System.arraycopy(data, 0, header, headerLen, take)
                            headerLen += take
                            if (headerLen == 4) {
                                val magic = String(header, Charsets.US_ASCII)
                                if (magic != "GGUF") {
                                    throw IllegalArgumentException("Invalid file header '$magic' (expected GGUF)")
                                }
                            }
                        }
                        total += count
                        outStream.write(data, 0, count)

                        val nowMs = System.currentTimeMillis()
                        if (nowMs - lastUiMs >= 150 || (fileLength > 0 && total == fileLength)) {
                            lastUiMs = nowMs
                            val elapsedSec = ((System.nanoTime() - startNs) / 1_000_000_000.0).coerceAtLeast(0.05)
                            val speedBps = total / elapsedSec
                            val speedMb = speedBps / (1024.0 * 1024.0)
                            val percent = if (fileLength > 0) ((total * 100) / fileLength).toInt().coerceIn(0, 100) else 0
                            val etaSec = if (fileLength > total && speedBps > 1.0) {
                                ((fileLength - total) / speedBps).toLong()
                            } else {
                                0L
                            }
                            val status = DownloadStatus(percent, total, fileLength, speedMb, etaSec)
                            withContext(Dispatchers.Main) {
                                onProgress(status)
                            }
                        }
                    }
                    outStream.flush()
                }
            }

            if (headerLen < 4 || String(header, Charsets.US_ASCII) != "GGUF") {
                throw IllegalArgumentException("File is too small or not a valid GGUF")
            }

            if (destinationFile.exists()) {
                destinationFile.delete()
            }
            if (!partFile.renameTo(destinationFile)) {
                partFile.copyTo(destinationFile, overwrite = true)
                partFile.delete()
            }
            Log.d(TAG, "Transfer complete: ${destinationFile.absolutePath} ($total bytes)")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Transfer error: ${e.message}", e)
            if (partFile.exists()) {
                partFile.delete()
            }
            if (destinationFile.exists() && destinationFile.length() == 0L) {
                destinationFile.delete()
            }
            false
        } finally {
            connection?.disconnect()
        }
    }
}
