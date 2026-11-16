package com.tree4five.gguf

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.nio.charset.StandardCharsets
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Lightweight embedded HTTP API server (OpenAI + Ollama compatible) backed directly by
 * [LLMInferenceService] so external tools and local apps on the device can call:
 *  - `GET  /health`
 *  - `GET  /v1/models` and `GET /api/tags`
 *  - `POST /v1/chat/completions` (streaming SSE & non-streaming)
 *  - `POST /v1/completions`
 *  - `POST /v1/embeddings`
 *  - `POST /api/generate` and `POST /api/chat`
 */
class LocalHttpServer(
    private val port: Int,
    private val engineBridge: EngineBridge
) {
    interface EngineBridge {
        fun isModelLoaded(): Boolean
        fun modelName(): String
        fun getMetricsJson(): String
        fun embedVector(text: String): FloatArray?
        fun generateStreaming(
            prompt: String,
            nPredict: Int,
            temperature: Float,
            topK: Int,
            topP: Float,
            repeatPenalty: Float,
            specDecoding: Boolean,
            onToken: (String) -> Boolean
        ): String
    }

    private val running = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null
    private val executor = Executors.newCachedThreadPool()

    fun start(): Boolean {
        if (running.get()) return true
        return try {
            val ss = ServerSocket(port, 16, InetAddress.getByName("127.0.0.1"))
            serverSocket = ss
            running.set(true)
            Thread({
                Log.i(TAG, "Local HTTP API listening on http://127.0.0.1:$port")
                while (running.get()) {
                    try {
                        val client = ss.accept()
                        executor.execute { handleClient(client) }
                    } catch (e: Exception) {
                        if (running.get()) {
                            Log.w(TAG, "Accept error: ${e.message}")
                        }
                    }
                }
            }, "LLMHttpAccept").apply { isDaemon = true }.start()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to bind 127.0.0.1:$port", e)
            false
        }
    }

    fun stop() {
        running.set(false)
        try {
            serverSocket?.close()
        } catch (_: Exception) {
        }
        serverSocket = null
        executor.shutdownNow()
    }

    private fun handleClient(socket: Socket) {
        socket.use { sock ->
            sock.soTimeout = 300_000
            val inp = BufferedInputStream(sock.getInputStream())
            val out = BufferedOutputStream(sock.getOutputStream())

            val headerBytes = readUntilHeadersEnd(inp) ?: return
            val headerText = String(headerBytes, StandardCharsets.ISO_8859_1)
            val lines = headerText.split("\r\n")
            if (lines.isEmpty()) return
            val parts = lines[0].split(" ")
            if (parts.size < 2) return
            val method = parts[0].uppercase()
            val path = parts[1].substringBefore("?")

            var contentLength = 0
            for (i in 1 until lines.size) {
                val line = lines[i]
                val colon = line.indexOf(':')
                if (colon > 0 && line.substring(0, colon).trim().equals("Content-Length", ignoreCase = true)) {
                    contentLength = line.substring(colon + 1).trim().toIntOrNull() ?: 0
                }
            }

            val bodyBytes = if (contentLength > 0 && contentLength <= 8 * 1024 * 1024) {
                val buf = ByteArray(contentLength)
                var read = 0
                while (read < contentLength) {
                    val r = inp.read(buf, read, contentLength - read)
                    if (r <= 0) break
                    read += r
                }
                buf.copyOf(read)
            } else {
                ByteArray(0)
            }
            val bodyStr = String(bodyBytes, StandardCharsets.UTF_8)

            if (method == "OPTIONS") {
                sendJson(out, 204, "No Content", "")
                return
            }

            when {
                method == "GET" && path == "/health" -> {
                    val metrics = JSONObject(engineBridge.getMetricsJson())
                    val ok = engineBridge.isModelLoaded()
                    metrics.put("status", if (ok) "ok" else "no_model")
                    sendJson(out, if (ok) 200 else 503, if (ok) "OK" else "Service Unavailable", metrics.toString())
                }

                method == "GET" && path == "/v1/models" -> {
                    val m = JSONObject().apply {
                        put("object", "list")
                        val data = JSONArray()
                        if (engineBridge.isModelLoaded()) {
                            data.put(JSONObject().apply {
                                put("id", engineBridge.modelName())
                                put("object", "model")
                                put("owned_by", "tree4five-rustingstrata")
                            })
                        }
                        put("data", data)
                    }
                    sendJson(out, 200, "OK", m.toString())
                }

                method == "GET" && path == "/api/tags" -> {
                    val models = JSONArray()
                    if (engineBridge.isModelLoaded()) {
                        models.put(JSONObject().apply {
                            put("name", engineBridge.modelName())
                            put("model", engineBridge.modelName())
                        })
                    }
                    sendJson(out, 200, "OK", JSONObject().put("models", models).toString())
                }

                method == "POST" && path == "/v1/embeddings" -> {
                    if (!engineBridge.isModelLoaded()) {
                        sendJson(out, 503, "Service Unavailable", """{"error":"Model not loaded"}""")
                        return
                    }
                    val req = JSONObject(bodyStr.ifBlank { "{}" })
                    val text = when (val raw = req.opt("input")) {
                        is JSONArray -> if (raw.length() > 0) raw.optString(0, "") else ""
                        else -> raw?.toString() ?: ""
                    }
                    val vec = engineBridge.embedVector(text)
                    if (vec == null) {
                        sendJson(out, 500, "Internal Server Error", """{"error":"Embedding failed"}""")
                        return
                    }
                    val embArr = JSONArray().apply { vec.forEach { put(it.toDouble()) } }
                    val resp = JSONObject().apply {
                        put("object", "list")
                        put("model", engineBridge.modelName())
                        put("data", JSONArray().put(JSONObject().apply {
                            put("object", "embedding")
                            put("index", 0)
                            put("embedding", embArr)
                        }))
                    }
                    sendJson(out, 200, "OK", resp.toString())
                }

                method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions") -> {
                    if (!engineBridge.isModelLoaded()) {
                        sendJson(out, 503, "Service Unavailable", """{"error":"Model not loaded"}""")
                        return
                    }
                    val req = JSONObject(bodyStr.ifBlank { "{}" })
                    val stream = req.optBoolean("stream", false)
                    val maxTokens = req.optInt("max_tokens", req.optInt("n_predict", 512))
                    val temp = req.optDouble("temperature", 0.0).toFloat()
                    val topK = req.optInt("top_k", 40)
                    val topP = req.optDouble("top_p", 0.95).toFloat()
                    val rep = req.optDouble("repeat_penalty", 1.10).toFloat()
                    val promptInput = if (req.has("messages")) req.toString() else req.optString("prompt", "")
                    val formatted = PromptManager.formatPrompt(promptInput, "OpenAI")
                    val modelId = engineBridge.modelName()
                    val id = "chatcmpl-${System.currentTimeMillis()}"

                    if (stream) {
                        writeSseHeaders(out)
                        engineBridge.generateStreaming(formatted, maxTokens, temp, topK, topP, rep, true) { tok ->
                            val chunk = JSONObject().apply {
                                put("id", id)
                                put("object", "chat.completion.chunk")
                                put("model", modelId)
                                put("choices", JSONArray().put(JSONObject().apply {
                                    put("index", 0)
                                    put("delta", JSONObject().put("content", tok))
                                    put("finish_reason", JSONObject.NULL)
                                }))
                            }
                            writeSseData(out, chunk.toString())
                        }
                        val endChunk = JSONObject().apply {
                            put("id", id)
                            put("object", "chat.completion.chunk")
                            put("model", modelId)
                            put("choices", JSONArray().put(JSONObject().apply {
                                put("index", 0)
                                put("delta", JSONObject())
                                put("finish_reason", "stop")
                            }))
                        }
                        writeSseData(out, endChunk.toString())
                        writeSseData(out, "[DONE]")
                    } else {
                        val full = engineBridge.generateStreaming(formatted, maxTokens, temp, topK, topP, rep, true) { true }
                        val resp = JSONObject().apply {
                            put("id", id)
                            put("object", "chat.completion")
                            put("model", modelId)
                            put("choices", JSONArray().put(JSONObject().apply {
                                put("index", 0)
                                put("message", JSONObject().put("role", "assistant").put("content", full))
                                put("finish_reason", "stop")
                            }))
                        }
                        sendJson(out, 200, "OK", resp.toString())
                    }
                }

                method == "POST" && (path == "/api/generate" || path == "/api/chat") -> {
                    if (!engineBridge.isModelLoaded()) {
                        sendJson(out, 503, "Service Unavailable", """{"error":"Model not loaded"}""")
                        return
                    }
                    val req = JSONObject(bodyStr.ifBlank { "{}" })
                    val stream = req.optBoolean("stream", false)
                    val opts = req.optJSONObject("options")
                    val maxTokens = opts?.optInt("num_predict", 512) ?: 512
                    val temp = opts?.optDouble("temperature", 0.0)?.toFloat() ?: 0.0f
                    val topK = opts?.optInt("top_k", 40) ?: 40
                    val topP = opts?.optDouble("top_p", 0.95)?.toFloat() ?: 0.95f
                    val rep = opts?.optDouble("repeat_penalty", 1.10)?.toFloat() ?: 1.10f
                    val promptInput = if (req.has("messages")) req.toString() else req.optString("prompt", "")
                    val formatted = PromptManager.formatPrompt(promptInput, "Ollama")
                    val modelId = engineBridge.modelName()

                    if (stream) {
                        writeNdjsonHeaders(out)
                        engineBridge.generateStreaming(formatted, maxTokens, temp, topK, topP, rep, true) { tok ->
                            val line = JSONObject().apply {
                                put("model", modelId)
                                if (path == "/api/chat") {
                                    put("message", JSONObject().put("role", "assistant").put("content", tok))
                                } else {
                                    put("response", tok)
                                }
                                put("done", false)
                            }
                            writeRawLine(out, line.toString())
                        }
                        val doneLine = JSONObject().apply {
                            put("model", modelId)
                            put("done", true)
                            if (path == "/api/chat") {
                                put("message", JSONObject().put("role", "assistant").put("content", ""))
                            } else {
                                put("response", "")
                            }
                        }
                        writeRawLine(out, doneLine.toString())
                    } else {
                        val full = engineBridge.generateStreaming(formatted, maxTokens, temp, topK, topP, rep, true) { true }
                        val resp = JSONObject().apply {
                            put("model", modelId)
                            put("done", true)
                            if (path == "/api/chat") {
                                put("message", JSONObject().put("role", "assistant").put("content", full))
                            } else {
                                put("response", full)
                            }
                        }
                        sendJson(out, 200, "OK", resp.toString())
                    }
                }

                else -> sendJson(out, 404, "Not Found", """{"error":"Unknown endpoint $path"}""")
            }
        }
    }

    private fun readUntilHeadersEnd(inp: BufferedInputStream): ByteArray? {
        val buf = ByteArrayOutputStream()
        var state = 0
        while (buf.size() < 65536) {
            val b = inp.read()
            if (b == -1) return if (buf.size() == 0) null else buf.toByteArray()
            buf.write(b)
            state = when (state) {
                0 -> if (b == '\r'.code) 1 else 0
                1 -> if (b == '\n'.code) 2 else if (b == '\r'.code) 1 else 0
                2 -> if (b == '\r'.code) 3 else 0
                3 -> if (b == '\n'.code) return buf.toByteArray() else 0
                else -> 0
            }
        }
        return buf.toByteArray()
    }

    private fun sendJson(out: BufferedOutputStream, code: Int, reason: String, body: String) {
        val bytes = body.toByteArray(StandardCharsets.UTF_8)
        val headers = buildString {
            append("HTTP/1.1 ").append(code).append(' ').append(reason).append("\r\n")
            append("Content-Type: application/json; charset=utf-8\r\n")
            append("Access-Control-Allow-Origin: *\r\n")
            append("Access-Control-Allow-Headers: Content-Type, Authorization\r\n")
            append("Content-Length: ").append(bytes.size).append("\r\n")
            append("Connection: close\r\n\r\n")
        }
        out.write(headers.toByteArray(StandardCharsets.ISO_8859_1))
        out.write(bytes)
        out.flush()
    }

    private fun writeSseHeaders(out: BufferedOutputStream) {
        val headers = "HTTP/1.1 200 OK\r\n" +
            "Content-Type: text/event-stream; charset=utf-8\r\n" +
            "Cache-Control: no-cache\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Connection: close\r\n\r\n"
        out.write(headers.toByteArray(StandardCharsets.ISO_8859_1))
        out.flush()
    }

    private fun writeSseData(out: BufferedOutputStream, payload: String): Boolean {
        return try {
            out.write("data: $payload\n\n".toByteArray(StandardCharsets.UTF_8))
            out.flush()
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun writeNdjsonHeaders(out: BufferedOutputStream) {
        val headers = "HTTP/1.1 200 OK\r\n" +
            "Content-Type: application/x-ndjson; charset=utf-8\r\n" +
            "Cache-Control: no-cache\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Connection: close\r\n\r\n"
        out.write(headers.toByteArray(StandardCharsets.ISO_8859_1))
        out.flush()
    }

    private fun writeRawLine(out: BufferedOutputStream, line: String): Boolean {
        return try {
            out.write((line + "\n").toByteArray(StandardCharsets.UTF_8))
            out.flush()
            true
        } catch (_: Exception) {
            false
        }
    }

    companion object {
        private const val TAG = "LocalHttpServer"
    }
}
