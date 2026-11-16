package com.tree4five.gguf

import org.json.JSONException
import org.json.JSONObject

object PromptManager {

    /**
     * Default ChatML formatter preserving full compatibility with existing clients and tests.
     */
    fun formatPrompt(input: String): String {
        return formatPrompt(input, protocol = "ChatML", systemPrompt = "")
    }

    /**
     * Formats either a JSON payload (`messages` or `prompt`) or raw text according to the
     * selected protocol (`ChatML`, `OpenAI`, `Ollama`, `Raw`) with an optional system prompt.
     */
    fun formatPrompt(input: String, protocol: String, systemPrompt: String = ""): String {
        val sys = systemPrompt.trim()
        return try {
            val json = JSONObject(input)
            if (json.has("messages")) {
                val messages = json.getJSONArray("messages")
                val builder = StringBuilder()
                var hasSystem = false
                for (i in 0 until messages.length()) {
                    if (messages.getJSONObject(i).optString("role", "") == "system") {
                        hasSystem = true
                        break
                    }
                }
                if (!hasSystem && sys.isNotEmpty()) {
                    builder.append("<|im_start|>system\n").append(sys).append("<|im_end|>\n")
                }
                for (i in 0 until messages.length()) {
                    val msg = messages.getJSONObject(i)
                    val role = msg.optString("role", "user")
                    val content = msg.optString("content", "")
                    builder.append("<|im_start|>").append(role).append("\n")
                    builder.append(content).append("<|im_end|>\n")
                }
                builder.append("<|im_start|>assistant\n")
                builder.toString()
            } else {
                json.optString("prompt", input)
            }
        } catch (_: JSONException) {
            if (protocol.equals("Raw", ignoreCase = true) || input.contains("<|im_start|>")) {
                input
            } else {
                val builder = StringBuilder()
                if (sys.isNotEmpty()) {
                    builder.append("<|im_start|>system\n").append(sys).append("<|im_end|>\n")
                }
                builder.append("<|im_start|>user\n").append(input).append("<|im_end|>\n<|im_start|>assistant\n")
                builder.toString()
            }
        }
    }
}
