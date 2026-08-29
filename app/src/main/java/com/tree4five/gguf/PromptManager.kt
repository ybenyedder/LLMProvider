package com.tree4five.gguf

import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject

object PromptManager {
    fun formatPrompt(input: String): String {
        return try {
            val json = JSONObject(input)
            if (json.has("messages")) {
                val messages = json.getJSONArray("messages")
                val builder = java.lang.StringBuilder()
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
        } catch (e: JSONException) {
            // Not a JSON string, assume it's a raw user prompt and wrap it in a standard ChatML template
            // This prevents instruct models from returning empty spaces or stopping immediately
            return "<|im_start|>user\n$input<|im_end|>\n<|im_start|>assistant\n"
        }
    }
}
