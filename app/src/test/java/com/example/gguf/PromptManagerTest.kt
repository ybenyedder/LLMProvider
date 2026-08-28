package com.example.gguf

import org.junit.Assert.assertEquals
import org.junit.Test

class PromptManagerTest {

    @Test
    fun testPlainStringPrompt() {
        val input = "Hello world"
        val output = PromptManager.formatPrompt(input)
        assertEquals("Hello world", output)
    }

    @Test
    fun testJsonPrompt() {
        val input = "{\"prompt\": \"Hello JSON\"}"
        val output = PromptManager.formatPrompt(input)
        assertEquals("Hello JSON", output)
    }

    @Test
    fun testJsonMessagesPrompt() {
        val input = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}"
        val output = PromptManager.formatPrompt(input)
        val expected = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        assertEquals(expected, output)
    }
}
