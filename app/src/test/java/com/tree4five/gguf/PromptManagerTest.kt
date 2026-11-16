package com.tree4five.gguf

import org.junit.Assert.assertEquals
import org.junit.Test

class PromptManagerTest {

    @Test
    fun testPlainStringPrompt() {
        val input = "Hello world"
        val output = PromptManager.formatPrompt(input)
        val expected = "<|im_start|>user\nHello world<|im_end|>\n<|im_start|>assistant\n"
        assertEquals(expected, output)
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

    @Test
    fun testRawProtocolAndNoDoubleWrap() {
        val preformatted = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        assertEquals(preformatted, PromptManager.formatPrompt(preformatted, "ChatML", "Sys"))
        assertEquals("plain", PromptManager.formatPrompt("plain", "Raw", "Sys"))
    }

    @Test
    fun testSystemPromptInjection() {
        val output = PromptManager.formatPrompt("Hello", "ChatML", "Be brief.")
        val expected = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        assertEquals(expected, output)
    }
}
