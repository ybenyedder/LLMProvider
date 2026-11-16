package com.tree4five.gguf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class BackgroundTaskCoordinatorTest {

    @Test
    fun chatTurnSurvivesAndStreamsAcrossObservers() {
        BackgroundTaskCoordinator.clearChat()
        val turnId = BackgroundTaskCoordinator.startChatTurn("Bonjour", "Prefill...")
        assertTrue(BackgroundTaskCoordinator.isBusy())
        assertEquals(2, BackgroundTaskCoordinator.chatState.value.messages.size)

        BackgroundTaskCoordinator.appendToken(turnId, "Salut", "TTFT 12 ms · 25.0 tok/s", "25.0 tok/s")
        BackgroundTaskCoordinator.appendToken(turnId, " !", "TTFT 12 ms · 26.0 tok/s", "26.0 tok/s")
        assertEquals("Salut !", BackgroundTaskCoordinator.chatState.value.messages.last().content)

        BackgroundTaskCoordinator.finishChatTurn(turnId)
        assertFalse(BackgroundTaskCoordinator.chatState.value.isGenerating)
        BackgroundTaskCoordinator.clearChat()
    }

    @Test
    fun downloadStateTracksActiveAndCompleted() {
        BackgroundTaskCoordinator.isDownloading.set(true)
        assertTrue(BackgroundTaskCoordinator.isBusy())
        BackgroundTaskCoordinator.emitDownloadFinished("qwen.gguf", true, "Modèle prêt : qwen.gguf")
        BackgroundTaskCoordinator.isDownloading.set(false)
        assertFalse(BackgroundTaskCoordinator.isBusy())
        assertEquals(true, BackgroundTaskCoordinator.downloadState.value.completedOk)
        assertEquals(100, BackgroundTaskCoordinator.downloadState.value.percent)
    }
}
