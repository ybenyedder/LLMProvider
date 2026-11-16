package com.tree4five.gguf

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

data class BackgroundDownloadState(
    val active: Boolean = false,
    val fileName: String = "",
    val percent: Int = 0,
    val downloadedBytes: Long = 0L,
    val totalBytes: Long = -1L,
    val speedMbPerSec: Double = 0.0,
    val etaSeconds: Long = 0L,
    val statusText: String = "",
    val completedOk: Boolean? = null,
    val eventId: Long = 0L
)

data class ChatUiSnapshot(
    val messages: List<ChatMessage> = emptyList(),
    val isGenerating: Boolean = false,
    val composerStats: String = "",
    val version: Long = 0L
)

/**
 * Process-wide state holder shared between [LLMInferenceService] and [MainActivity] so
 * both model downloads and active LLM conversations continue uninterrupted when the user
 * leaves the app and restore seamlessly upon return.
 */
object BackgroundTaskCoordinator {

    val downloadCancelled = AtomicBoolean(false)
    val isDownloading = AtomicBoolean(false)

    private val lock = Any()
    private val nextEventId = AtomicLong(1L)
    private val nextTurnId = AtomicLong(1L)
    private val chatMessages = mutableListOf<ChatMessage>()
    private var activeTurnId: Long = 0L
    private var chatVersion: Long = 0L

    private val _downloadState = MutableStateFlow(BackgroundDownloadState())
    val downloadState: StateFlow<BackgroundDownloadState> = _downloadState.asStateFlow()

    private val _chatState = MutableStateFlow(ChatUiSnapshot())
    val chatState: StateFlow<ChatUiSnapshot> = _chatState.asStateFlow()

    fun isBusy(): Boolean = isDownloading.get() || _chatState.value.isGenerating

    fun updateDownloadState(transform: (BackgroundDownloadState) -> BackgroundDownloadState) {
        _downloadState.value = transform(_downloadState.value)
    }

    fun emitDownloadFinished(fileName: String, ok: Boolean, message: String) {
        _downloadState.value = BackgroundDownloadState(
            active = false,
            fileName = fileName,
            percent = if (ok) 100 else 0,
            statusText = message,
            completedOk = ok,
            eventId = nextEventId.getAndIncrement()
        )
    }

    fun startChatTurn(userPrompt: String, initialMeta: String = "Prefill..."): Long {
        synchronized(lock) {
            chatMessages.add(ChatMessage("user", userPrompt))
            chatMessages.add(ChatMessage("assistant", "", initialMeta))
            activeTurnId = nextTurnId.getAndIncrement()
            chatVersion++
            _chatState.value = ChatUiSnapshot(
                messages = chatMessages.map { it.copy() },
                isGenerating = true,
                composerStats = initialMeta,
                version = chatVersion
            )
            return activeTurnId
        }
    }

    fun appendToken(turnId: Long, token: String, statStr: String, shortStat: String) {
        synchronized(lock) {
            if (turnId != activeTurnId || chatMessages.isEmpty()) return
            val last = chatMessages.last()
            last.content += token
            last.meta = statStr
            chatVersion++
            _chatState.value = ChatUiSnapshot(
                messages = chatMessages.map { it.copy() },
                isGenerating = true,
                composerStats = shortStat,
                version = chatVersion
            )
        }
    }

    fun finishChatTurn(turnId: Long, finalText: String? = null, finalMeta: String? = null) {
        synchronized(lock) {
            if (turnId != activeTurnId || chatMessages.isEmpty()) return
            val last = chatMessages.last()
            if (finalText != null && last.content.isEmpty()) {
                last.content = finalText
            }
            if (finalMeta != null) {
                last.meta = finalMeta
            } else if (last.meta == "Prefill...") {
                last.meta = "Terminé"
            }
            chatVersion++
            _chatState.value = ChatUiSnapshot(
                messages = chatMessages.map { it.copy() },
                isGenerating = false,
                composerStats = _chatState.value.composerStats,
                version = chatVersion
            )
        }
    }

    fun clearChat() {
        synchronized(lock) {
            activeTurnId = 0L
            chatMessages.clear()
            chatVersion++
            _chatState.value = ChatUiSnapshot(
                messages = emptyList(),
                isGenerating = false,
                composerStats = "",
                version = chatVersion
            )
        }
    }

    fun recentMessagesSnapshot(limit: Int = 10): List<ChatMessage> {
        synchronized(lock) {
            val start = maxOf(0, chatMessages.size - limit)
            return chatMessages.subList(start, chatMessages.size).map { it.copy() }
        }
    }
}
