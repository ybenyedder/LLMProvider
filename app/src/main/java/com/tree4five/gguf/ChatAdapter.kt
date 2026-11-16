package com.tree4five.gguf

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class ChatMessage(
    val role: String,
    var content: String,
    var meta: String = ""
)

class ChatAdapter(
    private val onCopyClick: (String) -> Unit,
    private val onUserClick: (Int, ChatMessage) -> Unit,
    private val onRegenerateClick: (Int) -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val items = mutableListOf<ChatMessage>()

    override fun getItemViewType(position: Int): Int =
        if (items[position].role == "user") VIEW_USER else VIEW_ASSISTANT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val inflater = LayoutInflater.from(parent.context)
        return if (viewType == VIEW_USER) {
            UserVH(inflater.inflate(R.layout.item_chat_user, parent, false))
        } else {
            AssistantVH(inflater.inflate(R.layout.item_chat_assistant, parent, false))
        }
    }

    override fun getItemCount(): Int = items.size

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val msg = items[position]
        when (holder) {
            is UserVH -> holder.bind(position, msg, onUserClick)
            is AssistantVH -> holder.bind(position, msg, onCopyClick, onRegenerateClick)
        }
    }

    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: MutableList<Any>
    ) {
        if (payloads.contains(PAYLOAD_STREAM) && holder is AssistantVH) {
            holder.bind(position, items[position], onCopyClick, onRegenerateClick)
        } else {
            super.onBindViewHolder(holder, position, payloads)
        }
    }

    fun submitMessages(newList: List<ChatMessage>) {
        if (newList.isEmpty()) {
            if (items.isNotEmpty()) {
                items.clear()
                notifyDataSetChanged()
            }
            return
        }
        if (newList.size == items.size && items.isNotEmpty()) {
            val lastIdx = items.size - 1
            items[lastIdx] = newList[lastIdx].copy()
            notifyItemChanged(lastIdx, PAYLOAD_STREAM)
            return
        }
        items.clear()
        items.addAll(newList.map { it.copy() })
        notifyDataSetChanged()
    }

    private class UserVH(view: View) : RecyclerView.ViewHolder(view) {
        private val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        fun bind(pos: Int, msg: ChatMessage, onUserClick: (Int, ChatMessage) -> Unit) {
            tvMessage.text = msg.content
            tvMessage.setOnClickListener { onUserClick(pos, msg) }
            tvMessage.setOnLongClickListener {
                onUserClick(pos, msg)
                true
            }
        }
    }

    private class AssistantVH(view: View) : RecyclerView.ViewHolder(view) {
        private val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        private val layoutMeta: View = view.findViewById(R.id.layoutMeta)
        private val tvMeta: TextView = view.findViewById(R.id.tvMeta)
        private val btnRegenerate: ImageView = view.findViewById(R.id.btnRegenerate)
        private val btnCopy: ImageView = view.findViewById(R.id.btnCopy)

        fun bind(
            pos: Int,
            msg: ChatMessage,
            onCopy: (String) -> Unit,
            onRegenerate: (Int) -> Unit
        ) {
            val display = formatAssistantDisplay(msg.content)
            tvMessage.text = display.ifEmpty { "..." }
            if (msg.meta.isBlank() && msg.content.isBlank()) {
                layoutMeta.visibility = View.GONE
            } else {
                layoutMeta.visibility = View.VISIBLE
                tvMeta.text = msg.meta
            }
            btnCopy.setOnClickListener {
                if (msg.content.isNotBlank()) onCopy(msg.content)
            }
            btnRegenerate.setOnClickListener {
                onRegenerate(pos)
            }
        }

        private fun formatAssistantDisplay(raw: String): String {
            if (!raw.contains("<think>")) return raw
            val afterOpen = raw.substringAfter("<think>")
            return if (afterOpen.contains("</think>")) {
                val thought = afterOpen.substringBefore("</think>").trim()
                val answer = afterOpen.substringAfter("</think>").trimStart()
                buildString {
                    if (thought.isNotEmpty()) {
                        append("Réflexion :\n").append(thought)
                        if (answer.isNotEmpty()) append("\n\n")
                    }
                    append(answer)
                }
            } else {
                val thought = afterOpen.trimStart()
                if (thought.isEmpty()) "Réflexion en cours..." else "Réflexion en cours :\n$thought"
            }
        }
    }

    companion object {
        private const val VIEW_USER = 0
        private const val VIEW_ASSISTANT = 1
        private const val PAYLOAD_STREAM = "stream"
    }
}
