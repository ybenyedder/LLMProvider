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
    private val items: MutableList<ChatMessage>,
    private val onCopyClick: (String) -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

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
            is UserVH -> holder.bind(msg)
            is AssistantVH -> holder.bind(msg, onCopyClick)
        }
    }

    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: MutableList<Any>
    ) {
        if (payloads.contains(PAYLOAD_STREAM) && holder is AssistantVH) {
            holder.bind(items[position], onCopyClick)
        } else {
            super.onBindViewHolder(holder, position, payloads)
        }
    }

    fun notifyLastStreamingUpdate() {
        if (items.isNotEmpty()) {
            notifyItemChanged(items.size - 1, PAYLOAD_STREAM)
        }
    }

    private class UserVH(view: View) : RecyclerView.ViewHolder(view) {
        private val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        fun bind(msg: ChatMessage) {
            tvMessage.text = msg.content
        }
    }

    private class AssistantVH(view: View) : RecyclerView.ViewHolder(view) {
        private val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        private val layoutMeta: View = view.findViewById(R.id.layoutMeta)
        private val tvMeta: TextView = view.findViewById(R.id.tvMeta)
        private val btnCopy: ImageView = view.findViewById(R.id.btnCopy)

        fun bind(msg: ChatMessage, onCopy: (String) -> Unit) {
            tvMessage.text = msg.content.ifEmpty { "..." }
            if (msg.meta.isBlank() && msg.content.isBlank()) {
                layoutMeta.visibility = View.GONE
            } else {
                layoutMeta.visibility = View.VISIBLE
                tvMeta.text = msg.meta
            }
            btnCopy.setOnClickListener {
                if (msg.content.isNotBlank()) onCopy(msg.content)
            }
        }
    }

    companion object {
        private const val VIEW_USER = 0
        private const val VIEW_ASSISTANT = 1
        private const val PAYLOAD_STREAM = "stream"
    }
}
