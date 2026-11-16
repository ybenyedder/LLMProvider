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
    private val expandedThoughtPositions = mutableSetOf<Int>()

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
            is AssistantVH -> holder.bind(
                pos = position,
                msg = msg,
                isExpanded = expandedThoughtPositions.contains(position),
                onToggleThought = { p ->
                    if (!expandedThoughtPositions.add(p)) {
                        expandedThoughtPositions.remove(p)
                    }
                    notifyItemChanged(p, PAYLOAD_STREAM)
                },
                onCopy = onCopyClick,
                onRegenerate = onRegenerateClick
            )
        }
    }

    override fun onBindViewHolder(
        holder: RecyclerView.ViewHolder,
        position: Int,
        payloads: MutableList<Any>
    ) {
        if (payloads.contains(PAYLOAD_STREAM) && holder is AssistantVH) {
            holder.bind(
                pos = position,
                msg = items[position],
                isExpanded = expandedThoughtPositions.contains(position),
                onToggleThought = { p ->
                    if (!expandedThoughtPositions.add(p)) {
                        expandedThoughtPositions.remove(p)
                    }
                    notifyItemChanged(p, PAYLOAD_STREAM)
                },
                onCopy = onCopyClick,
                onRegenerate = onRegenerateClick
            )
        } else {
            super.onBindViewHolder(holder, position, payloads)
        }
    }

    fun submitMessages(newList: List<ChatMessage>) {
        if (newList.isEmpty()) {
            expandedThoughtPositions.clear()
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
        expandedThoughtPositions.retainAll { it < newList.size }
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
        private val layoutThoughtCard: View = view.findViewById(R.id.layoutThoughtCard)
        private val layoutThoughtHeader: View = view.findViewById(R.id.layoutThoughtHeader)
        private val viewThoughtDot: View = view.findViewById(R.id.viewThoughtDot)
        private val tvThoughtLabel: TextView = view.findViewById(R.id.tvThoughtLabel)
        private val ivThoughtChevron: ImageView = view.findViewById(R.id.ivThoughtChevron)
        private val tvThoughtContent: TextView = view.findViewById(R.id.tvThoughtContent)
        private val tvMessage: TextView = view.findViewById(R.id.tvMessage)
        private val layoutMeta: View = view.findViewById(R.id.layoutMeta)
        private val tvMeta: TextView = view.findViewById(R.id.tvMeta)
        private val btnRegenerate: ImageView = view.findViewById(R.id.btnRegenerate)
        private val btnCopy: ImageView = view.findViewById(R.id.btnCopy)

        fun bind(
            pos: Int,
            msg: ChatMessage,
            isExpanded: Boolean,
            onToggleThought: (Int) -> Unit,
            onCopy: (String) -> Unit,
            onRegenerate: (Int) -> Unit
        ) {
            val parsed = parseThinking(msg.content)

            if (parsed.thought != null) {
                layoutThoughtCard.visibility = View.VISIBLE
                tvThoughtLabel.text = if (parsed.inProgress) {
                    "Réflexion en cours..."
                } else {
                    "Réflexion"
                }
                viewThoughtDot.setBackgroundResource(
                    if (parsed.inProgress) R.drawable.dot_status_gen else R.drawable.dot_status_ready
                )
                tvThoughtContent.text = parsed.thought.ifEmpty { "..." }
                tvThoughtContent.visibility = if (isExpanded) View.VISIBLE else View.GONE
                ivThoughtChevron.rotation = if (isExpanded) 180f else 0f
                layoutThoughtHeader.setOnClickListener { onToggleThought(pos) }
            } else {
                layoutThoughtCard.visibility = View.GONE
            }

            if (parsed.answer.isNotEmpty()) {
                tvMessage.visibility = View.VISIBLE
                tvMessage.text = parsed.answer
            } else if (parsed.thought != null) {
                tvMessage.visibility = View.GONE
            } else {
                tvMessage.visibility = View.VISIBLE
                tvMessage.text = "..."
            }

            if (msg.meta.isBlank() && msg.content.isBlank()) {
                layoutMeta.visibility = View.GONE
            } else {
                layoutMeta.visibility = View.VISIBLE
                tvMeta.text = msg.meta
            }

            btnCopy.setOnClickListener {
                val copyText = parsed.answer.ifBlank { msg.content }
                if (copyText.isNotBlank()) onCopy(copyText)
            }
            btnRegenerate.setOnClickListener {
                onRegenerate(pos)
            }
        }

        private data class ParsedReply(
            val thought: String?,
            val answer: String,
            val inProgress: Boolean
        )

        private fun parseThinking(raw: String): ParsedReply {
            val openTag = "<think>"
            val closeTag = "</think>"
            if (!raw.contains(openTag)) {
                return ParsedReply(thought = null, answer = raw, inProgress = false)
            }
            val before = raw.substringBefore(openTag).trim()
            val afterOpen = raw.substringAfter(openTag)
            return if (afterOpen.contains(closeTag)) {
                val thought = afterOpen.substringBefore(closeTag).trim()
                val after = afterOpen.substringAfter(closeTag).trimStart()
                val combinedAnswer = if (before.isNotEmpty() && after.isNotEmpty()) {
                    "$before\n$after"
                } else {
                    before.ifEmpty { after }
                }
                ParsedReply(thought = thought, answer = combinedAnswer, inProgress = false)
            } else {
                ParsedReply(thought = afterOpen.trimStart(), answer = before, inProgress = true)
            }
        }
    }

    companion object {
        private const val VIEW_USER = 0
        private const val VIEW_ASSISTANT = 1
        private const val PAYLOAD_STREAM = "stream"
    }
}
