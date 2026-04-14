package ai.ondev.snapdragonlab.domain.model

enum class MessageRole {
    SYSTEM,
    USER,
    ASSISTANT,
}

data class ChatMessage(
    val id: String,
    val conversationId: String,
    val role: MessageRole,
    val content: String,
    val tokenCount: Int? = null,
    val createdAt: Long,
)
