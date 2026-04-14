package ai.ondev.snapdragonlab.domain.model

data class ConversationSummary(
    val id: String,
    val title: String,
    val modelId: String?,
    val modelName: String?,
    val lastMessage: String?,
    val messageCount: Int,
    val createdAt: Long,
    val updatedAt: Long,
)
