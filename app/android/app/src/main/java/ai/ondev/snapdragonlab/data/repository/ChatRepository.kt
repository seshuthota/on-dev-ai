package ai.ondev.snapdragonlab.data.repository

import ai.ondev.snapdragonlab.data.local.ConversationDao
import ai.ondev.snapdragonlab.data.local.MessageDao
import ai.ondev.snapdragonlab.data.local.entity.ConversationEntity
import ai.ondev.snapdragonlab.data.local.entity.MessageEntity
import ai.ondev.snapdragonlab.domain.model.ChatMessage
import ai.ondev.snapdragonlab.domain.model.ConversationSummary
import ai.ondev.snapdragonlab.domain.model.MessageRole
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.util.UUID

class ChatRepository(
    private val conversationDao: ConversationDao,
    private val messageDao: MessageDao,
) {

    fun getConversations(): Flow<List<ConversationSummary>> =
        conversationDao.getAllWithPreview().map { rows ->
            rows.map { row ->
                ConversationSummary(
                    id = row.id,
                    title = row.title,
                    modelId = row.modelId,
                    modelName = row.modelFilename?.removeSuffix(".gguf"),
                    lastMessage = row.lastMessageContent,
                    messageCount = row.messageCount,
                    createdAt = row.createdAt,
                    updatedAt = row.updatedAt,
                )
            }
        }

    suspend fun getConversation(id: String): ConversationSummary? {
        val entity = conversationDao.getById(id) ?: return null
        val lastMsg = messageDao.getLastByConversationId(id)
        val count = messageDao.countByConversationId(id)
        return ConversationSummary(
            id = entity.id,
            title = entity.title,
            modelId = entity.modelId,
            modelName = null, // caller can resolve from ModelRepository if needed
            lastMessage = lastMsg?.content,
            messageCount = count,
            createdAt = entity.createdAt,
            updatedAt = entity.updatedAt,
        )
    }

    suspend fun createConversation(title: String, modelId: String? = null): String {
        val id = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()
        conversationDao.insert(
            ConversationEntity(
                id = id,
                title = title,
                modelId = modelId,
                createdAt = now,
                updatedAt = now,
            ),
        )
        return id
    }

    suspend fun updateConversationTitle(id: String, title: String) {
        conversationDao.updateTitle(id, title, System.currentTimeMillis())
    }

    suspend fun updateConversationModel(id: String, modelId: String?) {
        conversationDao.updateModelId(id, modelId, System.currentTimeMillis())
    }

    suspend fun deleteConversation(id: String) {
        conversationDao.deleteById(id) // messages cascade-deleted by FK
    }

    fun getMessages(conversationId: String): Flow<List<ChatMessage>> =
        messageDao.getByConversationId(conversationId).map { entities ->
            entities.map { it.toDomain() }
        }

    suspend fun addMessage(
        conversationId: String,
        role: MessageRole,
        content: String,
        tokenCount: Int? = null,
    ): ChatMessage {
        val id = UUID.randomUUID().toString()
        val now = System.currentTimeMillis()
        val entity = MessageEntity(
            id = id,
            conversationId = conversationId,
            role = role.name,
            content = content,
            tokenCount = tokenCount,
            createdAt = now,
        )
        messageDao.insert(entity)
        conversationDao.touch(conversationId, now)
        return entity.toDomain()
    }

    private fun MessageEntity.toDomain() = ChatMessage(
        id = id,
        conversationId = conversationId,
        role = try {
            MessageRole.valueOf(role)
        } catch (_: IllegalArgumentException) {
            MessageRole.SYSTEM
        },
        content = content,
        tokenCount = tokenCount,
        createdAt = createdAt,
    )
}
