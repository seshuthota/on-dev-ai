package ai.ondev.snapdragonlab.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import ai.ondev.snapdragonlab.data.local.entity.ConversationEntity
import kotlinx.coroutines.flow.Flow

data class ConversationWithPreview(
    val id: String,
    val title: String,
    val modelId: String?,
    val modelFilename: String?,
    val lastMessageContent: String?,
    val messageCount: Int,
    val createdAt: Long,
    val updatedAt: Long,
)

@Dao
interface ConversationDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(conversation: ConversationEntity)

    @Update
    suspend fun update(conversation: ConversationEntity)

    @Query("SELECT * FROM conversations WHERE id = :id")
    suspend fun getById(id: String): ConversationEntity?

    @Query(
        """
        SELECT 
            c.id,
            c.title,
            c.model_id AS modelId,
            m.filename AS modelFilename,
            (SELECT content FROM messages WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) AS lastMessageContent,
            (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) AS messageCount,
            c.created_at AS createdAt,
            c.updated_at AS updatedAt
        FROM conversations c
        LEFT JOIN models m ON c.model_id = m.id
        ORDER BY c.updated_at DESC
        """
    )
    fun getAllWithPreview(): Flow<List<ConversationWithPreview>>

    @Query("DELETE FROM conversations WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("UPDATE conversations SET title = :title, updated_at = :updatedAt WHERE id = :id")
    suspend fun updateTitle(id: String, title: String, updatedAt: Long)

    @Query("UPDATE conversations SET model_id = :modelId, updated_at = :updatedAt WHERE id = :id")
    suspend fun updateModelId(id: String, modelId: String?, updatedAt: Long)

    @Query("UPDATE conversations SET updated_at = :updatedAt WHERE id = :id")
    suspend fun touch(id: String, updatedAt: Long)
}
