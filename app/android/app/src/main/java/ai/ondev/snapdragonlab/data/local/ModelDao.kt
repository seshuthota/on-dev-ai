package ai.ondev.snapdragonlab.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import ai.ondev.snapdragonlab.data.local.entity.ModelEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface ModelDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(model: ModelEntity)

    @Update
    suspend fun update(model: ModelEntity)

    @Query("SELECT * FROM models WHERE id = :id")
    suspend fun getById(id: String): ModelEntity?

    @Query("SELECT * FROM models ORDER BY updated_at DESC")
    fun getAll(): Flow<List<ModelEntity>>

    @Query("SELECT * FROM models WHERE status = :status ORDER BY updated_at DESC")
    fun getByStatus(status: String): Flow<List<ModelEntity>>

    @Query("SELECT * FROM models WHERE status = 'READY' ORDER BY updated_at DESC")
    fun getReadyModels(): Flow<List<ModelEntity>>

    @Query("DELETE FROM models WHERE id = :id")
    suspend fun deleteById(id: String)

    @Query("SELECT COUNT(*) FROM models WHERE hf_repo_id = :repoId AND filename = :filename AND status != 'DELETED'")
    suspend fun countByRepoAndFilename(repoId: String, filename: String): Int
}
