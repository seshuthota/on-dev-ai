package ai.ondev.snapdragonlab.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import ai.ondev.snapdragonlab.data.local.entity.DownloadTaskEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface DownloadTaskDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(task: DownloadTaskEntity)

    @Update
    suspend fun update(task: DownloadTaskEntity)

    @Query("SELECT * FROM download_tasks WHERE model_id = :modelId")
    suspend fun getByModelId(modelId: String): DownloadTaskEntity?

    @Query("SELECT * FROM download_tasks WHERE state IN ('QUEUED', 'DOWNLOADING')")
    fun getActive(): Flow<List<DownloadTaskEntity>>

    @Query("DELETE FROM download_tasks WHERE model_id = :modelId")
    suspend fun deleteByModelId(modelId: String)
}
