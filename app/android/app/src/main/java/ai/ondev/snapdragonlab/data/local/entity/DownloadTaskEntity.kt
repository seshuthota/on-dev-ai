package ai.ondev.snapdragonlab.data.local.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "download_tasks",
    foreignKeys = [
        ForeignKey(
            entity = ModelEntity::class,
            parentColumns = ["id"],
            childColumns = ["model_id"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("model_id")],
)
data class DownloadTaskEntity(
    @PrimaryKey
    val id: String,
    @ColumnInfo(name = "model_id")
    val modelId: String,
    @ColumnInfo(name = "work_manager_id")
    val workManagerId: String?,
    @ColumnInfo(name = "progress_bytes")
    val progressBytes: Long,
    @ColumnInfo(name = "total_bytes")
    val totalBytes: Long,
    val state: String, // QUEUED, DOWNLOADING, PAUSED, COMPLETED, FAILED
    @ColumnInfo(name = "error_message")
    val errorMessage: String?,
)
