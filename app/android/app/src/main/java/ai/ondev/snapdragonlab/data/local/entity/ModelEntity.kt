package ai.ondev.snapdragonlab.data.local.entity

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "models")
data class ModelEntity(
    @PrimaryKey
    val id: String,
    @ColumnInfo(name = "hf_repo_id")
    val hfRepoId: String,
    val filename: String,
    @ColumnInfo(name = "local_path")
    val localPath: String?,
    @ColumnInfo(name = "size_bytes")
    val sizeBytes: Long,
    val sha256: String?,
    val status: String, // DOWNLOADING, READY, FAILED, DELETED
    @ColumnInfo(name = "created_at")
    val createdAt: Long,
    @ColumnInfo(name = "updated_at")
    val updatedAt: Long,
)
