package ai.ondev.snapdragonlab.data.repository

import ai.ondev.snapdragonlab.data.local.ModelDao
import ai.ondev.snapdragonlab.data.local.entity.ModelEntity
import ai.ondev.snapdragonlab.domain.model.ModelStatus
import ai.ondev.snapdragonlab.domain.model.ModelSummary
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class ModelRepository(private val modelDao: ModelDao) {

    fun getAllModels(): Flow<List<ModelSummary>> =
        modelDao.getAll().map { entities -> entities.map { it.toDomain() } }

    fun getReadyModels(): Flow<List<ModelSummary>> =
        modelDao.getReadyModels().map { entities -> entities.map { it.toDomain() } }

    suspend fun getModelById(id: String): ModelSummary? =
        modelDao.getById(id)?.toDomain()

    suspend fun refreshModelSizeFromFile(id: String) {
        val entity = modelDao.getById(id) ?: return
        entity.localPath?.let { path ->
            val file = java.io.File(path)
            if (file.exists()) {
                modelDao.update(entity.copy(sizeBytes = file.length(), updatedAt = System.currentTimeMillis()))
            }
        }
    }

    suspend fun insertModel(model: ModelSummary) {
        modelDao.insert(model.toEntity())
    }

    suspend fun updateModelStatus(id: String, status: ModelStatus, localPath: String? = null) {
        val entity = modelDao.getById(id) ?: return
        modelDao.update(
            entity.copy(
                status = status.name,
                localPath = localPath ?: entity.localPath,
                updatedAt = System.currentTimeMillis(),
            ),
        )
    }

    suspend fun deleteModel(id: String) {
        modelDao.deleteById(id)
    }

    suspend fun hasExistingModel(repoId: String, filename: String): Boolean =
        modelDao.countByRepoAndFilename(repoId, filename) > 0

    private fun ModelEntity.toDomain() = ModelSummary(
        id = id,
        hfRepoId = hfRepoId,
        filename = filename,
        localPath = localPath,
        sizeBytes = sizeBytes,
        quantSuffix = ModelSummary.extractQuantSuffix(filename),
        status = try {
            ModelStatus.valueOf(status)
        } catch (_: IllegalArgumentException) {
            ModelStatus.FAILED
        },
    )

    private fun ModelSummary.toEntity() = ModelEntity(
        id = id,
        hfRepoId = hfRepoId,
        filename = filename,
        localPath = localPath,
        sizeBytes = sizeBytes,
        sha256 = null,
        status = status.name,
        createdAt = System.currentTimeMillis(),
        updatedAt = System.currentTimeMillis(),
    )
}
