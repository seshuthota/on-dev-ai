package ai.ondev.snapdragonlab.data.repository

import android.content.Context
import android.util.Log
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import ai.ondev.snapdragonlab.data.download.ModelDownloadWorker
import ai.ondev.snapdragonlab.data.local.DownloadTaskDao
import ai.ondev.snapdragonlab.data.local.ModelDao
import ai.ondev.snapdragonlab.data.local.entity.DownloadTaskEntity
import ai.ondev.snapdragonlab.data.local.entity.ModelEntity
import ai.ondev.snapdragonlab.data.remote.HuggingFaceApi
import ai.ondev.snapdragonlab.domain.model.DownloadState
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import java.io.File
import java.util.UUID

class DownloadRepository(
    private val context: Context,
    private val modelDao: ModelDao,
    private val downloadTaskDao: DownloadTaskDao,
    private val hfApi: HuggingFaceApi,
) {
    private val workManager = WorkManager.getInstance(context)

    /**
     * Enqueue a download for a GGUF file from Hugging Face.
     * Creates the model record and download task, then schedules the WorkManager job.
     *
     * @return model ID
     */
    suspend fun enqueueDownload(
        repoId: String,
        filename: String,
        sizeBytes: Long,
    ): Result<String> {
        // Check for existing download/model
        val existingCount = modelDao.countByRepoAndFilename(repoId, filename)
        if (existingCount > 0) {
            return Result.failure(
                IllegalStateException("Model $filename from $repoId is already downloaded or downloading"),
            )
        }

        val modelId = UUID.randomUUID().toString()
        val modelsDir = File(context.filesDir, "models")
        modelsDir.mkdirs()
        val targetPath = File(modelsDir, filename).absolutePath
        val downloadUrl = hfApi.getDownloadUrl(repoId, filename)
        val now = System.currentTimeMillis()

        // Create model record
        modelDao.insert(
            ModelEntity(
                id = modelId,
                hfRepoId = repoId,
                filename = filename,
                localPath = targetPath,
                sizeBytes = sizeBytes,
                sha256 = null,
                status = "DOWNLOADING",
                createdAt = now,
                updatedAt = now,
            ),
        )

        // Create WorkManager job
        val workRequest = OneTimeWorkRequestBuilder<ModelDownloadWorker>()
            .setInputData(
                workDataOf(
                    ModelDownloadWorker.KEY_DOWNLOAD_URL to downloadUrl,
                    ModelDownloadWorker.KEY_TARGET_PATH to targetPath,
                    ModelDownloadWorker.KEY_MODEL_ID to modelId,
                    ModelDownloadWorker.KEY_TOTAL_BYTES to sizeBytes,
                ),
            )
            .build()

        val workName = "${ModelDownloadWorker.UNIQUE_WORK_PREFIX}$modelId"

        // Create download task record
        downloadTaskDao.insert(
            DownloadTaskEntity(
                id = UUID.randomUUID().toString(),
                modelId = modelId,
                workManagerId = workRequest.id.toString(),
                progressBytes = 0,
                totalBytes = sizeBytes,
                state = "QUEUED",
                errorMessage = null,
            ),
        )

        // Enqueue — KEEP policy prevents duplicate work for same model
        workManager.enqueueUniqueWork(workName, ExistingWorkPolicy.KEEP, workRequest)

        Log.i("OnDevAI", "Download enqueued: modelId=$modelId file=$filename url=$downloadUrl")
        return Result.success(modelId)
    }

    /**
     * Cancel an active download.
     */
    suspend fun cancelDownload(modelId: String) {
        val workName = "${ModelDownloadWorker.UNIQUE_WORK_PREFIX}$modelId"
        workManager.cancelUniqueWork(workName)

        // Update records
        val task = downloadTaskDao.getByModelId(modelId)
        if (task != null) {
            downloadTaskDao.update(task.copy(state = "FAILED", errorMessage = "Cancelled by user"))
        }

        modelDao.getById(modelId)?.let { model ->
            modelDao.update(model.copy(status = "FAILED", updatedAt = System.currentTimeMillis()))
        }

        // Clean up partial file
        modelDao.getById(modelId)?.localPath?.let { path ->
            File(path).delete()
        }
    }

    /**
     * Delete a downloaded model and its file.
     */
    suspend fun deleteDownloadedModel(modelId: String) {
        modelDao.getById(modelId)?.localPath?.let { path ->
            File(path).delete()
        }
        downloadTaskDao.deleteByModelId(modelId)
        modelDao.deleteById(modelId)
    }

    /**
     * Get download state for a specific model by observing WorkManager.
     */
    fun getDownloadProgress(modelId: String): Flow<DownloadState> {
        val workName = "${ModelDownloadWorker.UNIQUE_WORK_PREFIX}$modelId"
        return workManager.getWorkInfosForUniqueWorkFlow(workName).map { workInfos ->
            val workInfo = workInfos.firstOrNull()
            mapWorkInfoToState(workInfo)
        }
    }

    /**
     * Get all active download tasks.
     */
    fun getActiveDownloads(): Flow<List<DownloadTaskEntity>> =
        downloadTaskDao.getActive()

    private fun mapWorkInfoToState(workInfo: WorkInfo?): DownloadState {
        if (workInfo == null) return DownloadState.Idle

        return when (workInfo.state) {
            WorkInfo.State.ENQUEUED -> DownloadState.Queued
            WorkInfo.State.RUNNING -> {
                val progress = workInfo.progress.getLong(ModelDownloadWorker.KEY_PROGRESS_BYTES, 0L)
                val total = workInfo.progress.getLong(ModelDownloadWorker.KEY_TOTAL_BYTES, 0L)
                DownloadState.Downloading(progress, total)
            }
            WorkInfo.State.SUCCEEDED -> DownloadState.Completed
            WorkInfo.State.FAILED -> {
                val error = workInfo.outputData.getString(ModelDownloadWorker.KEY_ERROR) ?: "Unknown error"
                DownloadState.Failed(error)
            }
            WorkInfo.State.CANCELLED -> DownloadState.Failed("Cancelled")
            WorkInfo.State.BLOCKED -> DownloadState.Queued
        }
    }
}
