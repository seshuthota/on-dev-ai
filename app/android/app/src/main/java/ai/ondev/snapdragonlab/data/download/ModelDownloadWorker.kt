package ai.ondev.snapdragonlab.data.download

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.TimeUnit
import ai.ondev.snapdragonlab.data.local.AppDatabase
import ai.ondev.snapdragonlab.data.local.DownloadTaskDao
import ai.ondev.snapdragonlab.data.local.ModelDao

/**
 * WorkManager worker that downloads a GGUF model file with progress reporting.
 *
 * Input data keys:
 *   - KEY_DOWNLOAD_URL: direct download URL
 *   - KEY_TARGET_PATH: local file path to save to
 *   - KEY_MODEL_ID: model entity ID for status updates
 *   - KEY_TOTAL_BYTES: expected file size (for progress, 0 if unknown)
 *
 * Progress output:
 *   - KEY_PROGRESS_BYTES: bytes downloaded so far
 *   - KEY_TOTAL_BYTES: total expected bytes
 */
class ModelDownloadWorker(
    appContext: Context,
    workerParams: WorkerParameters,
) : CoroutineWorker(appContext, workerParams) {

    private val client = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()

    override suspend fun doWork(): Result {
        val downloadUrl = inputData.getString(KEY_DOWNLOAD_URL)
            ?: return Result.failure(workDataOf(KEY_ERROR to "Missing download URL"))
        val targetPath = inputData.getString(KEY_TARGET_PATH)
            ?: return Result.failure(workDataOf(KEY_ERROR to "Missing target path"))
        val modelId = inputData.getString(KEY_MODEL_ID) ?: ""
        val expectedTotalBytes = inputData.getLong(KEY_TOTAL_BYTES, 0L)

        val targetFile = File(targetPath)
        targetFile.parentFile?.mkdirs()

        Log.i(TAG, "Download starting: modelId=$modelId url=$downloadUrl target=$targetPath")

        val database = AppDatabase.create(applicationContext)
        val modelDao = database.modelDao()
        val taskDao = database.downloadTaskDao()

        return try {
            taskDao.getByModelId(modelId)?.let {
                taskDao.update(it.copy(state = "DOWNLOADING"))
            }

            val metrics = downloadFile(downloadUrl, targetFile, expectedTotalBytes, taskDao, modelDao, modelId)

            modelDao.getById(modelId)?.let {
                modelDao.update(it.copy(status = "READY", updatedAt = System.currentTimeMillis()))
            }
            taskDao.getByModelId(modelId)?.let {
                taskDao.update(
                    it.copy(
                        state = "COMPLETED",
                        progressBytes = metrics.currentBytes,
                        totalBytes = metrics.totalBytes,
                        errorMessage = null,
                    ),
                )
            }
            Result.success(
                workDataOf(
                    KEY_PROGRESS_BYTES to metrics.currentBytes,
                    KEY_TOTAL_BYTES to metrics.totalBytes,
                ),
            )
        } catch (e: Exception) {
            Log.e(TAG, "Download failed: modelId=$modelId", e)
            val error = e.message ?: "Download failed"
            modelDao.getById(modelId)?.let {
                modelDao.update(it.copy(status = "FAILED", updatedAt = System.currentTimeMillis()))
            }
            taskDao.getByModelId(modelId)?.let {
                taskDao.update(it.copy(state = "FAILED", errorMessage = error))
            }
            Result.failure(workDataOf(KEY_ERROR to error))
        }
    }

    private suspend fun downloadFile(
        url: String,
        targetFile: File,
        expectedTotalBytes: Long,
        taskDao: DownloadTaskDao,
        modelDao: ModelDao,
        modelId: String,
    ): DownloadMetrics {
        val downloadedBytes = if (targetFile.exists()) targetFile.length() else 0L

        val requestBuilder = Request.Builder().url(url)
        if (downloadedBytes > 0) {
            requestBuilder.header("Range", "bytes=$downloadedBytes-")
        }
        val request = requestBuilder.build()
        val response = client.newCall(request).execute()

        val isPartial = response.code == 206
        if (!response.isSuccessful && !isPartial) {
            throw IOException("HTTP ${response.code}: ${response.message}")
        }

        val body = response.body
            ?: throw IOException("Empty response body")

        val append = isPartial && downloadedBytes > 0
        if (!append && targetFile.exists()) {
            targetFile.delete()
        }

        val startBytes = if (append) downloadedBytes else 0L
        val totalBytes = if (append) startBytes + body.contentLength() else body.contentLength().let { len ->
            if (len > 0) len else expectedTotalBytes
        }

        var currentBytes = startBytes
        var lastProgressUpdate = 0L

        body.byteStream().use { inputStream ->
            FileOutputStream(targetFile, append).use { outputStream ->
                val buffer = ByteArray(BUFFER_SIZE)
                var bytesRead: Int

                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    if (isStopped) {
                        Log.i(TAG, "Download cancelled")
                        modelDao.getById(modelId)?.let {
                            modelDao.update(it.copy(status = "FAILED", updatedAt = System.currentTimeMillis()))
                        }
                        taskDao.getByModelId(modelId)?.let {
                            taskDao.update(it.copy(state = "FAILED", errorMessage = "Cancelled"))
                        }
                        throw IOException("Download cancelled")
                    }

                    outputStream.write(buffer, 0, bytesRead)
                    currentBytes += bytesRead

                    // Throttle progress updates to every 500ms
                    val now = System.currentTimeMillis()
                    if (now - lastProgressUpdate > PROGRESS_UPDATE_INTERVAL_MS) {
                        setProgress(
                            workDataOf(
                                KEY_PROGRESS_BYTES to currentBytes,
                                KEY_TOTAL_BYTES to totalBytes,
                            ),
                        )
                        taskDao.getByModelId(modelId)?.let {
                            taskDao.update(it.copy(progressBytes = currentBytes, totalBytes = totalBytes))
                        }
                        lastProgressUpdate = now
                    }
                }
            }
        }

        // Final progress update
        setProgress(
            workDataOf(
                KEY_PROGRESS_BYTES to currentBytes,
                KEY_TOTAL_BYTES to totalBytes,
            ),
        )

        // Validate download
        if (!targetFile.exists() || targetFile.length() == 0L) {
            throw IOException("Downloaded file is empty")
        }

        if (totalBytes > 0 && targetFile.length() != totalBytes) {
            Log.w(TAG, "Size mismatch: expected=$totalBytes actual=${targetFile.length()}")
            // Don't fail — content-length from CDN can be inaccurate for redirected responses
        }

        Log.i(TAG, "Download complete: ${targetFile.absolutePath} (${targetFile.length()} bytes)")

        return DownloadMetrics(
            currentBytes = currentBytes,
            totalBytes = totalBytes,
        )
    }

    private data class DownloadMetrics(
        val currentBytes: Long,
        val totalBytes: Long,
    )

    companion object {
        const val TAG = "OnDevAI"
        const val UNIQUE_WORK_PREFIX = "ondevai_model_download_"

        const val KEY_DOWNLOAD_URL = "download_url"
        const val KEY_TARGET_PATH = "target_path"
        const val KEY_MODEL_ID = "model_id"
        const val KEY_TOTAL_BYTES = "total_bytes"
        const val KEY_PROGRESS_BYTES = "progress_bytes"
        const val KEY_ERROR = "error"

        private const val BUFFER_SIZE = 8 * 1024 // 8 KB
        private const val PROGRESS_UPDATE_INTERVAL_MS = 500L
    }
}
