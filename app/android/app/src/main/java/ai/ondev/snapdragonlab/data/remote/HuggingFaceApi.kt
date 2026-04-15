package ai.ondev.snapdragonlab.data.remote

import ai.ondev.snapdragonlab.data.remote.model.HfModelSearchResult
import ai.ondev.snapdragonlab.data.remote.model.HfRepoDetail
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.IOException
import java.util.concurrent.TimeUnit

class HuggingFaceApi(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build(),
) {
    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
        coerceInputValues = true
    }

    /**
     * Search for GGUF models on Hugging Face.
     * Uses the public models API with GGUF tag filter.
     */
    suspend fun searchModels(
        query: String,
        limit: Int = 20,
    ): Result<List<HfModelSearchResult>> = withContext(Dispatchers.IO) {
        val url = buildString {
            append(BASE_URL)
            append("/api/models?")
            append("search=")
            append(java.net.URLEncoder.encode(query, "UTF-8"))
            append("&filter=gguf")
            append("&sort=downloads")
            append("&direction=-1")
            append("&limit=")
            append(limit)
        }

        executeRequest(url) { body ->
            json.decodeFromString<List<HfModelSearchResult>>(body)
        }
    }

    /**
     * Get detailed repo info including file listing.
     * Siblings list contains all files in the repo.
     */
    suspend fun getRepoDetail(repoId: String): Result<HfRepoDetail> = withContext(Dispatchers.IO) {
        val url = "$BASE_URL/api/models/$repoId?blobs=true"
        executeRequest(url) { body ->
            json.decodeFromString<HfRepoDetail>(body)
        }
    }

    /**
     * Build the direct download URL for a file in a repo.
     * This URL redirects to the CDN; OkHttp follows redirects automatically.
     */
    fun getDownloadUrl(repoId: String, filename: String): String =
        "$BASE_URL/$repoId/resolve/main/$filename"

    private inline fun <T> executeRequest(
        url: String,
        parse: (String) -> T,
    ): Result<T> {
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/json")
            .build()

        return try {
            val response = client.newCall(request).execute()
            if (!response.isSuccessful) {
                return Result.failure(
                    IOException("HF API error: ${response.code} ${response.message}"),
                )
            }
            val body = response.body?.string()
                ?: return Result.failure(IOException("Empty response body"))
            Result.success(parse(body))
        } catch (e: IOException) {
            Result.failure(e)
        } catch (e: Exception) {
            Result.failure(IOException("Parse error: ${e.message}", e))
        }
    }

    companion object {
        const val BASE_URL = "https://huggingface.co"
        const val LARGE_FILE_THRESHOLD_BYTES = 4L * 1024 * 1024 * 1024 // 4 GB
    }
}
