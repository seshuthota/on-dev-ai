package ai.ondev.snapdragonlab.data.remote.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class HfModelSearchResult(
    @SerialName("_id") val id: String? = null,
    @SerialName("id") val repoId: String,
    @SerialName("author") val author: String? = null,
    @SerialName("modelId") val modelId: String? = null,
    @SerialName("downloads") val downloads: Int = 0,
    @SerialName("likes") val likes: Int = 0,
    @SerialName("tags") val tags: List<String> = emptyList(),
    @SerialName("lastModified") val lastModified: String? = null,
    @SerialName("pipeline_tag") val pipelineTag: String? = null,
)

@Serializable
data class HfRepoDetail(
    @SerialName("id") val repoId: String,
    @SerialName("author") val author: String? = null,
    @SerialName("downloads") val downloads: Int = 0,
    @SerialName("likes") val likes: Int = 0,
    @SerialName("tags") val tags: List<String> = emptyList(),
    @SerialName("siblings") val siblings: List<HfRepoSibling> = emptyList(),
)

@Serializable
data class HfRepoSibling(
    @SerialName("rfilename") val filename: String,
    @SerialName("size") val size: Long? = null,
)
