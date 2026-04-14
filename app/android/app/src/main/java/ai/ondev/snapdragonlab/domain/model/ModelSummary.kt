package ai.ondev.snapdragonlab.domain.model

data class ModelSummary(
    val id: String,
    val hfRepoId: String,
    val filename: String,
    val localPath: String?,
    val sizeBytes: Long,
    val quantSuffix: String?,
    val status: ModelStatus,
) {
    val displayName: String
        get() = filename.removeSuffix(".gguf")

    val sizeLabel: String
        get() = formatBytes(sizeBytes)

    companion object {
        fun formatBytes(bytes: Long): String = when {
            bytes < 1024L -> "$bytes B"
            bytes < 1024L * 1024 -> "%.1f KB".format(bytes / 1024.0)
            bytes < 1024L * 1024 * 1024 -> "%.1f MB".format(bytes / (1024.0 * 1024))
            else -> "%.2f GB".format(bytes / (1024.0 * 1024 * 1024))
        }

        fun extractQuantSuffix(filename: String): String? {
            val quantPattern = Regex(
                """[_-](Q[0-9]+_[A-Z0-9_]+|F16|F32|BF16|IQ[0-9]+_[A-Z]+)""",
                RegexOption.IGNORE_CASE,
            )
            return quantPattern.find(filename)?.groupValues?.getOrNull(1)?.uppercase()
        }
    }
}
