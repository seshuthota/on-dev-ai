package ai.ondev.snapdragonlab.domain.model

sealed class DownloadState {
    data object Idle : DownloadState()
    data object Queued : DownloadState()
    data class Downloading(val progressBytes: Long, val totalBytes: Long) : DownloadState() {
        val progressFraction: Float
            get() = if (totalBytes > 0) progressBytes.toFloat() / totalBytes else 0f
    }
    data object Paused : DownloadState()
    data object Completed : DownloadState()
    data class Failed(val error: String) : DownloadState()
}
