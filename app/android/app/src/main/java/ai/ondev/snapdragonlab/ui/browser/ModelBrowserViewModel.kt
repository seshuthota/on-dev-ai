package ai.ondev.snapdragonlab.ui.browser

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import ai.ondev.snapdragonlab.data.remote.HuggingFaceApi
import ai.ondev.snapdragonlab.data.remote.model.HfModelSearchResult
import ai.ondev.snapdragonlab.data.remote.model.HfRepoSibling
import ai.ondev.snapdragonlab.data.repository.DownloadRepository
import ai.ondev.snapdragonlab.data.repository.ModelRepository
import ai.ondev.snapdragonlab.domain.model.DownloadState
import ai.ondev.snapdragonlab.domain.model.ModelSummary
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class GgufFileInfo(
    val filename: String,
    val sizeBytes: Long,
    val quantSuffix: String?,
    val sizeLabel: String,
) {
    val isLargeFile: Boolean get() = sizeBytes > HuggingFaceApi.LARGE_FILE_THRESHOLD_BYTES
}

data class RepoDetail(
    val repoId: String,
    val ggufFiles: List<GgufFileInfo>,
    val isLoading: Boolean = false,
)

data class BrowserUiState(
    val searchQuery: String = "",
    val searchResults: List<HfModelSearchResult> = emptyList(),
    val isSearching: Boolean = false,
    val searchError: String? = null,
    val expandedRepo: RepoDetail? = null,
    val downloadStates: Map<String, DownloadState> = emptyMap(),
    val showLargeFileWarning: GgufFileInfo? = null,
    val pendingDownloadRepoId: String? = null,
)

@OptIn(FlowPreview::class)
class ModelBrowserViewModel(
    private val hfApi: HuggingFaceApi,
    private val downloadRepository: DownloadRepository,
    private val modelRepository: ModelRepository,
) : ViewModel() {

    private val _uiState = MutableStateFlow(BrowserUiState())
    val uiState: StateFlow<BrowserUiState> = _uiState.asStateFlow()

    val existingModels: StateFlow<List<ModelSummary>> =
        modelRepository.getAllModels()
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _searchQuery = MutableStateFlow("")
    private var searchJob: Job? = null
    private var repoDetailJob: Job? = null

    init {
        // Debounced search
        viewModelScope.launch {
            _searchQuery.debounce(400).collectLatest { query ->
                if (query.isBlank()) {
                    _uiState.value = _uiState.value.copy(
                        searchResults = emptyList(),
                        isSearching = false,
                        searchError = null,
                    )
                    return@collectLatest
                }
                performSearch(query)
            }
        }
    }

    fun updateSearchQuery(query: String) {
        _searchQuery.value = query
        _uiState.value = _uiState.value.copy(
            searchQuery = query,
            isSearching = query.isNotBlank(),
        )
    }

    fun expandRepo(repoId: String) {
        if (_uiState.value.expandedRepo?.repoId == repoId) {
            _uiState.value = _uiState.value.copy(expandedRepo = null)
            return
        }

        _uiState.value = _uiState.value.copy(
            expandedRepo = RepoDetail(repoId = repoId, ggufFiles = emptyList(), isLoading = true),
        )

        repoDetailJob?.cancel()
        repoDetailJob = viewModelScope.launch {
            val result = hfApi.getRepoDetail(repoId)
            result.fold(
                onSuccess = { detail ->
                    val ggufFiles = detail.siblings
                        .filter { it.filename.endsWith(".gguf", ignoreCase = true) }
                        .map { sibling ->
                            GgufFileInfo(
                                filename = sibling.filename,
                                sizeBytes = sibling.size ?: 0L,
                                quantSuffix = ModelSummary.extractQuantSuffix(sibling.filename),
                                sizeLabel = ModelSummary.formatBytes(sibling.size ?: 0L),
                            )
                        }
                        .sortedBy { it.sizeBytes }

                    _uiState.value = _uiState.value.copy(
                        expandedRepo = RepoDetail(
                            repoId = repoId,
                            ggufFiles = ggufFiles,
                            isLoading = false,
                        ),
                    )
                },
                onFailure = { error ->
                    _uiState.value = _uiState.value.copy(
                        expandedRepo = RepoDetail(
                            repoId = repoId,
                            ggufFiles = emptyList(),
                            isLoading = false,
                        ),
                        searchError = "Failed to load repo files: ${error.message}",
                    )
                },
            )
        }
    }

    fun requestDownload(repoId: String, file: GgufFileInfo) {
        if (file.isLargeFile) {
            _uiState.value = _uiState.value.copy(
                showLargeFileWarning = file,
                pendingDownloadRepoId = repoId,
            )
            return
        }
        startDownload(repoId, file)
    }

    fun confirmLargeFileDownload() {
        val file = _uiState.value.showLargeFileWarning ?: return
        val repoId = _uiState.value.pendingDownloadRepoId ?: return
        _uiState.value = _uiState.value.copy(
            showLargeFileWarning = null,
            pendingDownloadRepoId = null,
        )
        startDownload(repoId, file)
    }

    fun dismissLargeFileWarning() {
        _uiState.value = _uiState.value.copy(
            showLargeFileWarning = null,
            pendingDownloadRepoId = null,
        )
    }

    fun cancelDownload(modelId: String) {
        viewModelScope.launch {
            downloadRepository.cancelDownload(modelId)
        }
    }

    fun deleteModel(modelId: String) {
        viewModelScope.launch {
            downloadRepository.deleteDownloadedModel(modelId)
        }
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(searchError = null)
    }

    fun observeDownloadState(modelId: String) {
        viewModelScope.launch {
            downloadRepository.getDownloadProgress(modelId).collectLatest { state ->
                val current = _uiState.value.downloadStates.toMutableMap()
                current[modelId] = state
                _uiState.value = _uiState.value.copy(downloadStates = current)

                // Sync completed state back to Room
                if (state is DownloadState.Completed) {
                    modelRepository.updateModelStatus(
                        modelId,
                        ai.ondev.snapdragonlab.domain.model.ModelStatus.READY,
                    )
                } else if (state is DownloadState.Failed) {
                    modelRepository.updateModelStatus(
                        modelId,
                        ai.ondev.snapdragonlab.domain.model.ModelStatus.FAILED,
                    )
                }
            }
        }
    }

    private fun startDownload(repoId: String, file: GgufFileInfo) {
        viewModelScope.launch {
            val result = downloadRepository.enqueueDownload(
                repoId = repoId,
                filename = file.filename,
                sizeBytes = file.sizeBytes,
            )
            result.fold(
                onSuccess = { modelId ->
                    observeDownloadState(modelId)
                },
                onFailure = { error ->
                    _uiState.value = _uiState.value.copy(
                        searchError = error.message ?: "Download failed to start",
                    )
                },
            )
        }
    }

    private suspend fun performSearch(query: String) {
        _uiState.value = _uiState.value.copy(isSearching = true, searchError = null)

        val result = hfApi.searchModels(query)
        result.fold(
            onSuccess = { models ->
                _uiState.value = _uiState.value.copy(
                    searchResults = models,
                    isSearching = false,
                    searchError = null,
                )
            },
            onFailure = { error ->
                _uiState.value = _uiState.value.copy(
                    searchResults = emptyList(),
                    isSearching = false,
                    searchError = error.message ?: "Search failed",
                )
            },
        )
    }

    class Factory(
        private val hfApi: HuggingFaceApi,
        private val downloadRepository: DownloadRepository,
        private val modelRepository: ModelRepository,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            ModelBrowserViewModel(hfApi, downloadRepository, modelRepository) as T
    }
}
