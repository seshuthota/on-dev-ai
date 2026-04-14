package ai.ondev.snapdragonlab.ui.settings

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import ai.ondev.snapdragonlab.data.repository.InferenceRepository
import ai.ondev.snapdragonlab.data.repository.InferenceState
import ai.ondev.snapdragonlab.data.repository.SettingsRepository
import ai.ondev.snapdragonlab.domain.model.BackendTarget
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class SettingsViewModel(
    private val inferenceRepository: InferenceRepository,
    private val settingsRepository: SettingsRepository,
) : ViewModel() {

    val inferenceState: StateFlow<InferenceState> =
        inferenceRepository.state
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), InferenceState())

    val selectedBackend: StateFlow<BackendTarget> =
        settingsRepository.backendTargetFlow
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), BackendTarget.CPU)

    val systemPrompt: StateFlow<String> =
        settingsRepository.systemPromptFlow
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), SettingsRepository.DEFAULT_SYSTEM_PROMPT)

    fun selectBackend(target: BackendTarget) {
        viewModelScope.launch {
            settingsRepository.setBackendTarget(target)
        }
    }

    fun updateSystemPrompt(prompt: String) {
        viewModelScope.launch {
            settingsRepository.setSystemPrompt(prompt)
        }
    }

    fun isBackendAvailable(target: BackendTarget): Boolean =
        inferenceRepository.isBackendAvailable(target)

    fun getRuntimeInfo(): String = inferenceRepository.getRuntimeInfo()

    class Factory(
        private val inferenceRepository: InferenceRepository,
        private val settingsRepository: SettingsRepository,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            SettingsViewModel(inferenceRepository, settingsRepository) as T
    }
}
