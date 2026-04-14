package ai.ondev.snapdragonlab.ui.chat

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import ai.ondev.snapdragonlab.data.repository.ChatRepository
import ai.ondev.snapdragonlab.data.repository.InferenceRepository
import ai.ondev.snapdragonlab.data.repository.ModelRepository
import ai.ondev.snapdragonlab.domain.model.BackendTarget
import ai.ondev.snapdragonlab.domain.model.ChatMessage
import ai.ondev.snapdragonlab.domain.model.ConversationSummary
import ai.ondev.snapdragonlab.domain.model.MessageRole
import ai.ondev.snapdragonlab.domain.model.ModelSummary
import ai.ondev.snapdragonlab.data.repository.SettingsRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class ChatUiState(
    val conversation: ConversationSummary? = null,
    val isGenerating: Boolean = false,
    val draftAssistantContent: String = "",
    val error: String? = null,
    val modelLoadStatus: String = "",
)

class ChatViewModel(
    private val conversationId: String,
    private val chatRepository: ChatRepository,
    private val modelRepository: ModelRepository,
    private val inferenceRepository: InferenceRepository,
    private val settingsRepository: SettingsRepository,
) : ViewModel() {

    val messages: StateFlow<List<ChatMessage>> =
        chatRepository.getMessages(conversationId)
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    val readyModels: StateFlow<List<ModelSummary>> =
        modelRepository.getReadyModels()
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _uiState = MutableStateFlow(ChatUiState())
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val conversation = chatRepository.getConversation(conversationId)
            _uiState.value = _uiState.value.copy(conversation = conversation)
        }
    }

    fun sendMessage(content: String) {
        if (content.isBlank() || _uiState.value.isGenerating) return

        viewModelScope.launch {
            // Check if conversation is empty to inject the system prompt
            val existingMessages = chatRepository.getMessages(conversationId).first()
            if (existingMessages.isEmpty()) {
                // Get the latest system prompt preference
                val promptToInject = settingsRepository.systemPromptFlow.first()
                chatRepository.addMessage(conversationId, MessageRole.SYSTEM, promptToInject)
            }

            // 1. Persist user message
            val persistedUserMessage = chatRepository.addMessage(conversationId, MessageRole.USER, content.trim())

            // 2. Get conversation info and current model
            val conversation = chatRepository.getConversation(conversationId)
            val modelId = conversation?.modelId

            if (modelId == null) {
                _uiState.value = _uiState.value.copy(
                    error = "No model selected. Tap the model picker to select a model.",
                )
                return@launch
            }

            val model = modelRepository.getModelById(modelId)
            if (model == null || model.localPath == null) {
                _uiState.value = _uiState.value.copy(
                    error = "Selected model not available on device.",
                )
                return@launch
            }

            // 3. Ensure model is loaded
            val inferState = inferenceRepository.state.value
            if (!inferState.isModelLoaded || inferState.loadedModelPath != model.localPath) {
                _uiState.value = _uiState.value.copy(
                    isGenerating = true,
                    draftAssistantContent = "",
                    error = null,
                    modelLoadStatus = "Loading model...",
                )
                // Get latest backend target from user settings
                val targetBackend = settingsRepository.backendTargetFlow.first()

                val loadResult = inferenceRepository.loadModel(
                    localPath = model.localPath,
                    backendTarget = targetBackend,
                )
                if (loadResult.isFailure) {
                    _uiState.value = _uiState.value.copy(
                        isGenerating = false,
                        error = "Failed to load model: ${loadResult.exceptionOrNull()?.message}",
                        modelLoadStatus = "Load failed",
                    )
                    return@launch
                }
                _uiState.value = _uiState.value.copy(modelLoadStatus = "Model loaded")
            }

            // 4. Build prompt from conversation history
            val currentDbMessages = chatRepository.getConversation(conversationId)
                ?.let { messages.value } ?: emptyList()
            val allMessages = currentDbMessages.toMutableList()
            
            // Defend against race condition: if StateFlow hasn't collected the new message yet, append it manually
            if (allMessages.none { it.id == persistedUserMessage.id }) {
                allMessages.add(persistedUserMessage)
            }
            
            val modelFamily = PromptTemplates.detectModelFamily(model.filename)
            val prompt = PromptTemplates.buildPrompt(allMessages, modelFamily)

            // 5. Generate
            _uiState.value = _uiState.value.copy(
                isGenerating = true,
                draftAssistantContent = "",
                error = null,
            )

            inferenceRepository.generateStreaming(
                prompt = prompt,
                maxTokens = 512,
                onToken = { token ->
                    _uiState.value = _uiState.value.copy(
                        draftAssistantContent = _uiState.value.draftAssistantContent + token,
                    )
                },
                onComplete = { summary ->
                    viewModelScope.launch {
                        val finalContent = _uiState.value.draftAssistantContent
                        if (finalContent.isNotBlank()) {
                            chatRepository.addMessage(
                                conversationId,
                                MessageRole.ASSISTANT,
                                finalContent,
                            )
                        }
                        _uiState.value = _uiState.value.copy(
                            isGenerating = false,
                            draftAssistantContent = "",
                        )
                    }
                },
                onError = { errorMsg ->
                    viewModelScope.launch {
                        val partial = _uiState.value.draftAssistantContent
                        if (partial.isNotBlank()) {
                            chatRepository.addMessage(
                                conversationId,
                                MessageRole.ASSISTANT,
                                partial,
                            )
                        }
                        _uiState.value = _uiState.value.copy(
                            isGenerating = false,
                            draftAssistantContent = "",
                            error = errorMsg,
                        )
                    }
                },
            )
        }
    }

    fun stopGeneration() {
        inferenceRepository.stopGeneration()
    }

    fun selectModel(modelId: String) {
        viewModelScope.launch {
            chatRepository.updateConversationModel(conversationId, modelId)
            val conversation = chatRepository.getConversation(conversationId)
            _uiState.value = _uiState.value.copy(conversation = conversation)
        }
    }

    fun updateTitle(newTitle: String) {
        viewModelScope.launch {
            chatRepository.updateConversationTitle(conversationId, newTitle)
            val conversation = chatRepository.getConversation(conversationId)
            _uiState.value = _uiState.value.copy(conversation = conversation)
        }
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(error = null)
    }

    class Factory(
        private val conversationId: String,
        private val chatRepository: ChatRepository,
        private val modelRepository: ModelRepository,
        private val inferenceRepository: InferenceRepository,
        private val settingsRepository: SettingsRepository,
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            ChatViewModel(
                conversationId,
                chatRepository,
                modelRepository,
                inferenceRepository,
                settingsRepository,
            ) as T
    }
}
