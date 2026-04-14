package ai.ondev.snapdragonlab.ui.conversations

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import ai.ondev.snapdragonlab.data.repository.ChatRepository
import ai.ondev.snapdragonlab.domain.model.ConversationSummary
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class ConversationsViewModel(
    private val chatRepository: ChatRepository,
) : ViewModel() {

    val conversations: StateFlow<List<ConversationSummary>> =
        chatRepository.getConversations()
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())

    private val _navigateToChat = MutableStateFlow<String?>(null)
    val navigateToChat: StateFlow<String?> = _navigateToChat

    fun createConversation() {
        viewModelScope.launch {
            val id = chatRepository.createConversation("New conversation")
            _navigateToChat.value = id
        }
    }

    fun deleteConversation(id: String) {
        viewModelScope.launch {
            chatRepository.deleteConversation(id)
        }
    }

    fun renameConversation(id: String, newTitle: String) {
        viewModelScope.launch {
            chatRepository.updateConversationTitle(id, newTitle)
        }
    }

    fun onNavigatedToChat() {
        _navigateToChat.value = null
    }

    class Factory(private val chatRepository: ChatRepository) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            ConversationsViewModel(chatRepository) as T
    }
}
