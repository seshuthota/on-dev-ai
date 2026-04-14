package ai.ondev.snapdragonlab.ui.chat

import ai.ondev.snapdragonlab.domain.model.ChatMessage
import ai.ondev.snapdragonlab.domain.model.MessageRole

object PromptTemplates {

    fun buildPrompt(messages: List<ChatMessage>, modelFamily: ModelFamily): String =
        when (modelFamily) {
            ModelFamily.LLAMA -> buildLlamaPrompt(messages)
            ModelFamily.QWEN -> buildQwenPrompt(messages)
            ModelFamily.DEFAULT -> buildChatMlPrompt(messages)
        }

    fun detectModelFamily(filename: String): ModelFamily {
        val lower = filename.lowercase()
        return when {
            "llama" in lower || "meta" in lower -> ModelFamily.LLAMA
            "qwen" in lower -> ModelFamily.QWEN
            else -> ModelFamily.DEFAULT
        }
    }

    private fun buildLlamaPrompt(messages: List<ChatMessage>): String = buildString {
        append("<|begin_of_text|>")
        for (message in messages) {
            val role = when (message.role) {
                MessageRole.SYSTEM -> "system"
                MessageRole.USER -> "user"
                MessageRole.ASSISTANT -> "assistant"
            }
            append("<|start_header_id|>")
            append(role)
            append("<|end_header_id|>\n\n")
            append(message.content)
            append("<|eot_id|>")
        }
        append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    }

    private fun buildQwenPrompt(messages: List<ChatMessage>): String = buildString {
        for (message in messages) {
            val role = when (message.role) {
                MessageRole.SYSTEM -> "system"
                MessageRole.USER -> "user"
                MessageRole.ASSISTANT -> "assistant"
            }
            append("<|im_start|>")
            append(role)
            append('\n')
            append(message.content)
            append("<|im_end|>\n")
        }
        append("<|im_start|>assistant\n")
    }

    private fun buildChatMlPrompt(messages: List<ChatMessage>): String = buildString {
        for (message in messages) {
            val role = when (message.role) {
                MessageRole.SYSTEM -> "system"
                MessageRole.USER -> "user"
                MessageRole.ASSISTANT -> "assistant"
            }
            append("<|im_start|>")
            append(role)
            append('\n')
            append(message.content)
            append("<|im_end|>\n")
        }
        append("<|im_start|>assistant\n")
    }

    enum class ModelFamily {
        LLAMA,
        QWEN,
        DEFAULT,
    }
}
