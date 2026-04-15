package ai.ondev.snapdragonlab.ui.chat

import ai.ondev.snapdragonlab.domain.model.ChatMessage
import ai.ondev.snapdragonlab.domain.model.MessageRole

object PromptTemplates {

    fun buildPrompt(messages: List<ChatMessage>, modelFamily: ModelFamily): String =
        when (modelFamily) {
            ModelFamily.LLAMA -> buildLlamaPrompt(messages)
            ModelFamily.QWEN -> buildQwenPrompt(messages)
            ModelFamily.GEMMA -> buildGemmaPrompt(messages)
            ModelFamily.DEFAULT -> buildChatMlPrompt(messages)
        }

    fun detectModelFamily(filename: String): ModelFamily {
        val lower = filename.lowercase()
        return when {
            "llama" in lower || "meta" in lower -> ModelFamily.LLAMA
            "qwen" in lower -> ModelFamily.QWEN
            "gemma" in lower -> ModelFamily.GEMMA
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

    private fun buildGemmaPrompt(messages: List<ChatMessage>): String = buildString {
        append("<bos>")
        var pendingSystem = mutableListOf<String>()
        var firstUserSeen = false

        for (message in messages) {
            when (message.role) {
                MessageRole.SYSTEM -> {
                    if (message.content.isNotBlank()) {
                        pendingSystem.add(message.content.trim())
                    }
                }
                MessageRole.USER -> {
                    firstUserSeen = true
                    if (pendingSystem.isNotEmpty()) {
                        append("<start_of_turn>user\n")
                        append("System instruction:\n")
                        append(pendingSystem.joinToString("\n\n"))
                        append("\n\n")
                        append(message.content)
                        append("<end_of_turn>\n")
                        pendingSystem.clear()
                    } else {
                        append("<start_of_turn>user\n")
                        append(message.content)
                        append("<end_of_turn>\n")
                    }
                }
                MessageRole.ASSISTANT -> {
                    append("<start_of_turn>model\n")
                    append(message.content)
                    append("<end_of_turn>\n")
                }
            }
        }

        // If system instruction exists without a following user turn, emit it now.
        // But for Gemma, system instructions should precede user turns, so this case
        // only occurs if the conversation ends without a user turn after system msg.
        if (pendingSystem.isNotEmpty() && !firstUserSeen) {
            append("<start_of_turn>user\n")
            append("System instruction:\n")
            append(pendingSystem.joinToString("\n\n"))
            append("<end_of_turn>\n")
        }

        append("<start_of_turn>model\n")
    }

    enum class ModelFamily {
        LLAMA,
        QWEN,
        GEMMA,
        DEFAULT,
    }
}
