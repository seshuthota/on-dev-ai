package ai.ondev.snapdragonlab.data.repository

import ai.ondev.snapdragonlab.NativeBridge
import ai.ondev.snapdragonlab.domain.model.BackendTarget
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

data class InferenceState(
    val isRuntimeInitialized: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isGenerating: Boolean = false,
    val loadedModelPath: String? = null,
    val loadedBackend: BackendTarget = BackendTarget.CPU,
    val runtimeInfo: String = "",
)

class InferenceRepository {
    private val _state = MutableStateFlow(InferenceState())
    val state: StateFlow<InferenceState> = _state.asStateFlow()

    private val busy = AtomicBoolean(false)

    suspend fun initializeRuntime(nativeLibDir: String): String = withContext(Dispatchers.IO) {
        val result = NativeBridge.nativeInitialize(nativeLibDir)
        _state.value = _state.value.copy(
            isRuntimeInitialized = true,
            runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
        )
        result
    }

    suspend fun loadModel(
        localPath: String,
        backendTarget: BackendTarget,
        contextSize: Int = 2048,
        threadCount: Int = 0,
    ): Result<String> = withContext(Dispatchers.IO) {
        if (!busy.compareAndSet(false, true)) {
            return@withContext Result.failure(IllegalStateException("Runtime is busy"))
        }
        try {
            val status = NativeBridge.nativeLoadModel(
                modelPath = localPath,
                contextSize = contextSize,
                threadCount = threadCount,
                backendTarget = backendTarget.key,
            )
            val loaded = status.startsWith("Model loaded") ||
                status.startsWith("QNN control model loaded")
            _state.value = _state.value.copy(
                isModelLoaded = loaded,
                loadedModelPath = if (loaded) localPath else _state.value.loadedModelPath,
                loadedBackend = if (loaded) backendTarget else _state.value.loadedBackend,
                runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
            )
            if (loaded) Result.success(status) else Result.failure(RuntimeException(status))
        } finally {
            busy.set(false)
        }
    }

    suspend fun generateStreaming(
        prompt: String,
        maxTokens: Int,
        onToken: (String) -> Unit,
        onComplete: (String) -> Unit,
        onError: (String) -> Unit,
    ) = withContext(Dispatchers.IO) {
        if (!busy.compareAndSet(false, true)) {
            onError("Runtime is busy with another operation")
            return@withContext
        }
        _state.value = _state.value.copy(isGenerating = true)

        NativeBridge.setStreamingListener(
            object : NativeBridge.StreamingListener {
                override fun onToken(token: String) {
                    onToken(token)
                }

                override fun onComplete(summary: String) {
                    _state.value = _state.value.copy(
                        isGenerating = false,
                        runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                    )
                    busy.set(false)
                    onComplete(summary)
                }

                override fun onError(message: String) {
                    _state.value = _state.value.copy(
                        isGenerating = false,
                        runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                    )
                    busy.set(false)
                    onError(message)
                }
            },
        )

        try {
            NativeBridge.nativeGenerateStream(prompt, maxTokens)
        } catch (e: Exception) {
            _state.value = _state.value.copy(
                isGenerating = false,
                runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
            )
            busy.set(false)
            onError(e.message ?: "Runtime failed during generation")
        }
    }

    fun stopGeneration() {
        NativeBridge.nativeStopGeneration()
    }

    fun getRuntimeInfo(): String = NativeBridge.nativeGetRuntimeInfo()

    fun isBackendAvailable(target: BackendTarget): Boolean {
        val info = _state.value.runtimeInfo
        return when (target) {
            BackendTarget.CPU -> true
            BackendTarget.OPENCL -> info.contains("OpenCL")
            BackendTarget.VULKAN -> info.contains("Vulkan")
            BackendTarget.QNN -> info.contains("qnn_status=runtime_packaged")
        }
    }
}
