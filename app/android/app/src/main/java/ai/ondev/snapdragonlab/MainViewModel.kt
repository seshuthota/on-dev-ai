package ai.ondev.snapdragonlab

import ai.ondev.snapdragonlab.BuildConfig
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

data class MainUiState(
    val nativeStatus: String = "Runtime not initialized",
    val backendTarget: String = "cpu",
    val backendLabel: String = "CPU",
    val loadedBackendLabel: String = "Not loaded",
    val modelLabel: String = "Primary benchmark model",
    val modelPath: String = "",
    val appFilesDir: String = "",
    val internalModelPath: String = "",
    val benchmarkHistoryPath: String = "",
    val pathDiagnostics: String = "",
    val prompt: String = "Write two short sentences about why low TTFT matters for on-device chat.",
    val output: String = "",
    val generationSummary: String = "",
    val loadStatus: String = "Model not loaded",
    val benchmarkStatus: String = "Benchmarks idle",
    val latestBenchmark: String = "No benchmark run yet",
    val benchmarkHistory: String = "No benchmark history yet",
    val runtimeInfo: String = "",
    val isQnnBuildEnabled: Boolean = BuildConfig.ONDEVAI_ENABLE_QNN,
    val isRuntimeInitialized: Boolean = false,
    val isModelLoaded: Boolean = false,
    val isBusy: Boolean = false,
)

class MainViewModel : ViewModel() {
    private val _uiState = MutableStateFlow(MainUiState())
    val uiState: StateFlow<MainUiState> = _uiState.asStateFlow()
    private var initialized = false

    init {
        NativeBridge.setStreamingListener(
            object : NativeBridge.StreamingListener {
                override fun onToken(token: String) {
                    _uiState.value = _uiState.value.copy(
                        output = _uiState.value.output + token,
                    )
                }

                override fun onComplete(summary: String) {
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        generationSummary = summary,
                        runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                    )
                }

                override fun onError(message: String) {
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        generationSummary = message,
                        runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                    )
                }
            },
        )
    }

    fun initializeRuntime(
        nativeLibDir: String,
        appFilesDir: String,
        modelPath: String,
        internalModelPath: String,
        benchmarkHistoryPath: String,
        pathDiagnostics: String,
    ) {
        if (initialized) {
            return
        }
        initialized = true

        _uiState.value = _uiState.value.copy(
            appFilesDir = appFilesDir,
            modelPath = modelPath,
            internalModelPath = internalModelPath,
            benchmarkHistoryPath = benchmarkHistoryPath,
            pathDiagnostics = pathDiagnostics,
        )

        viewModelScope.launch(Dispatchers.IO) {
            val initStatus = NativeBridge.nativeInitialize(nativeLibDir)
            val preparedModelPath = prepareModelPathForBackend(
                backendTarget = _uiState.value.backendTarget,
                sourceModelPath = modelPath,
                internalModelPath = internalModelPath,
                appFilesDir = appFilesDir,
            )
            val modelStatus = NativeBridge.nativeLoadModel(
                modelPath = preparedModelPath,
                contextSize = 2048,
                threadCount = 0, // auto-resolve via native code (uses cpu_count - 2)
                backendTarget = _uiState.value.backendTarget,
            )
            val runtimeInfo = NativeBridge.nativeGetRuntimeInfo()

            _uiState.value = _uiState.value.copy(
                nativeStatus = initStatus,
                runtimeInfo = runtimeInfo,
                loadStatus = modelStatus,
                modelPath = preparedModelPath,
                benchmarkHistory = loadBenchmarkHistory(benchmarkHistoryPath),
                isRuntimeInitialized = true,
                isModelLoaded = isSuccessfulLoadStatus(modelStatus),
                loadedBackendLabel = if (isSuccessfulLoadStatus(modelStatus)) {
                    backendDisplayName(_uiState.value.backendTarget)
                } else {
                    "Not loaded"
                },
            )
        }
    }

    fun selectBackendTarget(target: String) {
        val normalized = normalizeBackendTarget(target)
        val current = _uiState.value
        val reloadRequired = current.isModelLoaded && current.backendTarget != normalized

        _uiState.value = current.copy(
            backendTarget = normalized,
            backendLabel = backendDisplayName(normalized),
            loadedBackendLabel = if (reloadRequired) "Reload required" else current.loadedBackendLabel,
            loadStatus = if (reloadRequired) {
                "Reload model to apply backend target: ${backendDisplayName(normalized)}"
            } else {
                current.loadStatus
            },
            isModelLoaded = current.isModelLoaded && !reloadRequired,
        )
    }

    fun updatePrompt(prompt: String) {
        _uiState.value = _uiState.value.copy(prompt = prompt)
    }

    fun loadModel() {
        val current = _uiState.value
        if (current.modelPath.isBlank()) {
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(isBusy = true, loadStatus = "Loading model...")

            val file = File(current.modelPath)
            val diagnostics = buildString {
                append("java_exists=")
                append(file.exists())
                append(" | java_canRead=")
                append(file.canRead())
                append(" | path=")
                append(current.modelPath)
            }

            val preparedModelPath = prepareModelPathForBackend(
                backendTarget = current.backendTarget,
                sourceModelPath = current.modelPath,
                internalModelPath = current.internalModelPath,
                appFilesDir = current.appFilesDir,
            )
            val modelStatus = NativeBridge.nativeLoadModel(
                modelPath = preparedModelPath,
                contextSize = 2048,
                threadCount = 0, // auto-resolve via native code (uses cpu_count - 2)
                backendTarget = current.backendTarget,
            )

            _uiState.value = _uiState.value.copy(
                isBusy = false,
                loadStatus = "$modelStatus | $diagnostics",
                modelPath = preparedModelPath,
                isModelLoaded = isSuccessfulLoadStatus(modelStatus),
                loadedBackendLabel = if (isSuccessfulLoadStatus(modelStatus)) {
                    backendDisplayName(current.backendTarget)
                } else {
                    "Not loaded"
                },
                runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
            )
        }
    }

    fun generate() {
        val current = _uiState.value
        if (!current.isModelLoaded || current.prompt.isBlank()) {
            return
        }

        if (current.backendTarget == "qnn") {
            _uiState.value = _uiState.value.copy(
                generationSummary = "Text generation is not implemented for the QNN control-model path",
            )
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                output = "",
                generationSummary = "Generating on ${current.loadedBackendLabel}...",
            )

            NativeBridge.nativeGenerateStream(
                prompt = current.prompt,
                maxTokens = 96,
            )
        }
    }

    fun stopGeneration() {
        NativeBridge.nativeStopGeneration()
        _uiState.value = _uiState.value.copy(
            generationSummary = "Stop requested...",
        )
    }

    fun runBenchmark(mode: String) {
        val current = _uiState.value
        if (!current.isModelLoaded || current.benchmarkHistoryPath.isBlank()) {
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                benchmarkStatus = "Running ${mode.uppercase()} benchmark...",
            )

            try {
                val rawResult = NativeBridge.nativeRunBenchmark(mode)
                appendBenchmarkHistory(current.benchmarkHistoryPath, rawResult)

                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    benchmarkStatus = "${mode.uppercase()} benchmark completed",
                    latestBenchmark = formatBenchmark(rawResult),
                    benchmarkHistory = loadBenchmarkHistory(current.benchmarkHistoryPath),
                    runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                )
            } catch (exception: Exception) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    benchmarkStatus = "Benchmark failed: ${exception.message}",
                )
            }
        }
    }

    override fun onCleared() {
        NativeBridge.setStreamingListener(null)
        super.onCleared()
    }

    private fun prepareModelPathForBackend(
        backendTarget: String,
        sourceModelPath: String,
        internalModelPath: String,
        appFilesDir: String,
    ): String {
        if (normalizeBackendTarget(backendTarget) == "qnn") {
            return appFilesDir
        }

        return prepareModelForNativeLoad(sourceModelPath, internalModelPath)
    }

    private fun prepareModelForNativeLoad(
        sourceModelPath: String,
        internalModelPath: String,
    ): String {
        val sourceFile = File(sourceModelPath)
        val internalFile = File(internalModelPath)

        if (sourceFile.absolutePath == internalFile.absolutePath) {
            return internalFile.absolutePath
        }

        if (!sourceFile.exists() || !sourceFile.canRead()) {
            return sourceFile.absolutePath
        }

        internalFile.parentFile?.mkdirs()

        val shouldCopy = !internalFile.exists() || internalFile.length() != sourceFile.length()
        if (shouldCopy) {
            sourceFile.inputStream().use { input ->
                internalFile.outputStream().use { output ->
                    input.copyTo(output, DEFAULT_BUFFER_SIZE)
                }
            }
        }

        return internalFile.absolutePath
    }

    private fun appendBenchmarkHistory(
        historyPath: String,
        rawJson: String,
    ) {
        val historyFile = File(historyPath)
        historyFile.parentFile?.mkdirs()
        historyFile.appendText(rawJson + "\n")
    }

    private fun loadBenchmarkHistory(historyPath: String): String {
        val historyFile = File(historyPath)
        if (!historyFile.exists()) {
            return "No benchmark history yet"
        }

        val lines = historyFile.readLines().filter { it.isNotBlank() }.takeLast(3)
        if (lines.isEmpty()) {
            return "No benchmark history yet"
        }

        return lines.joinToString("\n\n") { raw -> formatBenchmark(raw) }
    }

    private fun formatBenchmark(rawJson: String): String {
        val benchmark = JSONObject(rawJson)
        val entries = benchmark.optJSONArray("entries") ?: JSONArray()
        val header = buildString {
            append("mode=")
            append(benchmark.optString("mode"))
            append(" | target=")
            append(benchmark.optString("backend_target"))
            append(" | loaded=")
            append(benchmark.optString("backend_effective"))
            append(" | model=")
            append(benchmark.optString("model_label"))
            append(" | threads=")
            append(benchmark.optInt("threads"))
            append(" | ctx=")
            append(benchmark.optInt("context_size"))
        }

        val body = buildString {
            for (index in 0 until entries.length()) {
                val entry = entries.getJSONObject(index)
                if (index > 0) {
                    append('\n')
                }
                append(entry.optString("name"))
                append(": prompt_chars=")
                append(entry.optInt("prompt_chars"))
                append(", decoded=")
                append(entry.optInt("decoded_tokens"))
                append(", elapsed_ms=")
                append(entry.optLong("elapsed_ms"))
                append(", ttft_ms=")
                append(entry.optLong("ttft_ms"))
                append(", tok_per_sec=")
                append(entry.optDouble("tok_per_sec"))
            }
        }

        return if (body.isBlank()) header else "$header\n$body"
    }

    private fun normalizeBackendTarget(target: String): String =
        when {
            target.equals("opencl", ignoreCase = true) -> "opencl"
            target.equals("vulkan", ignoreCase = true) -> "vulkan"
            BuildConfig.ONDEVAI_ENABLE_QNN && target.equals("qnn", ignoreCase = true) -> "qnn"
            else -> "cpu"
        }

    private fun backendDisplayName(target: String): String =
        when (normalizeBackendTarget(target)) {
            "opencl" -> "OpenCL"
            "vulkan" -> "Vulkan"
            "qnn" -> "QNN"
            else -> "CPU"
        }

    private fun isSuccessfulLoadStatus(status: String): Boolean =
        status.startsWith("Model loaded") || status.startsWith("QNN control model loaded")
}
