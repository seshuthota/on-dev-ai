package ai.ondev.snapdragonlab.ui.developer

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import ai.ondev.snapdragonlab.NativeBridge
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

data class DeveloperUiState(
    val benchmarkStatus: String = "Benchmarks idle",
    val latestBenchmark: String = "No benchmark run yet",
    val benchmarkHistory: String = "No benchmark history yet",
    val runtimeInfo: String = "",
    val isBusy: Boolean = false,
    val isModelLoaded: Boolean = false,
)

class DeveloperViewModel(
    private val benchmarkHistoryPath: String,
) : ViewModel() {

    private val _uiState = MutableStateFlow(DeveloperUiState())
    val uiState: StateFlow<DeveloperUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                benchmarkHistory = loadBenchmarkHistory(),
                runtimeInfo = try { NativeBridge.nativeGetRuntimeInfo() } catch (_: Exception) { "" },
            )
        }
    }

    fun runBenchmark(mode: String) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = _uiState.value.copy(
                isBusy = true,
                benchmarkStatus = "Running ${mode.uppercase()} benchmark...",
            )
            try {
                val rawResult = NativeBridge.nativeRunBenchmark(mode)
                appendBenchmarkHistory(rawResult)
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    benchmarkStatus = "${mode.uppercase()} benchmark completed",
                    latestBenchmark = formatBenchmark(rawResult),
                    benchmarkHistory = loadBenchmarkHistory(),
                    runtimeInfo = NativeBridge.nativeGetRuntimeInfo(),
                )
            } catch (e: Exception) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    benchmarkStatus = "Benchmark failed: ${e.message}",
                )
            }
        }
    }

    private fun appendBenchmarkHistory(rawJson: String) {
        val historyFile = java.io.File(benchmarkHistoryPath)
        historyFile.parentFile?.mkdirs()
        historyFile.appendText(rawJson + "\n")
    }

    private fun loadBenchmarkHistory(): String {
        val historyFile = java.io.File(benchmarkHistoryPath)
        if (!historyFile.exists()) return "No benchmark history yet"
        val lines = historyFile.readLines().filter { it.isNotBlank() }.takeLast(3)
        if (lines.isEmpty()) return "No benchmark history yet"
        return lines.joinToString("\n\n") { formatBenchmark(it) }
    }

    private fun formatBenchmark(rawJson: String): String {
        return try {
            val benchmark = JSONObject(rawJson)
            val entries = benchmark.optJSONArray("entries") ?: JSONArray()
            val header = buildString {
                append("mode="); append(benchmark.optString("mode"))
                append(" | target="); append(benchmark.optString("backend_target"))
                append(" | loaded="); append(benchmark.optString("backend_effective"))
                append(" | threads="); append(benchmark.optInt("threads"))
            }
            val body = buildString {
                for (i in 0 until entries.length()) {
                    val entry = entries.getJSONObject(i)
                    if (i > 0) append('\n')
                    append(entry.optString("name"))
                    append(": decoded="); append(entry.optInt("decoded_tokens"))
                    append(", tok/s="); append("%.1f".format(entry.optDouble("tok_per_sec")))
                    append(", ttft="); append(entry.optLong("ttft_ms")); append("ms")
                }
            }
            if (body.isBlank()) header else "$header\n$body"
        } catch (_: Exception) {
            rawJson.take(200)
        }
    }

    class Factory(private val benchmarkHistoryPath: String) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T =
            DeveloperViewModel(benchmarkHistoryPath) as T
    }
}
