package ai.ondev.snapdragonlab

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import java.io.File

class DebugBenchmarkWorker(
    appContext: Context,
    workerParams: WorkerParameters,
) : CoroutineWorker(appContext, workerParams) {
    override suspend fun doWork(): Result {
        val action = inputData.getString(KEY_ACTION) ?: return Result.failure()

        return try {
            when (action) {
                ACTION_PROBE_OPENCL -> runProbeOpenCl()
                ACTION_OPENCL_SMOKE -> runBackendSmoke()
                ACTION_QNN_CONTROL_SMOKE -> runQnnControlSmoke()
                ACTION_QNN_ENGINE_SMOKE -> runQnnEngineSmoke()
                else -> runBenchmark()
            }
            Result.success()
        } catch (exception: Exception) {
            Log.e(TAG, "DebugReceiver: failure action=$action", exception)
            Result.failure()
        }
    }

    private fun runProbeOpenCl() {
        Log.i(TAG, "DebugReceiver: starting OpenCL probe")
        Log.i(TAG, "DebugReceiver: ${NativeBridge.preloadLibrary()}")
        val result = NativeBridge.nativeProbeOpenCl()
        Log.i(TAG, "DebugReceiver: OpenCL probe = $result")
    }

    private fun runBackendSmoke() {
        val backendTarget = inputData.getString(KEY_BACKEND).orEmpty().lowercase().ifBlank { "opencl" }
        val benchmarkMode = inputData.getString(KEY_BENCHMARK_MODE).orEmpty().ifBlank { "smoke" }
        val gpuLayers = inputData.getInt(KEY_GPU_LAYERS, -1)
        val modelPath = resolveModelPath()

        Log.i(
            TAG,
            "DebugReceiver: starting backend smoke backend=$backendTarget mode=$benchmarkMode model_path=$modelPath",
        )
        Log.i(TAG, "DebugReceiver: ${NativeBridge.preloadLibrary()}")
        Log.i(TAG, "DebugReceiver: ${NativeBridge.nativeSetOpenClGpuLayers(gpuLayers)}")

        if (!File(modelPath).canRead()) {
            Log.e(TAG, "DebugReceiver: model is not readable model_path=$modelPath")
            return
        }

        val initStatus = NativeBridge.nativeInitialize(applicationContext.applicationInfo.nativeLibraryDir)
        val loadStatus = NativeBridge.nativeLoadModel(
            modelPath = modelPath,
            contextSize = 2048,
            threadCount = 0,
            backendTarget = backendTarget,
        )
        val runtimeInfo = NativeBridge.nativeGetRuntimeInfo()

        Log.i(TAG, "DebugReceiver: backend init = $initStatus")
        Log.i(TAG, "DebugReceiver: backend load = $loadStatus")
        Log.i(TAG, "DebugReceiver: backend runtime = $runtimeInfo")

        if (!loadStatus.startsWith("Model loaded")) {
            return
        }

        val rawResult = NativeBridge.nativeRunBenchmark(benchmarkMode)
        appendBenchmarkHistory(rawResult)
        Log.i(TAG, "DebugReceiver: backend benchmark mode=$benchmarkMode result = $rawResult")
    }

    private fun runQnnControlSmoke() {
        val backendTarget = inputData.getString(KEY_BACKEND).orEmpty().lowercase().ifBlank { "htp-v79" }
        Log.i(TAG, "DebugReceiver: starting qnn control smoke backend=$backendTarget")
        val result = QnnControlRunner.runControlSmoke(applicationContext, backendTarget)
        Log.i(TAG, "DebugReceiver: qnn control smoke result = $result")
    }

    private fun runQnnEngineSmoke() {
        Log.i(TAG, "DebugReceiver: starting qnn engine smoke")
        val prep = QnnControlRunner.prepareBackendEnvironment(applicationContext, "htp-v79")
        Log.i(TAG, "DebugReceiver: qnn engine prep = $prep")

        val initStatus = NativeBridge.nativeInitialize(applicationContext.applicationInfo.nativeLibraryDir)
        val loadStatus = NativeBridge.nativeLoadModel(
            modelPath = applicationContext.filesDir.absolutePath,
            contextSize = 2048,
            threadCount = 0,
            backendTarget = "qnn",
        )
        val runtimeInfo = NativeBridge.nativeGetRuntimeInfo()
        val benchmark = NativeBridge.nativeRunBenchmark("smoke")

        Log.i(TAG, "DebugReceiver: qnn engine init = $initStatus")
        Log.i(TAG, "DebugReceiver: qnn engine load = $loadStatus")
        Log.i(TAG, "DebugReceiver: qnn engine runtime = $runtimeInfo")
        Log.i(TAG, "DebugReceiver: qnn engine benchmark = $benchmark")
    }

    private fun runBenchmark() {
        val mode = inputData.getString(KEY_MODE).orEmpty().ifBlank { "smoke" }
        Log.i(TAG, "DebugReceiver: running $mode benchmark")
        Log.i(TAG, "DebugReceiver: ${NativeBridge.preloadLibrary()}")

        val rawResult = NativeBridge.nativeRunBenchmark(mode)
        appendBenchmarkHistory(rawResult)
        Log.i(TAG, "DebugReceiver: $mode result = $rawResult")
    }

    private fun appendBenchmarkHistory(rawJson: String) {
        val historyPath = File(applicationContext.filesDir, "benchmarks/history.jsonl")
        historyPath.parentFile?.mkdirs()
        historyPath.appendText(rawJson + "\n")
    }

    private fun resolveModelPath(): String {
        val explicitPath = inputData.getString(KEY_MODEL_PATH).orEmpty()
        if (explicitPath.isNotBlank()) {
            return explicitPath
        }

        val modelName = inputData.getString(KEY_MODEL_NAME).orEmpty().ifBlank { DEFAULT_MODEL_NAME }
        return File(applicationContext.filesDir, "models/$modelName").absolutePath
    }

    companion object {
        const val UNIQUE_WORK_NAME = "ondevai_debug_benchmark_worker"
        const val KEY_ACTION = "action"
        const val KEY_BACKEND = "backend"
        const val KEY_BENCHMARK_MODE = "benchmark_mode"
        const val KEY_MODE = "mode"
        const val KEY_GPU_LAYERS = "gpu_layers"
        const val KEY_MODEL_NAME = "model_name"
        const val KEY_MODEL_PATH = "model_path"

        const val ACTION_PROBE_OPENCL = "ai.ondev.snapdragonlab.DEBUG_PROBE_OPENCL"
        const val ACTION_OPENCL_SMOKE = "ai.ondev.snapdragonlab.DEBUG_OPENCL_SMOKE"
        const val ACTION_QNN_CONTROL_SMOKE = "ai.ondev.snapdragonlab.DEBUG_QNN_CONTROL_SMOKE"
        const val ACTION_QNN_ENGINE_SMOKE = "ai.ondev.snapdragonlab.DEBUG_QNN_ENGINE_SMOKE"
        const val DEFAULT_MODEL_NAME = "primary-model.gguf"

        private const val TAG = "OnDevAI"
    }
}
