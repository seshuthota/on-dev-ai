package ai.ondev.snapdragonlab

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.workDataOf

class DebugReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val action = intent.action ?: return
        val backend = intent.getStringExtra("backend").orEmpty()
        val benchmarkMode = intent.getStringExtra("benchmark_mode").orEmpty()
        val mode = intent.getStringExtra("mode").orEmpty()
        val modelName = intent.getStringExtra("model_name").orEmpty()
        val modelPath = intent.getStringExtra("model_path").orEmpty()
        val workData = workDataOf(
            DebugBenchmarkWorker.KEY_ACTION to action,
            DebugBenchmarkWorker.KEY_BACKEND to backend,
            DebugBenchmarkWorker.KEY_BENCHMARK_MODE to benchmarkMode,
            DebugBenchmarkWorker.KEY_MODE to mode,
            DebugBenchmarkWorker.KEY_GPU_LAYERS to intent.getIntExtra("gpu_layers", -1),
            DebugBenchmarkWorker.KEY_MODEL_NAME to modelName,
            DebugBenchmarkWorker.KEY_MODEL_PATH to modelPath,
        )
        val request = OneTimeWorkRequestBuilder<DebugBenchmarkWorker>()
            .setInputData(workData)
            .build()

        WorkManager.getInstance(context.applicationContext)
            .enqueueUniqueWork(
                DebugBenchmarkWorker.UNIQUE_WORK_NAME,
                ExistingWorkPolicy.REPLACE,
                request,
            )

        when (action) {
            DebugBenchmarkWorker.ACTION_OPENCL_SMOKE -> {
                val backendTarget = backend.lowercase().ifBlank { "opencl" }
                val normalizedMode = benchmarkMode.ifBlank { "smoke" }
                Log.i(
                    "OnDevAI",
                    "DebugReceiver: queued backend smoke backend=$backendTarget mode=$normalizedMode model_name=${modelName.ifBlank { "<default>" }} model_path=${modelPath.ifBlank { "<none>" }}",
                )
            }
            DebugBenchmarkWorker.ACTION_PROBE_OPENCL -> {
                Log.i("OnDevAI", "DebugReceiver: queued OpenCL probe")
            }
            DebugBenchmarkWorker.ACTION_QNN_CONTROL_SMOKE -> {
                val backendTarget = backend.lowercase().ifBlank { "htp-v79" }
                Log.i("OnDevAI", "DebugReceiver: queued qnn control smoke backend=$backendTarget")
            }
            DebugBenchmarkWorker.ACTION_QNN_ENGINE_SMOKE -> {
                Log.i("OnDevAI", "DebugReceiver: queued qnn engine smoke")
            }
            else -> {
                val normalizedMode = mode.ifBlank { "smoke" }
                Log.i("OnDevAI", "DebugReceiver: queued $normalizedMode benchmark")
            }
        }
    }
}
