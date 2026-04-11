package ai.ondev.snapdragonlab

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ai.ondev.snapdragonlab.ui.SnapdragonLabApp
import java.io.File

class MainActivity : ComponentActivity() {
    private val viewModel: MainViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        if (BuildConfig.ONDEVAI_ENABLE_QNN) {
            Log.i("OnDevAI", "QNN prep: ${QnnControlRunner.prepareBackendEnvironment(this, "htp-v79")}")
        }

        viewModel.initializeRuntime(
            nativeLibDir = applicationInfo.nativeLibraryDir,
            appFilesDir = filesDir.absolutePath,
            modelPath = resolveModelPath(),
            internalModelPath = File(filesDir, "models/primary-model.gguf").absolutePath,
            benchmarkHistoryPath = File(filesDir, "benchmarks/history.jsonl").absolutePath,
            pathDiagnostics = buildPathDiagnostics(),
        )

        // Log runtime info to logcat for adb-based debugging
        Log.i("OnDevAI", "RuntimeInfo: ${NativeBridge.nativeGetRuntimeInfo()}")

        setContent {
            Surface(modifier = Modifier.fillMaxSize()) {
                SnapdragonLabApp(viewModel = viewModel)
            }
        }
    }

    private fun resolveModelPath(): String {
        val externalDir = getExternalFilesDir(null)
        val candidates = modelPathCandidates(externalDir)

        candidates.firstOrNull { it.exists() }?.let { return it.absolutePath }

        val baseDir = externalDir ?: filesDir
        val modelsDir = File(baseDir, "models")
        modelsDir.mkdirs()
        return File(modelsDir, "primary-model.gguf").absolutePath
    }

    private fun buildPathDiagnostics(): String {
        val externalDir = getExternalFilesDir(null)
        return buildString {
            append("externalDir=")
            append(externalDir?.absolutePath ?: "<null>")
            modelPathCandidates(externalDir).forEachIndexed { index, file ->
                append(" | [")
                append(index)
                append("] exists=")
                append(file.exists())
                append(", canRead=")
                append(file.canRead())
                append(", path=")
                append(file.absolutePath)
            }
        }
    }

    private fun modelPathCandidates(externalDir: File?): List<File> {
        val packageName = applicationContext.packageName
        return listOfNotNull(
            File(filesDir, "models/primary-model.gguf"),
            externalDir?.let { File(it, "models/primary-model.gguf") },
            File("/storage/emulated/0/Android/data/$packageName/files/models/primary-model.gguf"),
            File("/sdcard/Android/data/$packageName/files/models/primary-model.gguf"),
        )
    }
}
