package ai.ondev.snapdragonlab

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import ai.ondev.snapdragonlab.ui.SnapdragonLabApp
import ai.ondev.snapdragonlab.ui.theme.OnDevAITheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val app = application as OnDevAIApplication
        val container = app.container

        // Initialize native runtime in background
        CoroutineScope(Dispatchers.IO).launch {
            try {
                if (BuildConfig.ONDEVAI_ENABLE_QNN) {
                    Log.i("OnDevAI", "QNN prep: ${QnnControlRunner.prepareBackendEnvironment(this@MainActivity, "htp-v79")}")
                }
                val initResult = container.inferenceRepository.initializeRuntime(
                    applicationInfo.nativeLibraryDir,
                )
                Log.i("OnDevAI", "Runtime init: $initResult")
                Log.i("OnDevAI", "RuntimeInfo: ${container.inferenceRepository.getRuntimeInfo()}")
            } catch (e: Exception) {
                Log.e("OnDevAI", "Runtime init failed", e)
            }
        }

        val benchmarkHistoryPath = File(filesDir, "benchmarks/history.jsonl").absolutePath

        setContent {
            OnDevAITheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    SnapdragonLabApp(
                        container = container,
                        benchmarkHistoryPath = benchmarkHistoryPath,
                    )
                }
            }
        }
    }
}
