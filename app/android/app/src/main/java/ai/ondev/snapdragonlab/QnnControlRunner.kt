package ai.ondev.snapdragonlab

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Runs the QNN control-model smoke test using the in-process native runner.
 *
 * Previous approach (subprocess via ProcessBuilder → qnn_smoke/qnn-net-run) was blocked by
 * Android 16 app sandbox W^X exec policy. This version calls the QNN API directly
 * via JNI through NativeBridge → qnn_runner.cpp.
 */
object QnnControlRunner {
    private const val TAG = "OnDevAI"
    private const val CONTROL_ASSET_ROOT = "qnn/control"
    private const val HEXAGON_V79_ASSET_ROOT = "qnn/hexagon-v79"

    fun prepareBackendEnvironment(
        context: Context,
        backend: String,
    ): String {
        if (!BuildConfig.ONDEVAI_ENABLE_QNN) {
            return "QNN control environment unavailable: QNN disabled at build"
        }

        val normalizedBackend = backend.lowercase().ifBlank { "htp-v79" }
        if (normalizedBackend != "cpu" && normalizedBackend != "htp-v79") {
            return "QNN control environment unavailable: unsupported backend=$normalizedBackend"
        }

        val controlDir = File(context.filesDir, "qnn/control")
        val v79SkelDir = File(context.filesDir, "qnn/hexagon-v79")
        try {
            extractControlAssets(context, controlDir)
            if (normalizedBackend == "htp-v79") {
                extractAssetDir(context, HEXAGON_V79_ASSET_ROOT, v79SkelDir)
            }
        } catch (error: Exception) {
            return "QNN control environment unavailable: asset extraction failed (${error.message ?: error::class.java.simpleName})"
        }

        val adspConfig = when (normalizedBackend) {
            "htp-v79" -> NativeBridge.configureQnnAdspLibraryPath(v79SkelDir.absolutePath, controlDir.absolutePath)
            else -> NativeBridge.configureQnnAdspLibraryPath(controlDir.absolutePath)
        }
        val preloadStatus = NativeBridge.preloadLibrary()
        return "$adspConfig | $preloadStatus"
    }

    fun runControlSmoke(
        context: Context,
        backend: String,
    ): String {
        if (!BuildConfig.ONDEVAI_ENABLE_QNN) {
            return "QNN control smoke unavailable: QNN disabled at build"
        }

        val normalizedBackend = backend.lowercase().ifBlank { "htp-v79" }
        if (normalizedBackend != "cpu" && normalizedBackend != "htp-v79") {
            return "QNN control smoke unavailable: unsupported backend=$normalizedBackend"
        }

        val controlDir = File(context.filesDir, "qnn/control")
        val prepStatus = prepareBackendEnvironment(context, normalizedBackend)
        if (prepStatus.startsWith("QNN control environment unavailable:")) {
            return prepStatus.replace("environment", "smoke")
        }

        val modelLib = when (normalizedBackend) {
            "cpu" -> File(controlDir, "libqnn_model_float.so").absolutePath
            else -> File(controlDir, "libqnn_model_8bit_quantized.so").absolutePath
        }
        val backendLib = when (normalizedBackend) {
            "cpu" -> "libQnnCpu.so"
            else -> "libQnnHtp.so"
        }
        val outputDir = File(controlDir, "output").absolutePath

        Log.i(TAG, "QnnControlRunner: in-process run backend=$normalizedBackend backendLib=$backendLib modelLib=$modelLib")

        return try {
            val result = NativeBridge.nativeRunQnnControlSmoke(
                backendLib = backendLib,
                modelLib = modelLib,
                outputDir = outputDir,
            )
            Log.i(TAG, "QnnControlRunner: result=$result")
            "backend=$normalizedBackend | $prepStatus | $result"
        } catch (error: Exception) {
            val message = "QNN control smoke failed: ${error.message ?: error::class.java.simpleName}"
            Log.e(TAG, "QnnControlRunner: $message", error)
            message
        }
    }

    private fun extractControlAssets(
        context: Context,
        destinationRoot: File,
    ) {
        destinationRoot.mkdirs()
        extractAssetDir(context, CONTROL_ASSET_ROOT, destinationRoot)
    }

    private fun extractAssetDir(
        context: Context,
        assetPath: String,
        destinationDir: File,
    ) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            destinationDir.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                destinationDir.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            return
        }

        destinationDir.mkdirs()
        for (child in children) {
            extractAssetDir(
                context = context,
                assetPath = "$assetPath/$child",
                destinationDir = File(destinationDir, child),
            )
        }
    }
}
