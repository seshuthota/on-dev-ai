package ai.ondev.snapdragonlab

import android.os.SystemClock
import android.system.Os

object NativeBridge {
    private val defaultAdspSearchPaths =
        listOf(
            "/vendor/dsp/cdsp",
            "/vendor/lib/rfsa/adsp",
            "/system/lib/rfsa/adsp",
            "/dsp",
        )

    private val qnnRuntimeLibraries =
        if (BuildConfig.ONDEVAI_ENABLE_QNN) {
            listOf(
                "QnnSystem",
                "QnnCpu",
                "QnnGpu",
                "QnnHtp",
                "QnnHtpPrepare",
                "QnnHtpV79Stub",
                "QnnHtpV81Stub",
                "Genie",
                "QnnGenAiTransformer",
            )
        } else {
            emptyList()
        }

    interface StreamingListener {
        fun onToken(token: String)
        fun onComplete(summary: String)
        fun onError(message: String)
    }

    @Volatile
    private var streamingListener: StreamingListener? = null

    @Volatile
    private var isLoaded = false

    @Volatile
    private var qnnRuntimePreloaded = false

    @Volatile
    private var qnnPreloadSummary = "QNN runtime disabled at build"

    @Volatile
    private var qnnAdspLibraryPath: String? = null

    private fun ensureLoaded() {
        if (isLoaded) {
            return
        }

        synchronized(this) {
            if (isLoaded) {
                return
            }
            System.loadLibrary("ondevai_native")
            isLoaded = true
        }
    }

    private fun preloadQnnRuntimeIfEnabled(): String {
        if (!BuildConfig.ONDEVAI_ENABLE_QNN) {
            qnnPreloadSummary = "QNN runtime disabled at build"
            return qnnPreloadSummary
        }
        if (qnnRuntimePreloaded) {
            return qnnPreloadSummary
        }

        synchronized(this) {
            if (qnnRuntimePreloaded) {
                return qnnPreloadSummary
            }

            try {
                // This must be set before the HTP stack is loaded so FastRPC can locate
                // the extracted skel plus standard vendor DSP search paths.
                val adspPath = qnnAdspLibraryPath ?: defaultAdspSearchPaths.joinToString(":")
                Os.setenv("ADSP_LIBRARY_PATH", adspPath, true)
            } catch (e: Exception) {
                // Ignore, try to proceed anyway
            }

            val results =
                qnnRuntimeLibraries.map { library ->
                    try {
                        System.loadLibrary(library)
                        "$library=ok"
                    } catch (error: UnsatisfiedLinkError) {
                        "$library=missing(${error.message ?: "unsatisfied_link_error"})"
                    } catch (error: SecurityException) {
                        "$library=blocked(${error.message ?: "security_exception"})"
                    }
                }

            qnnPreloadSummary = "QNN preload: ${results.joinToString(", ")}"
            qnnRuntimePreloaded = true
            return qnnPreloadSummary
        }
    }

    fun preloadLibrary(): String {
        val startMs = SystemClock.elapsedRealtime()
        ensureLoaded()
        val qnnStatus = preloadQnnRuntimeIfEnabled()
        val elapsedMs = SystemClock.elapsedRealtime() - startMs
        return "NativeBridge library loaded in ${elapsedMs} ms | $qnnStatus"
    }

    fun configureQnnAdspLibraryPath(vararg preferredDirs: String): String {
        val path = (preferredDirs.asList() + defaultAdspSearchPaths)
            .filter { it.isNotBlank() }
            .distinct()
            .joinToString(";")
        qnnAdspLibraryPath = path
        return try {
            Os.setenv("ADSP_LIBRARY_PATH", path, true)
            "ADSP_LIBRARY_PATH=$path"
        } catch (error: Exception) {
            "ADSP_LIBRARY_PATH(set failed: ${error.message ?: error::class.java.simpleName})=$path"
        }
    }

    fun setStreamingListener(listener: StreamingListener?) {
        streamingListener = listener
    }

    fun nativeInitialize(nativeLibDir: String): String {
        ensureLoaded()
        preloadQnnRuntimeIfEnabled()
        return nativeInitializeImpl(nativeLibDir)
    }

    fun nativeLoadModel(
        modelPath: String,
        contextSize: Int,
        threadCount: Int,
        backendTarget: String,
    ): String {
        ensureLoaded()
        return nativeLoadModelImpl(modelPath, contextSize, threadCount, backendTarget)
    }

    fun nativeGenerateStream(
        prompt: String,
        maxTokens: Int,
    ) {
        ensureLoaded()
        nativeGenerateStreamImpl(prompt, maxTokens)
    }

    fun nativeStopGeneration() {
        ensureLoaded()
        nativeStopGenerationImpl()
    }

    fun nativeRunBenchmark(mode: String): String {
        ensureLoaded()
        return nativeRunBenchmarkImpl(mode)
    }

    fun nativeGetRuntimeInfo(): String {
        ensureLoaded()
        return nativeGetRuntimeInfoImpl()
    }

    fun nativeProbeOpenCl(): String {
        ensureLoaded()
        return nativeProbeOpenClImpl()
    }

    fun nativeSetOpenClGpuLayers(gpuLayers: Int): String {
        ensureLoaded()
        return nativeSetOpenClGpuLayersImpl(gpuLayers)
    }

    fun nativeRunQnnControlSmoke(
        backendLib: String,
        modelLib: String,
        outputDir: String,
    ): String {
        ensureLoaded()
        preloadQnnRuntimeIfEnabled()
        return nativeRunQnnControlSmokeImpl(backendLib, modelLib, outputDir)
    }

    private external fun nativeInitializeImpl(nativeLibDir: String): String

    private external fun nativeLoadModelImpl(
        modelPath: String,
        contextSize: Int,
        threadCount: Int,
        backendTarget: String,
    ): String

    private external fun nativeGenerateStreamImpl(
        prompt: String,
        maxTokens: Int,
    )

    private external fun nativeStopGenerationImpl()

    private external fun nativeRunBenchmarkImpl(mode: String): String

    private external fun nativeGetRuntimeInfoImpl(): String

    private external fun nativeProbeOpenClImpl(): String

    private external fun nativeSetOpenClGpuLayersImpl(gpuLayers: Int): String

    private external fun nativeRunQnnControlSmokeImpl(
        backendLib: String,
        modelLib: String,
        outputDir: String,
    ): String

    @JvmStatic
    fun handleNativeToken(token: String) {
        streamingListener?.onToken(token)
    }

    @JvmStatic
    fun handleNativeComplete(summary: String) {
        streamingListener?.onComplete(summary)
    }

    @JvmStatic
    fun handleNativeError(message: String) {
        streamingListener?.onError(message)
    }
}
