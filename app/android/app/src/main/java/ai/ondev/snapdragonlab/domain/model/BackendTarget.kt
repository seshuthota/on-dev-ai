package ai.ondev.snapdragonlab.domain.model

enum class BackendTarget(val key: String, val displayName: String) {
    CPU("cpu", "CPU"),
    OPENCL("opencl", "OpenCL"),
    VULKAN("vulkan", "Vulkan"),
    QNN("qnn", "QNN");

    companion object {
        fun fromKey(key: String): BackendTarget =
            entries.firstOrNull { it.key.equals(key, ignoreCase = true) } ?: CPU
    }
}
