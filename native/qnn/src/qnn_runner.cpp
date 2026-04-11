#include "qnn_runner.h"

#include <dlfcn.h>
#include <errno.h>
#include <sys/stat.h>

#include <chrono>
#include <cstring>
#include <sstream>

#ifdef __ANDROID__
#include <android/log.h>
#define QNN_RUNNER_LOG(level, fmt, ...) \
    __android_log_print(level, "OnDevAI", "QnnRunner: " fmt, ##__VA_ARGS__)
#define QNN_RUNNER_INFO(fmt, ...)  QNN_RUNNER_LOG(ANDROID_LOG_INFO, fmt, ##__VA_ARGS__)
#define QNN_RUNNER_WARN(fmt, ...)  QNN_RUNNER_LOG(ANDROID_LOG_WARN, fmt, ##__VA_ARGS__)
#define QNN_RUNNER_ERROR(fmt, ...) QNN_RUNNER_LOG(ANDROID_LOG_ERROR, fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define QNN_RUNNER_INFO(fmt, ...)  fprintf(stdout, "QnnRunner INFO: " fmt "\n", ##__VA_ARGS__)
#define QNN_RUNNER_WARN(fmt, ...)  fprintf(stderr, "QnnRunner WARN: " fmt "\n", ##__VA_ARGS__)
#define QNN_RUNNER_ERROR(fmt, ...) fprintf(stderr, "QnnRunner ERROR: " fmt "\n", ##__VA_ARGS__)
#endif

// QNN SDK headers
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnTypes.h"

namespace ondevai {
namespace qnn {

namespace {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

long long ElapsedMs(const TimePoint& start, const TimePoint& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

bool EnsureDirectory(const std::string& path) {
    if (path.empty()) return false;
    if (::mkdir(path.c_str(), 0755) == 0) return true;
    if (errno == EEXIST) return true;
    return false;
}

/// Resolve a typed symbol from a dlopen handle, logging errors.
template <typename T>
T ResolveSymbol(void* handle, const char* symbol_name, std::string& error) {
    ::dlerror();  // Clear any previous error.
    void* sym = ::dlsym(handle, symbol_name);
    const char* err = ::dlerror();
    if (sym == nullptr || err != nullptr) {
        error = std::string("dlsym failed for ") + symbol_name + ": " + (err ? err : "null symbol");
        return nullptr;
    }
    return reinterpret_cast<T>(sym);
}

/// RAII wrapper for a dlopen handle.
struct DlHandle {
    void* handle = nullptr;

    explicit DlHandle(void* h) : handle(h) {}
    ~DlHandle() {
        if (handle) {
            ::dlclose(handle);
            handle = nullptr;
        }
    }

    DlHandle(const DlHandle&) = delete;
    DlHandle& operator=(const DlHandle&) = delete;
    DlHandle(DlHandle&& other) noexcept : handle(other.handle) { other.handle = nullptr; }
};

// Type aliases for QNN function pointer resolution.
using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(
    const QnnInterface_t*** providerList, uint32_t* numProviders);

// The model .so exports these entry points (per QNN SDK convention):
//   QnnModel_composeGraphs - builds the graph from model weights
//   QnnModel_freeGraphsInfo - frees graph metadata

// GraphInfo struct matching the QNN model .so convention.
struct GraphInfo {
    Qnn_GraphHandle_t graph;
    char* graphName;
    Qnn_Tensor_t* inputTensors;
    uint32_t numInputTensors;
    Qnn_Tensor_t* outputTensors;
    uint32_t numOutputTensors;
};
using GraphInfoPtr = GraphInfo*;

// Function signatures exported by model .so files.
// These match the QNN SDK convention used by qnn-model-lib-generator output.
// Note: ModelError_t is int32_t where 0 = success.
using ComposeGraphsFn = int32_t (*)(
    Qnn_BackendHandle_t,
    QNN_INTERFACE_VER_TYPE,
    Qnn_ContextHandle_t,
    const void*,        // graphConfigsInfo (nullptr for default)
    const uint32_t,     // graphConfigsInfoCount
    GraphInfoPtr**,     // out: graphsInfo
    uint32_t*,          // out: graphsCount
    bool,               // debug
    QnnLog_Callback_t,  // log callback
    QnnLog_Level_t);    // log level

using FreeGraphsInfoFn = int32_t (*)(GraphInfoPtr**, uint32_t);

#ifdef __ANDROID__
void QnnLogCallback(const char* fmt, QnnLog_Level_t level, uint64_t /*timestamp*/, va_list args) {
    int priority = ANDROID_LOG_INFO;
    if (level >= QNN_LOG_LEVEL_ERROR) priority = ANDROID_LOG_ERROR;
    else if (level == QNN_LOG_LEVEL_WARN) priority = ANDROID_LOG_WARN;
    else if (level == QNN_LOG_LEVEL_DEBUG) priority = ANDROID_LOG_DEBUG;
    __android_log_vprint(priority, "QnnRuntime", fmt, args);
}
#else
void QnnLogCallback(const char* fmt, QnnLog_Level_t /*level*/, uint64_t /*timestamp*/, va_list args) {
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}
#endif

}  // namespace

RunResult RunModel(
    const std::string& backend_lib,
    const std::string& model_lib,
    const std::string& output_dir) {

    RunResult result;
    const auto total_start = SteadyClock::now();

    QNN_RUNNER_INFO("begin backend_lib=%s model_lib=%s output_dir=%s",
                    backend_lib.c_str(), model_lib.c_str(), output_dir.c_str());

    if (!output_dir.empty() && !EnsureDirectory(output_dir)) {
        result.error = "Failed to create output directory: " + output_dir;
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }

    // ─────────────────────────────────────────────────────────────
    // Step 1: Load backend shared library
    // ─────────────────────────────────────────────────────────────
    const auto backend_load_start = SteadyClock::now();
    QNN_RUNNER_INFO("loading backend: %s", backend_lib.c_str());

    ::dlerror();
    void* raw_backend_handle = ::dlopen(backend_lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!raw_backend_handle) {
        const char* err = ::dlerror();
        result.error = std::string("Failed to dlopen backend: ") + (err ? err : "unknown");
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }
    DlHandle backend_handle(raw_backend_handle);

    // Resolve QnnInterface_getProviders
    std::string sym_error;
    auto getProviders = ResolveSymbol<QnnInterfaceGetProvidersFn>(
        backend_handle.handle, "QnnInterface_getProviders", sym_error);
    if (!getProviders) {
        result.error = sym_error;
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }

    // Get the interface table
    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    if (QNN_SUCCESS != getProviders(&providers, &numProviders) ||
        providers == nullptr || numProviders == 0) {
        result.error = "QnnInterface_getProviders returned no providers";
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }

    // Find a compatible interface version
    const QnnInterface_t* selectedProvider = nullptr;
    for (uint32_t i = 0; i < numProviders; ++i) {
        if (providers[i] != nullptr &&
            QNN_API_VERSION_MAJOR == providers[i]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <= providers[i]->apiVersion.coreApiVersion.minor) {
            selectedProvider = providers[i];
            break;
        }
    }
    if (!selectedProvider) {
        result.error = "No compatible QNN interface version found";
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }

    const auto& qnn = selectedProvider->QNN_INTERFACE_VER_NAME;

    const auto backend_load_end = SteadyClock::now();
    result.backend_load_ms = ElapsedMs(backend_load_start, backend_load_end);
    QNN_RUNNER_INFO("backend loaded in %lld ms", result.backend_load_ms);

    // Get backend build ID
    if (qnn.backendGetBuildId) {
        const char* buildId = nullptr;
        if (QNN_SUCCESS == qnn.backendGetBuildId(&buildId) && buildId) {
            result.backend_build_id = buildId;
            QNN_RUNNER_INFO("backend build_id=%s", buildId);
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 2: Initialize QNN backend + logging
    // ─────────────────────────────────────────────────────────────
    Qnn_LogHandle_t logHandle = nullptr;
    if (qnn.logCreate) {
        qnn.logCreate(QnnLogCallback, QNN_LOG_LEVEL_WARN, &logHandle);
    }

    Qnn_BackendHandle_t backendHandle = nullptr;
    auto qnnStatus = qnn.backendCreate(logHandle, nullptr, &backendHandle);
    if (QNN_SUCCESS != qnnStatus) {
        std::ostringstream err;
        err << "backendCreate failed with status=" << qnnStatus;
        result.error = err.str();
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        return result;
    }
    QNN_RUNNER_INFO("backend created");

    // ─────────────────────────────────────────────────────────────
    // Step 3: Create device (optional, some backends don't need it)
    // ─────────────────────────────────────────────────────────────
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    if (qnn.deviceCreate) {
        auto deviceStatus = qnn.deviceCreate(logHandle, nullptr, &deviceHandle);
        if (QNN_SUCCESS != deviceStatus) {
            QNN_RUNNER_WARN("deviceCreate returned %d (non-fatal, continuing)", deviceStatus);
            deviceHandle = nullptr;
        } else {
            QNN_RUNNER_INFO("device created");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Step 4: Create context
    // ─────────────────────────────────────────────────────────────
    Qnn_ContextHandle_t contextHandle = nullptr;
    qnnStatus = qnn.contextCreate(backendHandle, deviceHandle, nullptr, &contextHandle);
    if (QNN_SUCCESS != qnnStatus) {
        std::ostringstream err;
        err << "contextCreate failed with status=" << qnnStatus;
        result.error = err.str();
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        qnn.backendFree(backendHandle);
        return result;
    }
    QNN_RUNNER_INFO("context created");

    // ─────────────────────────────────────────────────────────────
    // Step 5: Load model shared library
    // ─────────────────────────────────────────────────────────────
    const auto model_load_start = SteadyClock::now();
    QNN_RUNNER_INFO("loading model: %s", model_lib.c_str());

    ::dlerror();
    void* raw_model_handle = ::dlopen(model_lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!raw_model_handle) {
        const char* err = ::dlerror();
        result.error = std::string("Failed to dlopen model: ") + (err ? err : "unknown");
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        qnn.contextFree(contextHandle, nullptr);
        qnn.backendFree(backendHandle);
        return result;
    }
    DlHandle model_handle(raw_model_handle);

    // Resolve model entry points
    auto composeGraphs = ResolveSymbol<ComposeGraphsFn>(
        model_handle.handle, "QnnModel_composeGraphs", sym_error);
    if (!composeGraphs) {
        result.error = "Model .so missing QnnModel_composeGraphs: " + sym_error;
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        qnn.contextFree(contextHandle, nullptr);
        qnn.backendFree(backendHandle);
        return result;
    }

    auto freeGraphsInfo = ResolveSymbol<FreeGraphsInfoFn>(
        model_handle.handle, "QnnModel_freeGraphsInfo", sym_error);
    // freeGraphsInfo is nice-to-have, not fatal if missing

    const auto model_load_end = SteadyClock::now();
    result.model_load_ms = ElapsedMs(model_load_start, model_load_end);
    QNN_RUNNER_INFO("model loaded in %lld ms", result.model_load_ms);

    // ─────────────────────────────────────────────────────────────
    // Step 6: Compose graphs from model
    // ─────────────────────────────────────────────────────────────
    const auto compose_start = SteadyClock::now();
    QNN_RUNNER_INFO("composing graphs...");

    GraphInfoPtr* graphsInfo = nullptr;
    uint32_t graphsCount = 0;
    int32_t modelStatus = composeGraphs(
        backendHandle,
        qnn,
        contextHandle,
        nullptr,    // graphConfigsInfo
        0,          // graphConfigsInfoCount
        &graphsInfo,
        &graphsCount,
        false,      // debug
        QnnLogCallback,
        QNN_LOG_LEVEL_WARN);

    if (modelStatus != 0) {
        std::ostringstream err;
        err << "QnnModel_composeGraphs failed with status=" << modelStatus;
        result.error = err.str();
        QNN_RUNNER_ERROR("%s", result.error.c_str());
        qnn.contextFree(contextHandle, nullptr);
        qnn.backendFree(backendHandle);
        return result;
    }

    const auto compose_end = SteadyClock::now();
    result.graph_compose_ms = ElapsedMs(compose_start, compose_end);
    QNN_RUNNER_INFO("composed %u graphs in %lld ms", graphsCount, result.graph_compose_ms);

    // ─────────────────────────────────────────────────────────────
    // Step 7: Finalize graphs
    // ─────────────────────────────────────────────────────────────
    const auto finalize_start = SteadyClock::now();
    bool finalize_ok = true;

    for (uint32_t i = 0; i < graphsCount; ++i) {
        if (graphsInfo == nullptr || graphsInfo[i] == nullptr) continue;
        QNN_RUNNER_INFO("finalizing graph[%u]: %s (inputs=%u, outputs=%u)",
                        i,
                        graphsInfo[i]->graphName ? graphsInfo[i]->graphName : "<unnamed>",
                        graphsInfo[i]->numInputTensors,
                        graphsInfo[i]->numOutputTensors);

        qnnStatus = qnn.graphFinalize(graphsInfo[i]->graph, nullptr, nullptr);
        if (QNN_SUCCESS != qnnStatus) {
            std::ostringstream err;
            err << "graphFinalize failed for graph[" << i << "] status=" << qnnStatus;
            result.error = err.str();
            QNN_RUNNER_ERROR("%s", result.error.c_str());
            finalize_ok = false;
            break;
        }
    }

    const auto finalize_end = SteadyClock::now();
    result.graph_finalize_ms = ElapsedMs(finalize_start, finalize_end);
    QNN_RUNNER_INFO("finalize completed in %lld ms ok=%d", result.graph_finalize_ms, finalize_ok ? 1 : 0);

    // ─────────────────────────────────────────────────────────────
    // Step 8: Build details summary
    // ─────────────────────────────────────────────────────────────
    {
        std::ostringstream details;
        details << "graphs_count=" << graphsCount;
        for (uint32_t i = 0; i < graphsCount && graphsInfo && graphsInfo[i]; ++i) {
            details << " | graph[" << i << "]="
                    << (graphsInfo[i]->graphName ? graphsInfo[i]->graphName : "<unnamed>")
                    << " inputs=" << graphsInfo[i]->numInputTensors
                    << " outputs=" << graphsInfo[i]->numOutputTensors;
        }
        if (!result.backend_build_id.empty()) {
            details << " | backend_build_id=" << result.backend_build_id;
        }
        result.details = details.str();
    }

    // ─────────────────────────────────────────────────────────────
    // Step 9: Cleanup
    // ─────────────────────────────────────────────────────────────
    if (freeGraphsInfo && graphsInfo) {
        freeGraphsInfo(&graphsInfo, graphsCount);
    }

    qnn.contextFree(contextHandle, nullptr);

    if (deviceHandle && qnn.deviceFree) {
        qnn.deviceFree(deviceHandle);
    }

    qnn.backendFree(backendHandle);

    if (logHandle && qnn.logFree) {
        qnn.logFree(logHandle);
    }

    // DlHandles auto-close via RAII

    const auto total_end = SteadyClock::now();
    result.total_ms = ElapsedMs(total_start, total_end);
    result.success = finalize_ok;

    QNN_RUNNER_INFO("complete success=%d total_ms=%lld backend_ms=%lld model_ms=%lld compose_ms=%lld finalize_ms=%lld",
                    result.success ? 1 : 0,
                    result.total_ms,
                    result.backend_load_ms,
                    result.model_load_ms,
                    result.graph_compose_ms,
                    result.graph_finalize_ms);
    return result;
}

}  // namespace qnn
}  // namespace ondevai
