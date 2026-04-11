#include "app_engine.h"

#include <android/log.h>
#include <dlfcn.h>
#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif
#include "llama.h"
#if ONDEVAI_ENABLE_QNN
#include "qnn_runner.h"
#endif

namespace ondevai {

namespace {

constexpr const char* kLogTag = "OnDevAI";
constexpr int kDefaultContextSize = 2048;
constexpr int kDefaultBatchSize = 512;
constexpr int kOpenClGpuLayers = 20;
constexpr const char* kVendorOpenClPath = "/vendor/lib64/libOpenCL.so";

#if ONDEVAI_ENABLE_QNN
constexpr const char* kQnnRequiredRuntimeLibs[] = {
        "libQnnSystem.so",
        "libQnnCpu.so",
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV79Stub.so",
        "libGenie.so",
        "libQnnGenAiTransformer.so",
};
#endif

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
struct _cl_platform_id;
using cl_platform_id = _cl_platform_id*;
using ClGetPlatformIDsFn = cl_int (*)(cl_uint, cl_platform_id*, cl_uint*);

std::string NormalizeBackendTarget(const std::string& backend_target) {
    std::string normalized;
    normalized.reserve(backend_target.size());
    for (const char ch : backend_target) {
        normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return normalized == "opencl" || normalized == "vulkan" || normalized == "qnn" ? normalized : "cpu";
}

std::string BackendDisplayName(const std::string& backend_target) {
    const auto normalized = NormalizeBackendTarget(backend_target);
    if (normalized == "opencl") return "OpenCL";
    if (normalized == "vulkan") return "Vulkan";
    if (normalized == "qnn") return "QNN";
    return "CPU";
}

const char* BackendDeviceTypeName(const enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:
            return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL:
            return "accel";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return "igpu";
    }

    return "unknown";
}

std::string DescribeBackendDevice(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return "<null>";
    }

    const char* device_name = ggml_backend_dev_name(device);
    const char* description = ggml_backend_dev_description(device);
    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    const char* registry_name = reg != nullptr ? ggml_backend_reg_name(reg) : "<unknown>";

    std::ostringstream summary;
    summary << (device_name != nullptr ? device_name : "<unnamed>")
            << "("
            << (description != nullptr ? description : "<no-description>")
            << ", reg="
            << registry_name
            << ", type="
            << BackendDeviceTypeName(ggml_backend_dev_type(device))
            << ")";
    return summary.str();
}

bool IsOpenClDevice(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return false;
    }

    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg != nullptr) {
        const char* registry_name = ggml_backend_reg_name(reg);
        if (registry_name != nullptr && std::string(registry_name) == "OpenCL") {
            return true;
        }
    }

    const char* device_name = ggml_backend_dev_name(device);
    return device_name != nullptr && std::string(device_name) == "GPUOpenCL";
}

bool IsVulkanDevice(ggml_backend_dev_t device) {
    if (device == nullptr) {
        return false;
    }

    const ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    if (reg != nullptr) {
        const char* registry_name = ggml_backend_reg_name(reg);
        if (registry_name != nullptr && std::string(registry_name) == "Vulkan") {
            return true;
        }
    }

    const char* device_name = ggml_backend_dev_name(device);
    return device_name != nullptr && std::string(device_name) == "GPU";
}

void AndroidLogCallback(ggml_log_level level, const char* text, void* /* user_data */) {
    int priority = ANDROID_LOG_INFO;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        priority = ANDROID_LOG_ERROR;
    } else if (level == GGML_LOG_LEVEL_WARN) {
        priority = ANDROID_LOG_WARN;
    } else if (level == GGML_LOG_LEVEL_DEBUG) {
        priority = ANDROID_LOG_DEBUG;
    }

    __android_log_print(priority, kLogTag, "%s", text);
}

std::string TokenToPiece(const llama_vocab* vocab, const llama_token token) {
    std::array<char, 256> buffer{};
    const int count = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (count < 0) {
        return {};
    }
    return {buffer.data(), static_cast<size_t>(count)};
}

std::string EscapeJson(const std::string& value) {
    std::ostringstream escaped;
    for (const char ch : value) {
        switch (ch) {
            case '\\':
                escaped << "\\\\";
                break;
            case '"':
                escaped << "\\\"";
                break;
            case '\n':
                escaped << "\\n";
                break;
            default:
                escaped << ch;
                break;
        }
    }
    return escaped.str();
}

std::string DescribeFileAccess(const char* path) {
    struct stat file_info {};
    const int stat_result = stat(path, &file_info);
    const int access_result = access(path, R_OK);
    const int saved_errno = errno;

    std::ostringstream status;
    status << path
           << "{stat=" << stat_result
           << ", access=" << access_result
           << ", errno=" << saved_errno;
    if (stat_result == 0) {
        status << ", size=" << static_cast<long long>(file_info.st_size);
    }
    status << "}";
    return status.str();
}

bool FileExists(const std::string& path) {
    struct stat file_info {};
    return stat(path.c_str(), &file_info) == 0;
}

bool DirectoryExists(const std::string& path) {
    struct stat file_info {};
    return stat(path.c_str(), &file_info) == 0 && S_ISDIR(file_info.st_mode);
}

std::string JoinPath(const std::string& base, const std::string& child) {
    if (base.empty()) {
        return child;
    }
    if (base.back() == '/') {
        return base + child;
    }
    return base + "/" + child;
}

bool CanDlopenLibrary(const char* soname) {
    dlerror();
    void* handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        dlerror();
        return false;
    }
    dlclose(handle);
    return true;
}

std::string BuildQnnInventory(const std::string& native_lib_dir) {
#if ONDEVAI_ENABLE_QNN
    std::ostringstream status;
    status << "qnn_compiled=true";
    status << " | qnn_native_lib_dir=" << (native_lib_dir.empty() ? "<unset>" : native_lib_dir);

    int present_libs = 0;
    const int total_libs = static_cast<int>(std::size(kQnnRequiredRuntimeLibs));
    status << " | qnn_packaged_libs=";
    for (int index = 0; index < total_libs; ++index) {
        if (index > 0) {
            status << ",";
        }
        const std::string full_path = native_lib_dir.empty()
                ? std::string(kQnnRequiredRuntimeLibs[index])
                : native_lib_dir + "/" + kQnnRequiredRuntimeLibs[index];
        const bool path_exists = !native_lib_dir.empty() && FileExists(full_path);
        const bool loadable = CanDlopenLibrary(kQnnRequiredRuntimeLibs[index]);
        const bool available = path_exists || loadable;
        if (available) {
            ++present_libs;
        }
        status << kQnnRequiredRuntimeLibs[index] << ":"
               << (available ? "ok" : "missing")
               << "("
               << (loadable ? "dlopen" : "no-dlopen")
               << ","
               << (path_exists ? "path" : "no-path")
               << ")";
    }
    status << " | qnn_packaged_count=" << present_libs << "/" << total_libs;
    status << " | qnn_status=" << (present_libs == total_libs ? "runtime_packaged" : "runtime_incomplete");
    return status.str();
#else
    (void)native_lib_dir;
    return "qnn_compiled=false | qnn_status=disabled_at_build";
#endif
}

std::string ProbeOpenClLibrary(const char* label, const char* library_path) {
    std::ostringstream status;
    status << label << "=";

    dlerror();
    void* handle = dlopen(library_path, RTLD_NOW | RTLD_LOCAL);
    const char* load_error = dlerror();
    if (handle == nullptr) {
        status << "dlopen_failed(" << (load_error != nullptr ? load_error : "<no-error>") << ")";
        return status.str();
    }

    status << "dlopen_ok";

    dlerror();
    auto cl_get_platform_ids = reinterpret_cast<ClGetPlatformIDsFn>(dlsym(handle, "clGetPlatformIDs"));
    const char* symbol_error = dlerror();
    if (cl_get_platform_ids == nullptr) {
        status << ", dlsym_failed(" << (symbol_error != nullptr ? symbol_error : "<no-error>") << ")";
        dlclose(handle);
        return status.str();
    }

    cl_uint platform_count = 0;
    const cl_int result = cl_get_platform_ids(0, nullptr, &platform_count);
    status << ", clGetPlatformIDs=" << result
           << ", platform_count=" << platform_count;

    dlclose(handle);
    return status.str();
}

DecodeMetrics DecodePrompt(
        llama_model* model,
        llama_context* context,
        llama_sampler* sampler,
        const llama_vocab* vocab,
        const std::string& prompt,
        const int max_tokens,
        std::atomic<bool>* stop_requested,
        const std::function<void(const std::string&)>& on_token) {
    DecodeMetrics metrics;
    metrics.prompt_chars = static_cast<int>(prompt.size());

    llama_memory_clear(llama_get_memory(context), true);
    llama_sampler_reset(sampler);

    const int token_count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    if (token_count <= 0) {
        metrics.error = "Prompt tokenization failed";
        return metrics;
    }

    std::vector<llama_token> prompt_tokens(static_cast<size_t>(token_count));
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        metrics.error = "Prompt tokenization write failed";
        return metrics;
    }

    if (static_cast<int>(prompt_tokens.size()) + max_tokens >= static_cast<int>(llama_n_ctx(context))) {
        metrics.error = "Prompt plus decode budget exceeds context window";
        return metrics;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(context, batch) != 0) {
            metrics.error = "Prompt encode failed";
            return metrics;
        }

        llama_token decoder_start = llama_model_decoder_start_token(model);
        if (decoder_start == LLAMA_TOKEN_NULL) {
            decoder_start = llama_vocab_bos(vocab);
        }
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    const auto generation_start = std::chrono::steady_clock::now();
    std::ostringstream generated;

    for (int position = 0; position + batch.n_tokens < token_count + max_tokens; ) {
        if (stop_requested != nullptr && stop_requested->load()) {
            metrics.cancelled = true;
            break;
        }

        if (llama_decode(context, batch) != 0) {
            metrics.error = "llama_decode failed";
            return metrics;
        }

        position += batch.n_tokens;
        const llama_token token = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        const std::string piece = TokenToPiece(vocab, token);
        if (metrics.ttft_ms < 0) {
            const auto first_token_time = std::chrono::steady_clock::now();
            metrics.ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - generation_start).count();
        }

        generated << piece;
        on_token(piece);
        batch = llama_batch_get_one(const_cast<llama_token*>(&token), 1);
        ++metrics.decoded_tokens;
    }

    const auto generation_end = std::chrono::steady_clock::now();
    metrics.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generation_end - generation_start).count();
    metrics.tok_per_sec = metrics.elapsed_ms > 0
            ? (metrics.decoded_tokens * 1000.0) / static_cast<double>(metrics.elapsed_ms)
            : 0.0;
    metrics.text = generated.str();
    metrics.success = true;
    return metrics;
}

}  // namespace

AppEngine& AppEngine::Instance() {
    static AppEngine engine;
    return engine;
}

AppEngine::~AppEngine() {
    std::lock_guard<std::mutex> lock(mutex_);
    ResetModelLocked();
    if (backend_initialized_) {
        llama_backend_free();
    }
}

void AppEngine::EnsureRuntimeInitializedLocked() {
    if (backend_initialized_) {
        return;
    }

    llama_log_set(AndroidLogCallback, nullptr);
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: llama_backend_init start");
    const auto init_start = std::chrono::steady_clock::now();
    llama_backend_init();
    const auto init_end = std::chrono::steady_clock::now();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: llama_backend_init complete in %lld ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count());
    backend_initialized_ = true;
}

void AppEngine::EnsureExternalBackendsLoadedLocked() {
    if (external_backends_loaded_) {
        return;
    }

    if (ggml_backend_reg_by_name("OpenCL") != nullptr) {
        external_backends_loaded_ = true;
        return;
    }

#ifdef GGML_USE_OPENCL
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: registering OpenCL backend on demand");
    ggml_backend_register(ggml_backend_opencl_reg());
#endif

    if (ggml_backend_reg_by_name("OpenCL") != nullptr) {
        external_backends_loaded_ = true;
        return;
    }

    if (native_lib_dir_.empty()) {
        return;
    }

    const auto load_start = std::chrono::steady_clock::now();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: ggml_backend_load_all_from_path start (lazy)");
    ggml_backend_load_all_from_path(native_lib_dir_.c_str());
    const auto load_end = std::chrono::steady_clock::now();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: ggml_backend_load_all_from_path complete in %lld ms (lazy)",
            std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count());
    external_backends_loaded_ = true;
}

std::string AppEngine::Initialize(const std::string& native_lib_dir) {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto init_start = std::chrono::steady_clock::now();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: begin native_lib_dir=%s backend_initialized=%d",
            native_lib_dir.c_str(),
            backend_initialized_ ? 1 : 0);

    if (native_lib_dir_.empty()) {
        native_lib_dir_ = native_lib_dir;
    } else if (native_lib_dir_ != native_lib_dir) {
        __android_log_print(
                ANDROID_LOG_WARN,
                kLogTag,
                "Initialize: native_lib_dir changed from %s to %s, keeping original path",
                native_lib_dir_.c_str(),
                native_lib_dir.c_str());
    }

    EnsureRuntimeInitializedLocked();

    last_backend_inventory_ = BuildBackendInventoryLocked();
    const auto init_end = std::chrono::steady_clock::now();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "Initialize: complete total=%lld ms inventory=%s",
            std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count(),
            last_backend_inventory_.c_str());

    std::ostringstream status;
    status << "Runtime initialized from " << native_lib_dir_
           << " | "
           << last_backend_inventory_;
    return status.str();
}

std::string AppEngine::LoadModel(
        const std::string& model_path,
        const int context_size,
        const int thread_count,
        const std::string& backend_target) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!backend_initialized_) {
        last_error_ = "Runtime is not initialized";
        return last_error_;
    }

    ResetModelLocked();

    const auto load_start = std::chrono::steady_clock::now();
    const std::string normalized_backend = NormalizeBackendTarget(backend_target);
    if (normalized_backend == "qnn") {
#if ONDEVAI_ENABLE_QNN
        last_backend_inventory_ = BuildBackendInventoryLocked();
        const std::string control_root = DirectoryExists(JoinPath(model_path, "qnn/control"))
                ? JoinPath(model_path, "qnn/control")
                : model_path;
        const std::string control_model = JoinPath(control_root, "libqnn_model_8bit_quantized.so");
        const std::string output_dir = JoinPath(control_root, "output");

        if (!FileExists(control_model)) {
            std::ostringstream error;
            error << "QNN control model assets not found"
                  << " | expected_model=" << control_model
                  << " | provided_path=" << model_path
                  << " | expected_layout=<filesDir>/qnn/control/libqnn_model_8bit_quantized.so"
                  << " | " << last_backend_inventory_;
            loaded_backend_target_ = "qnn";
            loaded_backend_effective_ = "QNN (not loaded)";
            last_error_ = error.str();
            return last_error_;
        }

        const auto qnn_result = ondevai::qnn::RunModel("libQnnHtp.so", control_model, output_dir);
        if (!qnn_result.success) {
            std::ostringstream error;
            error << "QNN control model load failed"
                  << " | backend_target=QNN"
                  << " | backend_loaded=QNN (HTP v79 control)"
                  << " | model_path=" << control_model;
            if (!qnn_result.error.empty()) {
                error << " | error=" << qnn_result.error;
            }
            if (!qnn_result.details.empty()) {
                error << " | " << qnn_result.details;
            }
            error << " | " << last_backend_inventory_;
            loaded_backend_target_ = "qnn";
            loaded_backend_effective_ = "QNN (not loaded)";
            last_error_ = error.str();
            return last_error_;
        }

        qnn_control_loaded_ = true;
        qnn_backend_lib_ = "libQnnHtp.so";
        qnn_model_lib_ = control_model;
        qnn_output_dir_ = output_dir;
        model_path_ = control_model;
        loaded_backend_target_ = "qnn";
        loaded_backend_effective_ = "QNN (HTP v79 control)";
        loaded_context_size_ = 0;
        loaded_thread_count_ = 0;
        last_model_load_ms_ = qnn_result.total_ms;
        last_error_.clear();

        std::ostringstream status;
        status << "QNN control model loaded in " << last_model_load_ms_ << " ms"
               << " | backend_target=QNN"
               << " | backend_loaded=" << loaded_backend_effective_
               << " | model_path=" << control_model;
        if (!qnn_result.details.empty()) {
            status << " | " << qnn_result.details;
        }
        status << " | " << last_backend_inventory_;
        return status.str();
#else
        last_backend_inventory_ = BuildBackendInventoryLocked();
        std::ostringstream error;
        error << "QNN backend selected but QNN support is not compiled"
              << " | " << last_backend_inventory_;
        loaded_backend_target_ = "qnn";
        loaded_backend_effective_ = "QNN (not loaded)";
        last_error_ = error.str();
        return last_error_;
#endif
    }

    if (normalized_backend == "cpu") {
        setenv("GGML_DISABLE_OPENCL", "1", 1);
    } else {
        unsetenv("GGML_DISABLE_OPENCL");
    }
    if (normalized_backend == "opencl" || normalized_backend == "vulkan") {
        EnsureExternalBackendsLoadedLocked();
    }
    last_backend_inventory_ = BuildBackendInventoryLocked();

    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: begin backend=%s model_path=%s inventory=%s",
            normalized_backend.c_str(),
            model_path.c_str(),
            last_backend_inventory_.c_str());

    struct stat file_info {};
    const int stat_result = stat(model_path.c_str(), &file_info);
    const int access_result = access(model_path.c_str(), R_OK);
    if (stat_result != 0 || access_result != 0) {
        std::ostringstream error;
        error << "Model file not accessible"
              << " | path=" << model_path
              << " | stat=" << stat_result
              << " | access=" << access_result
              << " | errno=" << errno
              << " | reason=" << std::strerror(errno);
        last_error_ = error.str();
        return last_error_;
    }

    llama_model_params model_params = llama_model_default_params();
    ggml_backend_dev_t requested_devices[2] = {nullptr, nullptr};
    std::string selected_device_summary = "CPU host execution";
    int effective_opencl_gpu_layers = -1;

    if (normalized_backend == "opencl") {
        ggml_backend_dev_t opencl_device = FindOpenClDeviceLocked();
        if (opencl_device == nullptr) {
            std::ostringstream error;
            error << "OpenCL backend requested but GPUOpenCL device is unavailable"
                  << " | " << last_backend_inventory_;
            last_error_ = error.str();
            return last_error_;
        }

        requested_devices[0] = opencl_device;
        model_params.devices = requested_devices;
        effective_opencl_gpu_layers = ResolveOpenClGpuLayersLocked();
        model_params.n_gpu_layers = effective_opencl_gpu_layers;
        selected_device_summary = DescribeBackendDevice(opencl_device);
    } else if (normalized_backend == "vulkan") {
        ggml_backend_dev_t vulkan_device = FindVulkanDeviceLocked();
        if (vulkan_device == nullptr) {
            std::ostringstream error;
            error << "Vulkan backend requested but Vulkan device is unavailable"
                  << " | " << last_backend_inventory_;
            last_error_ = error.str();
            return last_error_;
        }

        requested_devices[0] = vulkan_device;
        model_params.devices = requested_devices;
        model_params.n_gpu_layers = -1;
        selected_device_summary = DescribeBackendDevice(vulkan_device);
    } else {
        ggml_backend_dev_t cpu_device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        if (cpu_device == nullptr) {
            last_error_ = "CPU backend device is unavailable";
            return last_error_;
        }

        requested_devices[0] = cpu_device;
        model_params.devices = requested_devices;
        model_params.n_gpu_layers = 0;
        selected_device_summary = DescribeBackendDevice(cpu_device);
    }

    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: llama_model_load_from_file start backend=%s selected_device=%s n_gpu_layers=%d",
            normalized_backend.c_str(),
            selected_device_summary.c_str(),
            model_params.n_gpu_layers);
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model_ == nullptr) {
        std::ostringstream error;
        error << "Failed to load model from " << model_path
              << " | backend_target=" << BackendDisplayName(normalized_backend)
              << " | selected_device=" << selected_device_summary
              << " | " << last_backend_inventory_;
        last_error_ = error.str();
        return last_error_;
    }
    const auto model_load_end = std::chrono::steady_clock::now();
    const auto model_load_elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(model_load_end - load_start).count();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: llama_model_load_from_file complete in %lld ms backend=%s",
            model_load_elapsed_ms,
            normalized_backend.c_str());

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = context_size > 0 ? context_size : kDefaultContextSize;
    context_params.n_batch = std::min(context_params.n_ctx, static_cast<uint32_t>(kDefaultBatchSize));
    context_params.n_threads = ResolveThreadCount(thread_count);
    context_params.n_threads_batch = context_params.n_threads;
    context_params.flash_attn_type = normalized_backend == "opencl"
            ? LLAMA_FLASH_ATTN_TYPE_DISABLED
            : LLAMA_FLASH_ATTN_TYPE_AUTO;
    context_params.no_perf = false;
    context_params.offload_kqv = normalized_backend == "opencl" || normalized_backend == "vulkan";
    context_params.op_offload = normalized_backend == "opencl" || normalized_backend == "vulkan";

    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: llama_init_from_model start backend=%s threads=%d ctx=%u offload_kqv=%d op_offload=%d",
            normalized_backend.c_str(),
            context_params.n_threads,
            context_params.n_ctx,
            context_params.offload_kqv ? 1 : 0,
            context_params.op_offload ? 1 : 0);
    context_ = llama_init_from_model(model_, context_params);
    if (context_ == nullptr) {
        last_error_ = "Failed to create llama context";
        ResetModelLocked();
        return last_error_;
    }
    const auto context_init_end = std::chrono::steady_clock::now();
    const auto context_init_elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(context_init_end - model_load_end).count();
    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: llama_init_from_model complete in %lld ms backend=%s",
            context_init_elapsed_ms,
            normalized_backend.c_str());

    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    sampler_ = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

    vocab_ = llama_model_get_vocab(model_);
    model_path_ = model_path;
    loaded_backend_target_ = normalized_backend;
    loaded_backend_effective_ = BackendDisplayName(normalized_backend);

    const auto load_end = std::chrono::steady_clock::now();
    last_model_load_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
    last_error_.clear();

    __android_log_print(
            ANDROID_LOG_INFO,
            kLogTag,
            "LoadModel: complete total=%lld ms backend=%s threads=%d ctx=%d",
            last_model_load_ms_,
            normalized_backend.c_str(),
            context_params.n_threads,
            static_cast<int>(context_params.n_ctx));

    char description[128];
    llama_model_desc(model_, description, sizeof(description));

    std::ostringstream status;
    status << "Model loaded in " << last_model_load_ms_ << " ms | "
           << description
           << " | backend_target=" << BackendDisplayName(normalized_backend)
           << " | backend_loaded=" << loaded_backend_effective_
           << " | selected_device=" << selected_device_summary
           << " | n_gpu_layers=" << model_params.n_gpu_layers
           << " | threads=" << context_params.n_threads
           << " | ctx=" << context_params.n_ctx
           << " | " << last_backend_inventory_;
    loaded_context_size_ = static_cast<int>(context_params.n_ctx);
    loaded_thread_count_ = context_params.n_threads;
    return status.str();
}

GenerationResult AppEngine::GenerateStreaming(
        const std::string& prompt,
        const int max_tokens,
        const std::function<void(const std::string&)>& on_token) {
    std::lock_guard<std::mutex> lock(mutex_);
    GenerationResult result;

    if (context_ == nullptr || model_ == nullptr || sampler_ == nullptr || vocab_ == nullptr) {
        if (qnn_control_loaded_) {
            last_error_ = "Text generation is not implemented for the QNN control-model path";
            result.error = last_error_;
            return result;
        }
        last_error_ = "Model is not loaded";
        result.error = last_error_;
        return result;
    }

    if (prompt.empty()) {
        last_error_ = "Prompt is empty";
        result.error = last_error_;
        return result;
    }

    stop_requested_.store(false);
    const DecodeMetrics metrics = DecodePrompt(
            model_,
            context_,
            sampler_,
            vocab_,
            prompt,
            max_tokens,
            &stop_requested_,
            on_token);
    if (!metrics.success) {
        last_error_ = metrics.error;
        result.error = last_error_;
        return result;
    }

    std::ostringstream summary;
    summary << "[" << loaded_backend_effective_ << "] decoded=" << metrics.decoded_tokens
            << " tokens, elapsed_ms=" << metrics.elapsed_ms
            << ", tok_per_sec=" << metrics.tok_per_sec;
    if (metrics.ttft_ms >= 0) {
        summary << ", ttft_ms=" << metrics.ttft_ms;
    }
    if (metrics.cancelled) {
        summary << ", cancelled=true";
    }

    result.success = true;
    result.cancelled = metrics.cancelled;
    result.summary = summary.str();
    last_result_summary_ = metrics.text + "\n\n" + result.summary;
    last_error_.clear();
    return result;
}

void AppEngine::RequestStop() {
    stop_requested_.store(true);
}

std::string AppEngine::RunBenchmark(const std::string& mode) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (context_ == nullptr || model_ == nullptr || sampler_ == nullptr || vocab_ == nullptr) {
        if (qnn_control_loaded_) {
#if ONDEVAI_ENABLE_QNN
            const auto qnn_result = ondevai::qnn::RunModel(qnn_backend_lib_, qnn_model_lib_, qnn_output_dir_);
            std::ostringstream json;
            json << "{";
            json << "\"mode\":\"" << EscapeJson(mode) << "\",";
            json << "\"backend_target\":\"qnn\",";
            json << "\"backend_effective\":\"" << EscapeJson(loaded_backend_effective_) << "\",";
            json << "\"model_label\":\"qnn_control_model\",";
            json << "\"model_path\":\"" << EscapeJson(model_path_) << "\",";
            json << "\"threads\":0,";
            json << "\"context_size\":0,";
            json << "\"entries\":[{";
            json << "\"name\":\"qnn_control_run\",";
            json << "\"prompt_chars\":0,";
            json << "\"decoded_tokens\":0,";
            json << "\"elapsed_ms\":" << qnn_result.total_ms << ",";
            json << "\"ttft_ms\":" << qnn_result.total_ms << ",";
            json << "\"tok_per_sec\":0.000000,";
            json << "\"success\":" << (qnn_result.success ? "true" : "false") << ",";
            json << "\"cancelled\":false,";
            json << "\"error\":\"" << EscapeJson(qnn_result.error) << "\"";
            json << "}]}";
            if (!qnn_result.success) {
                last_error_ = qnn_result.error;
            } else {
                last_error_.clear();
            }
            return json.str();
#else
            return R"({"error":"QNN support not compiled"})";
#endif
        }
        return R"({"error":"Model is not loaded"})";
    }

    struct BenchmarkCase {
        const char* name;
        const char* prompt;
        int max_tokens;
    };

    const std::vector<BenchmarkCase> smoke_cases = {
            {"smoke_short", "Summarize why low latency matters for local on-device chat in one sentence.", 32},
    };
    const std::vector<BenchmarkCase> micro_cases = {
            {"micro_short", "Why does low latency matter?", 8},
    };
    const std::vector<BenchmarkCase> standard_cases = {
            {"short", "Explain what Time To First Token means for a chat assistant.", 48},
            {"medium", "Write a concise comparison between CPU-only local inference and GPU-accelerated local inference for mobile chat workloads.", 64},
            {"long", "Describe the engineering tradeoffs among token latency, sustained throughput, thermal throttling, and memory pressure when running a local language model on a Snapdragon phone.", 72},
    };

    const std::vector<BenchmarkCase>* selected_cases = &smoke_cases;
    if (mode == "standard") {
        selected_cases = &standard_cases;
    } else if (mode == "micro") {
        selected_cases = &micro_cases;
    }
    stop_requested_.store(false);

    char description[128];
    llama_model_desc(model_, description, sizeof(description));

    std::ostringstream json;
    json << "{";
    json << "\"mode\":\"" << EscapeJson(mode) << "\",";
    json << "\"backend_target\":\"" << EscapeJson(loaded_backend_target_) << "\",";
    json << "\"backend_effective\":\"" << EscapeJson(loaded_backend_effective_) << "\",";
    json << "\"model_label\":\"" << EscapeJson(description) << "\",";
    json << "\"model_path\":\"" << EscapeJson(model_path_) << "\",";
    json << "\"threads\":" << loaded_thread_count_ << ",";
    json << "\"context_size\":" << loaded_context_size_ << ",";
    json << "\"entries\":[";

    for (size_t index = 0; index < selected_cases->size(); ++index) {
        const auto& benchmark_case = (*selected_cases)[index];
        const DecodeMetrics metrics = DecodePrompt(
                model_,
                context_,
                sampler_,
                vocab_,
                benchmark_case.prompt,
                benchmark_case.max_tokens,
                nullptr,
                [](const std::string&) {});

        if (index > 0) {
            json << ",";
        }
        json << "{";
        json << "\"name\":\"" << EscapeJson(benchmark_case.name) << "\",";
        json << "\"prompt_chars\":" << metrics.prompt_chars << ",";
        json << "\"decoded_tokens\":" << metrics.decoded_tokens << ",";
        json << "\"elapsed_ms\":" << metrics.elapsed_ms << ",";
        json << "\"ttft_ms\":" << metrics.ttft_ms << ",";
        json << "\"tok_per_sec\":" << std::fixed << std::setprecision(6) << metrics.tok_per_sec << ",";
        json << "\"success\":" << (metrics.success ? "true" : "false") << ",";
        json << "\"cancelled\":" << (metrics.cancelled ? "true" : "false") << ",";
        json << "\"error\":\"" << EscapeJson(metrics.error) << "\"";
        json << "}";
    }

    json << "]}";
    return json.str();
}

std::string AppEngine::GetRuntimeInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string backend_inventory = BuildBackendInventoryLocked();

    std::ostringstream info;
    info << "backend_initialized=" << (backend_initialized_ ? "true" : "false")
         << " | model_loaded=" << ((model_ != nullptr || qnn_control_loaded_) ? "true" : "false")
         << " | backend_target=" << loaded_backend_target_
         << " | backend_loaded=" << (loaded_backend_effective_.empty() ? "<none>" : loaded_backend_effective_)
         << " | model_path=" << (model_path_.empty() ? "<unset>" : model_path_)
         << " | last_load_ms=" << last_model_load_ms_
         << " | threads=" << loaded_thread_count_
         << " | ctx=" << loaded_context_size_
         << " | opencl_n_gpu_layers=" << ResolveOpenClGpuLayersLocked()
         << " | " << backend_inventory;

    if (!last_error_.empty()) {
        info << " | last_error=" << last_error_;
    }

    return info.str();
}

std::string AppEngine::ProbeOpenCl() const {
    std::ostringstream info;
    info << "opencl_probe"
         << " | vendor_path=" << DescribeFileAccess(kVendorOpenClPath)
         << " | " << ProbeOpenClLibrary("soname", "libOpenCL.so")
         << " | " << ProbeOpenClLibrary("vendor_abs", kVendorOpenClPath);
    return info.str();
}

std::string AppEngine::SetOpenClGpuLayersOverride(const int gpu_layers) {
    std::lock_guard<std::mutex> lock(mutex_);
    opencl_gpu_layers_override_ = gpu_layers >= 0 ? gpu_layers : -1;

    std::ostringstream status;
    status << "OpenCL n_gpu_layers=" << ResolveOpenClGpuLayersLocked();
    if (opencl_gpu_layers_override_ >= 0) {
        status << " (override)";
    } else {
        status << " (default)";
    }
    return status.str();
}

std::string AppEngine::BuildBackendInventoryLocked() const {
    const std::string qnn_inventory = BuildQnnInventory(native_lib_dir_);

    if (!external_backends_loaded_) {
        std::ostringstream inventory;
        inventory << "backend_inventory=CPU(CPU, reg=CPU, type=cpu); external_backends=deferred"
                  << " | " << qnn_inventory;
        return inventory.str();
    }

    std::ostringstream inventory;
    inventory << "backend_inventory=";

    const size_t device_count = ggml_backend_dev_count();
    if (device_count == 0) {
        inventory << "<none>";
    } else {
        for (size_t index = 0; index < device_count; ++index) {
            if (index > 0) {
                inventory << "; ";
            }
            inventory << DescribeBackendDevice(ggml_backend_dev_get(index));
        }
    }

    inventory << " | " << qnn_inventory;
    return inventory.str();
}

ggml_backend_dev_t AppEngine::FindOpenClDeviceLocked() const {
    if (!external_backends_loaded_) {
        return nullptr;
    }

    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (IsOpenClDevice(device)) {
            return device;
        }
    }

    return nullptr;
}

ggml_backend_dev_t AppEngine::FindVulkanDeviceLocked() const {
    if (!external_backends_loaded_) {
        return nullptr;
    }

    for (size_t index = 0; index < ggml_backend_dev_count(); ++index) {
        ggml_backend_dev_t device = ggml_backend_dev_get(index);
        if (IsVulkanDevice(device)) {
            return device;
        }
    }

    return nullptr;
}

void AppEngine::ResetModelLocked() {
    if (sampler_ != nullptr) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }

    if (context_ != nullptr) {
        llama_free(context_);
        context_ = nullptr;
    }

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    vocab_ = nullptr;
    qnn_control_loaded_ = false;
    qnn_backend_lib_.clear();
    qnn_model_lib_.clear();
    qnn_output_dir_.clear();
    model_path_.clear();
    last_result_summary_.clear();
    loaded_backend_target_ = "cpu";
    loaded_backend_effective_.clear();
    loaded_context_size_ = 0;
    loaded_thread_count_ = 0;
}

int AppEngine::ResolveThreadCount(const int requested_threads) const {
    if (requested_threads > 0) {
        return requested_threads;
    }

    const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count <= 2) {
        return 2;
    }

    return static_cast<int>(std::max(2L, cpu_count - 2));
}

int AppEngine::ResolveOpenClGpuLayersLocked() const {
    return opencl_gpu_layers_override_ >= 0 ? opencl_gpu_layers_override_ : kOpenClGpuLayers;
}

}  // namespace ondevai
