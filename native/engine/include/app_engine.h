#pragma once

#include <atomic>
#include <functional>
#include <mutex>
#include <string>

#include "llama.h"

namespace ondevai {

struct GenerationResult {
    bool success = false;
    bool cancelled = false;
    std::string summary;
    std::string error;
};

struct DecodeMetrics {
    bool success = false;
    bool cancelled = false;
    int decoded_tokens = 0;
    int prompt_chars = 0;
    long long elapsed_ms = 0;
    long long ttft_ms = -1;
    double tok_per_sec = 0.0;
    std::string text;
    std::string error;
};

class AppEngine {
 public:
    static AppEngine& Instance();

    std::string Initialize(const std::string& native_lib_dir);
    std::string LoadModel(
            const std::string& model_path,
            int context_size,
            int thread_count,
            const std::string& backend_target);
    GenerationResult GenerateStreaming(
            const std::string& prompt,
            int max_tokens,
            const std::function<void(const std::string&)>& on_token);
    void RequestStop();
    std::string RunBenchmark(const std::string& mode);
    std::string GetRuntimeInfo() const;
    std::string ProbeOpenCl() const;
    std::string SetOpenClGpuLayersOverride(int gpu_layers);

    ~AppEngine();

 private:
    AppEngine() = default;
    AppEngine(const AppEngine&) = delete;
    AppEngine& operator=(const AppEngine&) = delete;

    std::string BuildBackendInventoryLocked() const;
    void EnsureRuntimeInitializedLocked();
    void EnsureExternalBackendsLoadedLocked();
    ggml_backend_dev_t FindOpenClDeviceLocked() const;
    ggml_backend_dev_t FindVulkanDeviceLocked() const;
    void ResetModelLocked();
    int ResolveThreadCount(int requested_threads) const;
    int ResolveOpenClGpuLayersLocked() const;

    mutable std::mutex mutex_;
    bool backend_initialized_ = false;
    bool external_backends_loaded_ = false;
    std::atomic<bool> stop_requested_ = false;
    std::string native_lib_dir_;
    std::string model_path_;
    std::string last_error_;
    std::string last_result_summary_;
    std::string last_backend_inventory_;
    std::string loaded_backend_target_ = "cpu";
    std::string loaded_backend_effective_;
    long long last_model_load_ms_ = 0;
    int loaded_context_size_ = 0;
    int loaded_thread_count_ = 0;
    int opencl_gpu_layers_override_ = -1;
    bool qnn_control_loaded_ = false;
    std::string qnn_backend_lib_;
    std::string qnn_model_lib_;
    std::string qnn_output_dir_;

    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
};

}  // namespace ondevai
