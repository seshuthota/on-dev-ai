#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ondevai/custom/benchmark.h"
#include "ondevai/custom/kv_cache.h"
#include "ondevai/custom/model.h"
#include "ondevai/custom/sampler.h"

namespace ondevai::custom {

struct RuntimeOptions {
    std::uint32_t context_length = 512;
    std::uint32_t thread_count = 1;
    bool greedy_decode = true;
    bool enable_profiling = false;
};

struct ProfilerResult {
    std::uint64_t rmsnorm_input_us = 0;
    std::uint64_t matvec_qkv_us = 0;
    std::uint64_t rope_q_us = 0;
    std::uint64_t rope_k_us = 0;
    std::uint64_t attention_us = 0;
    std::uint64_t matvec_o_us = 0;
    std::uint64_t rmsnorm_post_us = 0;
    std::uint64_t matvec_mlp_us = 0;
    std::uint64_t total_layer_us = 0;
};

class Profiler {
public:
    static Profiler& instance();

    void begin_section(const char* name);
    void end_section(const char* name);
    void reset();

    [[nodiscard]] ProfilerResult result() const;
    [[nodiscard]] bool is_enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }

private:
    Profiler() = default;

    struct SectionTiming {
        const char* name = nullptr;
        std::uint64_t elapsed_ns = 0;
        std::chrono::steady_clock::time_point start;
        bool active = false;
    };
    SectionTiming sections_[16];
    std::size_t section_count_ = 0;

    std::uint64_t rmsnorm_input_ns_ = 0;
    std::uint64_t matvec_qkv_ns_ = 0;
    std::uint64_t rope_q_ns_ = 0;
    std::uint64_t rope_k_ns_ = 0;
    std::uint64_t attention_ns_ = 0;
    std::uint64_t matvec_o_ns_ = 0;
    std::uint64_t rmsnorm_post_ns_ = 0;
    std::uint64_t matvec_mlp_ns_ = 0;
    std::uint64_t total_layer_ns_ = 0;

    bool enabled_ = false;
};

struct RuntimeStatus {
    bool ok = false;
    std::string message;
};

class Runtime {
public:
    explicit Runtime(RuntimeOptions options = {});

    [[nodiscard]] const RuntimeOptions& options() const;
    [[nodiscard]] const Model* model() const;

    RuntimeStatus load_model(std::unique_ptr<Model> model);
    RuntimeStatus load_model(std::unique_ptr<Model> model, const std::string& model_bin_path);
    RuntimeStatus reset();

    // Single forward step: takes token_id and position, returns next token_id
    // position=0 for first token (prefill), position>=1 for subsequent tokens
    [[nodiscard]] std::uint32_t forward(std::uint32_t token_id, std::uint32_t position);

    // Reset KV cache and internal state for new generation
    void reset_kv_cache();

    // Get current token count in KV cache
    [[nodiscard]] std::size_t kv_cache_token_count() const;

    // Access logits from last forward() call (for benchmarking)
    [[nodiscard]] std::span<const float> get_logits() const;

    // Get profiler results if profiling is enabled
    [[nodiscard]] ProfilerResult get_profiler_result() const;

    // Enable/disable profiling
    void set_profiler_enabled(bool enabled);

private:
    void compute_layer(std::uint32_t layer_idx,
                      std::span<const float> input,
                      std::span<float> output,
                      std::uint32_t position);
    void compute_attention(std::uint32_t layer_idx,
                          std::span<const float> q,
                          std::span<const float> k,
                          std::span<const float> v,
                          std::span<float> output,
                          std::uint32_t position);

    // Cached weights (all FP16, loaded once at model load)
    std::vector<std::uint16_t> embed_tokens_fp16_;
    std::vector<std::uint16_t> lm_head_fp16_;
    std::vector<std::uint16_t> final_norm_fp16_;
    // Flat layer weights: [layer_count][9 matrices concatenated]
    std::vector<std::vector<std::uint16_t>> layer_weights_;  // [22][9 weight matrices]
    // Offsets into each layer's weight vector for each of the 9 matrices
    std::vector<std::array<std::size_t, 10>> layer_weight_offsets_;  // [22][10 offsets (9 starts + 1 end)]

    // Working buffers
    std::vector<float> hidden_states_;
    std::vector<float> norm_scratch_;
    std::vector<float> q_scratch_;
    std::vector<float> k_scratch_;
    std::vector<float> v_scratch_;
    std::vector<float> attn_scratch_;
    std::vector<float> mlp_scratch_;
    std::vector<float> residual_;
    std::vector<float> logits_;

    // KV Cache
    KvCache kv_cache_;

    // Sampler
    GreedySampler sampler_;

    // Path to model binary (needed for loading weights on demand)
    std::string model_bin_path_;

    RuntimeOptions options_;
    std::unique_ptr<Model> model_;
};

[[nodiscard]] ModelConfig tinyllama_v1_config();

}  // namespace ondevai::custom
