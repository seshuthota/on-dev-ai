#pragma once

#include <array>
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

    RuntimeStatus load_model(std::unique_ptr<Model> model, const std::string& model_bin_path);
    RuntimeStatus reset();

    // Single forward step: takes token_id and position, returns next token_id
    // position=0 for first token (prefill), position>=1 for subsequent tokens
    [[nodiscard]] std::uint32_t forward(std::uint32_t token_id, std::uint32_t position);

    // Reset KV cache and internal state for new generation
    void reset_kv_cache();

    // Get current token count in KV cache
    [[nodiscard]] std::size_t kv_cache_token_count() const;

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
