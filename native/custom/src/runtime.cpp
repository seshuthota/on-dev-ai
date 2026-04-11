#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "ondevai/custom/kv_cache.h"
#include "ondevai/custom/packed_tensor_reader.h"
#include "ondevai/custom/reference_kernels.h"
#include "ondevai/custom/sampler.h"
#include "ondevai/custom/tokenizer.h"

namespace ondevai::custom {

std::size_t element_size_bytes(const DType dtype) {
    switch (dtype) {
        case DType::fp16:
            return 2;
        case DType::fp32:
            return 4;
    }
    return 0;
}

std::size_t element_count(const TensorInfo& tensor) {
    if (tensor.shape.empty()) {
        return 0;
    }
    return std::accumulate(
        tensor.shape.begin(),
        tensor.shape.end(),
        std::size_t{1},
        [](const std::size_t acc, const std::uint32_t dim) {
            return acc * static_cast<std::size_t>(dim);
        });
}

Model::Model(ModelConfig config) : config_(std::move(config)) {}

const ModelConfig& Model::config() const {
    return config_;
}

const std::vector<TensorInfo>& Model::tensors() const {
    return tensors_;
}

void Model::add_tensor(TensorInfo tensor) {
    tensors_.push_back(std::move(tensor));
}

std::uint32_t GreedySampler::sample(const std::span<const float> logits) const {
    if (logits.empty()) {
        return 0;
    }
    const auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<std::uint32_t>(std::distance(logits.begin(), max_it));
}

static float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

Runtime::Runtime(RuntimeOptions options)
    : options_(options),
      kv_cache_({}),
      sampler_() {}

const RuntimeOptions& Runtime::options() const {
    return options_;
}

const Model* Runtime::model() const {
    return model_.get();
}

RuntimeStatus Runtime::load_model(std::unique_ptr<Model> model) {
    if (model == nullptr) {
        return {false, "model is null"};
    }
    if (model->config().model_family != "tinyllama_v1") {
        return {false, "unsupported model family: " + model->config().model_family};
    }
    model_ = std::move(model);
    return {true, "custom runtime model loaded"};
}

RuntimeStatus Runtime::load_model(std::unique_ptr<Model> model, const std::string& model_bin_path) {
    if (model == nullptr) {
        return {false, "model is null"};
    }
    if (model->config().model_family != "tinyllama_v1") {
        return {false, "unsupported model family: " + model->config().model_family};
    }
    if (model->config().runtime_context_cap > options_.context_length) {
        return {false, "model context cap exceeds runtime context length"};
    }

    const auto& config = model->config();

    // Initialize KV cache shape
    const std::uint32_t head_dim = config.hidden_size / config.attention_head_count;
    KvCacheShape kv_shape{
        config.layer_count,
        config.kv_head_count,
        head_dim,
        options_.context_length,
    };
    kv_cache_ = KvCache(kv_shape);

    // Resize weight vectors
    const std::size_t embed_size = static_cast<std::size_t>(config.vocab_size) * config.hidden_size;
    const std::size_t lm_head_size = static_cast<std::size_t>(config.vocab_size) * config.hidden_size;
    const std::size_t final_norm_size = config.hidden_size;
    embed_tokens_fp16_.resize(embed_size);
    lm_head_fp16_.resize(lm_head_size);
    final_norm_fp16_.resize(final_norm_size);

    // Load shared weights
    {
        const auto result = load_fp16_tensor(model_bin_path, *model, "model.embed_tokens.weight");
        if (!result.ok) {
            return {false, "failed to load embed_tokens: " + result.error};
        }
        embed_tokens_fp16_ = std::move(result.fp16_data);
    }
    {
        const auto result = load_fp16_tensor(model_bin_path, *model, "lm_head.weight");
        if (!result.ok) {
            return {false, "failed to load lm_head: " + result.error};
        }
        lm_head_fp16_ = std::move(result.fp16_data);
    }
    {
        const auto result = load_fp16_tensor(model_bin_path, *model, "model.norm.weight");
        if (!result.ok) {
            return {false, "failed to load final norm: " + result.error};
        }
        final_norm_fp16_ = std::move(result.fp16_data);
    }

    // Load layer weights into flat vectors with offset tracking
    layer_weights_.resize(config.layer_count);
    layer_weight_offsets_.resize(config.layer_count);
    for (std::uint32_t layer_idx = 0; layer_idx < config.layer_count; ++layer_idx) {
        const std::string prefix = "model.layers." + std::to_string(layer_idx);
        std::vector<std::uint16_t>& lw = layer_weights_[layer_idx];
        auto& offsets = layer_weight_offsets_[layer_idx];

        offsets[0] = 0;
        std::size_t current_offset = 0;

        const auto load_into = [&](const std::string& name, std::size_t matrix_idx) -> bool {
            const auto result = load_fp16_tensor(model_bin_path, *model, name);
            if (!result.ok) return false;
            current_offset += result.fp16_data.size();
            offsets[matrix_idx + 1] = current_offset;
            lw.insert(lw.end(), result.fp16_data.begin(), result.fp16_data.end());
            return true;
        };

        if (!load_into(prefix + ".input_layernorm.weight", 0)) return {false, "failed to load layer input_layernorm"};
        if (!load_into(prefix + ".self_attn.q_proj.weight", 1)) return {false, "failed to load layer q_proj"};
        if (!load_into(prefix + ".self_attn.k_proj.weight", 2)) return {false, "failed to load layer k_proj"};
        if (!load_into(prefix + ".self_attn.v_proj.weight", 3)) return {false, "failed to load layer v_proj"};
        if (!load_into(prefix + ".self_attn.o_proj.weight", 4)) return {false, "failed to load layer o_proj"};
        if (!load_into(prefix + ".post_attention_layernorm.weight", 5)) return {false, "failed to load layer post_attn_layernorm"};
        if (!load_into(prefix + ".mlp.gate_proj.weight", 6)) return {false, "failed to load layer gate_proj"};
        if (!load_into(prefix + ".mlp.up_proj.weight", 7)) return {false, "failed to load layer up_proj"};
        if (!load_into(prefix + ".mlp.down_proj.weight", 8)) return {false, "failed to load layer down_proj"};
    }

    // Initialize working buffers
    hidden_states_.resize(config.hidden_size);
    norm_scratch_.resize(config.hidden_size);
    q_scratch_.resize(config.hidden_size);
    k_scratch_.resize(config.kv_head_count * head_dim);
    v_scratch_.resize(config.kv_head_count * head_dim);
    attn_scratch_.resize(config.hidden_size);
    mlp_scratch_.resize(config.intermediate_size * 3);
    residual_.resize(config.hidden_size);
    logits_.resize(config.vocab_size);

    model_bin_path_ = model_bin_path;
    model_ = std::move(model);
    return {true, "custom runtime model loaded"};
}

RuntimeStatus Runtime::reset() {
    model_.reset();
    kv_cache_.reset();
    return {true, "custom runtime reset"};
}

void Runtime::reset_kv_cache() {
    kv_cache_.reset();
}

std::size_t Runtime::kv_cache_token_count() const {
    return kv_cache_.token_count();
}

std::span<const float> Runtime::get_logits() const {
    return logits_;
}

std::uint32_t Runtime::forward(std::uint32_t token_id, std::uint32_t position) {
    const auto& config = model_->config();
    const std::uint32_t hidden_size = config.hidden_size;
    const std::uint32_t num_heads = config.attention_head_count;
    const std::uint32_t kv_heads = config.kv_head_count;
    const std::uint32_t head_dim = hidden_size / num_heads;
    const std::uint32_t intermediate_size = config.intermediate_size;
    const std::uint32_t vocab_size = config.vocab_size;
    const float rope_theta = config.rope_theta;
    const float rms_norm_eps = config.rms_norm_eps;

    // Step 1: Embedding lookup
    const std::size_t embed_offset = static_cast<std::size_t>(token_id) * hidden_size;
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        hidden_states_[i] = fp16_to_float(embed_tokens_fp16_[embed_offset + i]);
    }

    // Step 2: Multi-layer forward pass
    std::copy(hidden_states_.begin(), hidden_states_.end(), residual_.begin());
    for (std::uint32_t layer_idx = 0; layer_idx < config.layer_count; ++layer_idx) {
        compute_layer(layer_idx, residual_, hidden_states_, position);
        std::copy(hidden_states_.begin(), hidden_states_.end(), residual_.begin());
    }

    // Increment KV cache token count: all layers have appended their K/V
    kv_cache_.set_token_count(kv_cache_.token_count() + 1);

    // Step 3: Final RMSNorm
    std::vector<float> final_norm_fp32(hidden_size);
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        final_norm_fp32[i] = fp16_to_float(final_norm_fp16_[i]);
    }
    rmsnorm_reference(residual_, final_norm_fp32, rms_norm_eps, hidden_states_);

    // Step 4: Compute logits
    matvec_fp16_reference(lm_head_fp16_, vocab_size, hidden_size, hidden_states_, logits_);

    // Step 5: Sample
    return sampler_.sample(logits_);
}

void Runtime::compute_layer(std::uint32_t layer_idx,
                             std::span<const float> input,
                             std::span<float> output,
                             std::uint32_t position) {
    const auto& config = model_->config();
    const std::uint32_t hidden_size = config.hidden_size;
    const std::uint32_t kv_heads = config.kv_head_count;
    const std::uint32_t head_dim = hidden_size / config.attention_head_count;
    const std::uint32_t intermediate_size = config.intermediate_size;
    const float rope_theta = config.rope_theta;
    const float rms_norm_eps = config.rms_norm_eps;

    const auto& offsets = layer_weight_offsets_[layer_idx];
    const std::vector<std::uint16_t>& lw = layer_weights_[layer_idx];

    // Get spans for each weight matrix
    auto span_w = [&](std::size_t idx) -> std::span<const std::uint16_t> {
        return {lw.data() + offsets[idx], offsets[idx + 1] - offsets[idx]};
    };

    // Convert RMSNorm weights from FP16 to float
    std::vector<float> input_ln_fp32(hidden_size);
    std::vector<float> post_attn_ln_fp32(hidden_size);
    auto ln0 = span_w(0);
    auto ln5 = span_w(5);
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        input_ln_fp32[i] = fp16_to_float(ln0[i]);
        post_attn_ln_fp32[i] = fp16_to_float(ln5[i]);
    }

    // Input RMSNorm
    rmsnorm_reference(input, input_ln_fp32, rms_norm_eps, norm_scratch_);

    // Q/K/V projections (FP16 matvec)
    matvec_fp16_reference(span_w(1), hidden_size, hidden_size, norm_scratch_, q_scratch_);
    matvec_fp16_reference(span_w(2), kv_heads * head_dim, hidden_size, norm_scratch_, k_scratch_);
    matvec_fp16_reference(span_w(3), kv_heads * head_dim, hidden_size, norm_scratch_, v_scratch_);

    // RoPE on Q and K
    rope_reference(q_scratch_, position, rope_theta, head_dim);
    rope_reference(k_scratch_, position, rope_theta, head_dim);

    // Attention with KV cache
    compute_attention(layer_idx, q_scratch_, k_scratch_, v_scratch_, attn_scratch_, position);

    // O projection
    matvec_fp16_reference(span_w(4), hidden_size, hidden_size, attn_scratch_, output);

    // Residual 1: post_attn_out = input + attn_out
    // Save in residual_ (member buffer) to preserve for final residual after MLP
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        residual_[i] = input[i] + output[i];
    }

    // Second RMSNorm (on post-attention residual)
    rmsnorm_reference(residual_, post_attn_ln_fp32, rms_norm_eps, norm_scratch_);

    // MLP gate/up projections (FP16 matvec into scratch buffer)
    float* gate_ptr = mlp_scratch_.data();
    float* up_ptr = mlp_scratch_.data() + intermediate_size;
    matvec_fp16_reference(span_w(6), intermediate_size, hidden_size, norm_scratch_,
                          std::span<float>(gate_ptr, intermediate_size));
    matvec_fp16_reference(span_w(7), intermediate_size, hidden_size, norm_scratch_,
                          std::span<float>(up_ptr, intermediate_size));

    // SiLU activation
    for (std::uint32_t i = 0; i < intermediate_size; ++i) {
        gate_ptr[i] = silu(gate_ptr[i]) * up_ptr[i];
    }

    // MLP down projection (into output, overwriting it)
    matvec_fp16_reference(span_w(8), hidden_size, intermediate_size,
                          std::span<float>(gate_ptr, intermediate_size), output);

    // Residual 2: output = post_attn_out + mlp_out
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        output[i] = residual_[i] + output[i];
    }
}

void Runtime::compute_attention(std::uint32_t layer_idx,
                                 std::span<const float> q,
                                 std::span<const float> k,
                                 std::span<const float> v,
                                 std::span<float> attn_output,
                                 std::uint32_t position) {
    const auto& config = model_->config();
    const std::uint32_t num_heads = config.attention_head_count;
    const std::uint32_t kv_heads = config.kv_head_count;
    const std::uint32_t head_dim = config.hidden_size / num_heads;

    // Append current K/V to cache at current position
    kv_cache_.append(layer_idx, k, v);

    // seq_len is now position+1 (all tokens including current)
    const std::size_t seq_len = kv_cache_.token_count();

    // Get cached K/V for all positions 0..seq_len-1
    auto k_range = kv_cache_.get_k_range(layer_idx, 0, seq_len);
    auto v_range = kv_cache_.get_v_range(layer_idx, 0, seq_len);

    // Compute attention using GQA reference kernel
    attention_gqa_reference(
        q, k_range, v_range,
        static_cast<std::uint32_t>(seq_len),
        num_heads, kv_heads, head_dim,
        attn_output);
}

ModelConfig tinyllama_v1_config() {
    return ModelConfig{
        .model_id = "TinyLlama-1.1B-Chat-v1.0",
        .model_family = "tinyllama_v1",
        .hidden_size = 2048,
        .intermediate_size = 5632,
        .layer_count = 22,
        .attention_head_count = 32,
        .kv_head_count = 4,
        .vocab_size = 32000,
        .max_position_embeddings = 2048,
        .runtime_context_cap = 512,
        .rope_theta = 10000.0F,
        .rms_norm_eps = 1.0e-5F,
    };
}

}  // namespace ondevai::custom
