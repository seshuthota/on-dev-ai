#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <limits>
#include <numeric>

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

KvCache::KvCache(KvCacheShape shape) : shape_(shape) {}

const KvCacheShape& KvCache::shape() const {
    return shape_;
}

std::size_t KvCache::token_count() const {
    return token_count_;
}

void KvCache::reset() {
    token_count_ = 0;
}

void KvCache::set_token_count(const std::size_t token_count) {
    token_count_ = std::min(token_count, static_cast<std::size_t>(shape_.context_capacity));
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

Runtime::Runtime(RuntimeOptions options) : options_(options) {}

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
    if (model->config().runtime_context_cap > options_.context_length) {
        return {false, "model context cap exceeds runtime context length"};
    }
    model_ = std::move(model);
    return {true, "custom runtime model loaded"};
}

RuntimeStatus Runtime::reset() {
    model_.reset();
    return {true, "custom runtime reset"};
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
