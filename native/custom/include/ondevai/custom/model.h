#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ondevai/custom/tensor.h"

namespace ondevai::custom {

struct ModelConfig {
    std::string model_id;
    std::string model_family;
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t layer_count = 0;
    std::uint32_t attention_head_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t runtime_context_cap = 0;
    float rope_theta = 10000.0F;
    float rms_norm_eps = 1.0e-5F;
};

class Model {
public:
    explicit Model(ModelConfig config);

    [[nodiscard]] const ModelConfig& config() const;
    [[nodiscard]] const std::vector<TensorInfo>& tensors() const;

    void add_tensor(TensorInfo tensor);

private:
    ModelConfig config_;
    std::vector<TensorInfo> tensors_;
};

}  // namespace ondevai::custom
