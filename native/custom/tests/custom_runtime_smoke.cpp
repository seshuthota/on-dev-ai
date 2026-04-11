#include "ondevai/custom/runtime.h"

#include <iostream>
#include <memory>

int main() {
    auto model = std::make_unique<ondevai::custom::Model>(ondevai::custom::tinyllama_v1_config());
    model->add_tensor(ondevai::custom::TensorInfo{
        .name = "model.embed_tokens.weight",
        .dtype = ondevai::custom::DType::fp16,
        .shape = {32000, 2048},
        .offset = 0,
        .byte_size = 32000ULL * 2048ULL * 2ULL,
    });

    ondevai::custom::Runtime runtime;
    const auto status = runtime.load_model(std::move(model));
    if (!status.ok) {
        std::cerr << status.message << "\n";
        return 1;
    }

    const auto* loaded_model = runtime.model();
    if (loaded_model == nullptr || loaded_model->config().model_family != "tinyllama_v1") {
        std::cerr << "loaded model metadata mismatch\n";
        return 1;
    }

    std::cout << status.message << "\n";
    return 0;
}
