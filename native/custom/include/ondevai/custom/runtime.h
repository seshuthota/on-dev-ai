#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ondevai/custom/benchmark.h"
#include "ondevai/custom/kv_cache.h"
#include "ondevai/custom/model.h"

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

    RuntimeStatus load_model(std::unique_ptr<Model> model);
    RuntimeStatus reset();

private:
    RuntimeOptions options_;
    std::unique_ptr<Model> model_;
};

[[nodiscard]] ModelConfig tinyllama_v1_config();

}  // namespace ondevai::custom
