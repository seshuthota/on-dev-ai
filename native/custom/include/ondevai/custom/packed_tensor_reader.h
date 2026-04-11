#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "ondevai/custom/model.h"

namespace ondevai::custom {

struct TensorLoadResult {
    bool ok = false;
    std::string error;
    std::vector<std::uint16_t> fp16_data;
};

[[nodiscard]] TensorLoadResult load_fp16_tensor(
    const std::filesystem::path& model_bin_path,
    const Model& model,
    std::string_view tensor_name);

}  // namespace ondevai::custom
