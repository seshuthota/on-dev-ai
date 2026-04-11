#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "ondevai/custom/model.h"

namespace ondevai::custom {

struct PackedModelLoadResult {
    bool ok = false;
    std::string error;
    std::unique_ptr<Model> model;
    std::uint64_t file_size = 0;
};

[[nodiscard]] PackedModelLoadResult load_packed_model_metadata(const std::filesystem::path& model_bin_path);

}  // namespace ondevai::custom
