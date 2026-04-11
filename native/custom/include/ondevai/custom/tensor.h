#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ondevai::custom {

enum class DType : std::uint8_t {
    fp16 = 1,
    fp32 = 2,
};

struct TensorInfo {
    std::string name;
    DType dtype = DType::fp16;
    std::vector<std::uint32_t> shape;
    std::uint64_t offset = 0;
    std::uint64_t byte_size = 0;
};

[[nodiscard]] std::size_t element_size_bytes(DType dtype);
[[nodiscard]] std::size_t element_count(const TensorInfo& tensor);

}  // namespace ondevai::custom
