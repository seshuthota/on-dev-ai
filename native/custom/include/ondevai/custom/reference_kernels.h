#pragma once

#include <cstdint>
#include <span>

namespace ondevai::custom {

[[nodiscard]] bool rmsnorm_reference(
    std::span<const float> input,
    std::span<const float> weight,
    float eps,
    std::span<float> output);

[[nodiscard]] bool rope_reference(
    std::span<float> vector,
    std::uint32_t position,
    float rope_theta,
    std::uint32_t head_dim);

[[nodiscard]] float fp16_to_float(std::uint16_t bits);

[[nodiscard]] bool matvec_fp16_reference(
    std::span<const std::uint16_t> matrix_row_major,
    std::uint32_t rows,
    std::uint32_t cols,
    std::span<const float> input,
    std::span<float> output);

}  // namespace ondevai::custom
