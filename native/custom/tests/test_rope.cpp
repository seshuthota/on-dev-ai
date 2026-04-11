#include "ondevai/custom/reference_kernels.h"

#include <array>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>

int main() {
    constexpr std::uint32_t head_dim = 4;
    std::array<float, head_dim> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, head_dim> original = vec;

    constexpr float rope_theta = 10000.0f;
    constexpr std::uint32_t position = 0;

    bool ok = ondevai::custom::rope_reference(vec, position, rope_theta, head_dim);
    if (!ok) {
        std::cerr << "FAIL: rope returned error\n";
        return 1;
    }

    constexpr float tol = 1e-6f;
    for (std::size_t i = 0; i < head_dim; ++i) {
        if (std::abs(vec[i] - original[i]) > tol) {
            std::cerr << "FAIL: RoPE position 0 should be identity, but [" << i
                      << "] = " << vec[i] << ", expected " << original[i] << "\n";
            return 1;
        }
    }

    vec = {1.0f, 0.0f, 0.0f, 1.0f};
    constexpr std::uint32_t pos_nonzero = 1;
    ok = ondevai::custom::rope_reference(vec, pos_nonzero, rope_theta, head_dim);
    if (!ok) {
        std::cerr << "FAIL: rope returned error\n";
        return 1;
    }

    const float angle0 = static_cast<float>(pos_nonzero) / std::pow(rope_theta, 0.0f / head_dim);
    const float cos0 = std::cos(angle0);
    const float sin0 = std::sin(angle0);

    const float angle1 = static_cast<float>(pos_nonzero) / std::pow(rope_theta, 2.0f / head_dim);
    const float cos1 = std::cos(angle1);
    const float sin1 = std::sin(angle1);

    const float expected0 = 1.0f * cos0 - 0.0f * sin0;
    const float expected1 = 0.0f * cos1 - 1.0f * sin1;
    const float expected2 = 1.0f * sin0 + 0.0f * cos0;
    const float expected3 = 0.0f * sin1 + 1.0f * cos1;

    if (std::abs(vec[0] - expected0) > tol || std::abs(vec[1] - expected1) > tol ||
        std::abs(vec[2] - expected2) > tol || std::abs(vec[3] - expected3) > tol) {
        std::cerr << "FAIL: RoPE mismatch at pos " << pos_nonzero << "\n";
        std::cerr << "  vec=[" << vec[0] << ", " << vec[1] << ", " << vec[2] << ", " << vec[3] << "]\n";
        std::cerr << "  expected=[" << expected0 << ", " << expected1 << ", " << expected2
                  << ", " << expected3 << "]\n";
        return 1;
    }

    std::array<float, 4> bad_vec = {1.0f, 2.0f, 3.0f, 4.0f};
    if (ondevai::custom::rope_reference(bad_vec, 0, rope_theta, 0)) {
        std::cerr << "FAIL: rope should reject head_dim 0\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, rope_theta, 3)) {
        std::cerr << "FAIL: rope should reject odd head_dim\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, rope_theta, 6)) {
        std::cerr << "FAIL: rope should reject mismatched vector/head_dim\n";
        return 1;
    }

    std::span<float> empty_vec;
    if (ondevai::custom::rope_reference(empty_vec, 0, rope_theta, head_dim)) {
        std::cerr << "FAIL: rope should reject empty vector\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, 0.0f, head_dim)) {
        std::cerr << "FAIL: rope should reject rope_theta <= 0\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, -1000.0f, head_dim)) {
        std::cerr << "FAIL: rope should reject negative rope_theta\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, std::numeric_limits<float>::infinity(), head_dim)) {
        std::cerr << "FAIL: rope should reject infinite rope_theta\n";
        return 1;
    }

    if (ondevai::custom::rope_reference(bad_vec, 0, std::nanf(""), head_dim)) {
        std::cerr << "FAIL: rope should reject NaN rope_theta\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
