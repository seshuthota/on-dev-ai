#include "ondevai/custom/reference_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

int main() {
    if (ondevai::custom::fp16_to_float(0x0000) != 0.0f) {
        std::cerr << "FAIL: fp16(+0) != 0.0f\n";
        return 1;
    }

    if (!std::signbit(ondevai::custom::fp16_to_float(0x8000))) {
        std::cerr << "FAIL: fp16(-0) signbit missing\n";
        return 1;
    }

    constexpr std::uint16_t one_bits = 0x3C00;
    const float one_val = ondevai::custom::fp16_to_float(one_bits);
    if (std::abs(one_val - 1.0f) > 1e-6f) {
        std::cerr << "FAIL: fp16(1.0) = " << one_val << ", expected 1.0\n";
        return 1;
    }

    constexpr std::uint16_t neg_two_bits = 0xC000;
    const float neg_two_val = ondevai::custom::fp16_to_float(neg_two_bits);
    if (std::abs(neg_two_val + 2.0f) > 1e-5f) {
        std::cerr << "FAIL: fp16(-2.0) = " << neg_two_val << ", expected -2.0\n";
        return 1;
    }

    constexpr std::uint16_t inf_bits = 0x7C00;
    const float inf_val = ondevai::custom::fp16_to_float(inf_bits);
    if (!std::isinf(inf_val) || inf_val <= 0) {
        std::cerr << "FAIL: fp16(+inf) = " << inf_val << ", expected +inf\n";
        return 1;
    }

    constexpr std::uint16_t neg_inf_bits = 0xFC00;
    const float neg_inf_val = ondevai::custom::fp16_to_float(neg_inf_bits);
    if (!std::isinf(neg_inf_val) || neg_inf_val >= 0) {
        std::cerr << "FAIL: fp16(-inf) = " << neg_inf_val << ", expected -inf\n";
        return 1;
    }

    constexpr std::uint16_t nan_bits = 0x7C01;
    const float nan_val = ondevai::custom::fp16_to_float(nan_bits);
    if (!std::isnan(nan_val)) {
        std::cerr << "FAIL: fp16(nan) = " << nan_val << ", expected nan\n";
        return 1;
    }

    constexpr std::uint32_t rows = 2;
    constexpr std::uint32_t cols = 3;
    const std::array<std::uint16_t, rows * cols> matrix = {
        0x3C00, 0x4000, 0x4200,
        0x0000, 0x8000, 0x3C00
    };
    const std::array<float, cols> input = {1.0f, 1.0f, 1.0f};
    std::array<float, rows> output = {0.0f, 0.0f};

    bool ok = ondevai::custom::matvec_fp16_reference(matrix, rows, cols, input, output);
    if (!ok) {
        std::cerr << "FAIL: matvec returned error\n";
        return 1;
    }

    const float m00 = ondevai::custom::fp16_to_float(matrix[0]);
    const float m01 = ondevai::custom::fp16_to_float(matrix[1]);
    const float m02 = ondevai::custom::fp16_to_float(matrix[2]);
    const float m10 = ondevai::custom::fp16_to_float(matrix[3]);
    const float m11 = ondevai::custom::fp16_to_float(matrix[4]);
    const float m12 = ondevai::custom::fp16_to_float(matrix[5]);

    const float expected0 = m00 * 1.0f + m01 * 1.0f + m02 * 1.0f;
    const float expected1 = m10 * 1.0f + m11 * 1.0f + m12 * 1.0f;

    constexpr float tol = 1e-4f;
    if (std::abs(output[0] - expected0) > tol || std::abs(output[1] - expected1) > tol) {
        std::cerr << "FAIL: matvec[0] = " << output[0] << ", expected " << expected0 << "\n";
        std::cerr << "FAIL: matvec[1] = " << output[1] << ", expected " << expected1 << "\n";
        return 1;
    }

    if (ondevai::custom::matvec_fp16_reference(matrix, 0, cols, input, output)) {
        std::cerr << "FAIL: matvec should reject zero rows\n";
        return 1;
    }

    if (ondevai::custom::matvec_fp16_reference(matrix, rows, 0, input, output)) {
        std::cerr << "FAIL: matvec should reject zero cols\n";
        return 1;
    }

    std::array<float, 2> short_input = {1.0f, 1.0f};
    if (ondevai::custom::matvec_fp16_reference(matrix, rows, cols, short_input, output)) {
        std::cerr << "FAIL: matvec should reject mismatched input size\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}