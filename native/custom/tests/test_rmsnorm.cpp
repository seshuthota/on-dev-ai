#include "ondevai/custom/reference_kernels.h"

#include <array>
#include <cstddef>
#include <cmath>
#include <iostream>

int main() {
    constexpr std::size_t N = 4;
    const std::array<float, N> input = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, N> weight = {0.1f, 0.2f, 0.3f, 0.4f};
    std::array<float, N> output = {0.0f, 0.0f, 0.0f, 0.0f};
    constexpr float eps = 1e-5f;

    const bool ok = ondevai::custom::rmsnorm_reference(input, weight, eps, output);
    if (!ok) {
        std::cerr << "FAIL: rmsnorm returned error\n";
        return 1;
    }

    float mean_sq = 0.0f;
    for (const float v : input) {
        mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(N);
    const float scale = 1.0f / std::sqrt(mean_sq + eps);

    constexpr float tol = 1e-6f;
    for (std::size_t i = 0; i < N; ++i) {
        const float expected = input[i] * scale * weight[i];
        const float diff = std::abs(output[i] - expected);
        if (diff > tol) {
            std::cerr << "FAIL: rmsnorm[" << i << "] = " << output[i]
                      << ", expected " << expected << ", diff " << diff << "\n";
            return 1;
        }
    }

    std::array<float, N> empty_out = {0.0f, 0.0f, 0.0f, 0.0f};
    if (ondevai::custom::rmsnorm_reference({}, weight, eps, empty_out)) {
        std::cerr << "FAIL: rmsnorm should reject empty input\n";
        return 1;
    }

    if (ondevai::custom::rmsnorm_reference(input, weight, eps, {})) {
        std::cerr << "FAIL: rmsnorm should reject empty output\n";
        return 1;
    }

    if (ondevai::custom::rmsnorm_reference(input, std::array<float, 3>{0.1f, 0.2f, 0.3f}, eps, output)) {
        std::cerr << "FAIL: rmsnorm should reject mismatched sizes\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
