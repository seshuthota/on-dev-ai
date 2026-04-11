#include "ondevai/custom/reference_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace ondevai::custom {

bool rmsnorm_reference(
    std::span<const float> input,
    std::span<const float> weight,
    float eps,
    std::span<float> output) {
    if (input.empty() || weight.empty() || input.size() != weight.size() ||
        input.size() != output.size()) {
        return false;
    }

    float mean_sq = 0.0F;
    for (const float v : input) {
        mean_sq += v * v;
    }
    mean_sq /= static_cast<float>(input.size());

    const float scale = 1.0F / std::sqrt(mean_sq + eps);

    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i] * scale * weight[i];
    }
    return true;
}

bool rope_reference(
    std::span<float> vector,
    std::uint32_t position,
    float rope_theta,
    std::uint32_t head_dim) {
    if (vector.empty()) {
        return false;
    }
    if (head_dim == 0 || (head_dim & 1u) != 0) {
        return false;
    }
    if (vector.size() % head_dim != 0) {
        return false;
    }
    if (rope_theta <= 0.0f || std::isnan(rope_theta) || std::isinf(rope_theta)) {
        return false;
    }

    const std::uint32_t num_heads = static_cast<std::uint32_t>(vector.size()) / head_dim;
    const std::uint32_t half_dim = head_dim / 2;

    for (std::uint32_t h = 0; h < num_heads; ++h) {
        const std::size_t head_offset = static_cast<std::size_t>(h) * head_dim;
        for (std::uint32_t i = 0; i < half_dim; ++i) {
            const float angle =
                static_cast<float>(position) /
                std::pow(rope_theta, (2.0F * static_cast<float>(i)) / static_cast<float>(head_dim));

            const float cos_val = std::cos(angle);
            const float sin_val = std::sin(angle);

            const std::size_t idx0 = head_offset + i;
            const std::size_t idx1 = head_offset + i + half_dim;

            const float x0 = vector[idx0];
            const float x1 = vector[idx1];

            vector[idx0] = x0 * cos_val - x1 * sin_val;
            vector[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
    return true;
}

float fp16_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 0x1U;
    const std::uint32_t exp = (bits >> 10) & 0x1FU;
    const std::uint32_t frac = bits & 0x3FFU;

    if (exp == 0) {
        if (frac == 0) {
            return sign ? -0.0f : 0.0f;
        }
        return std::ldexp(
            static_cast<float>(sign ? -1 : 1) * static_cast<float>(frac) / 1024.0f,
            -14);
    }
    if (exp == 31) {
        if (frac == 0) {
            return sign ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();
        }
        return sign ? -std::nanf("") : std::nanf("");
    }

    const std::int32_t ieee_exp = static_cast<std::int32_t>(exp) - 15;
    const std::uint32_t ieee_mantissa = frac << 13;
    const std::uint32_t ieee_bits =
        (sign << 31) | (static_cast<std::uint32_t>(ieee_exp + 127) << 23) | ieee_mantissa;

    float result;
    std::memcpy(&result, &ieee_bits, sizeof(result));
    return result;
}

bool matvec_fp16_reference(
    std::span<const std::uint16_t> matrix_row_major,
    std::uint32_t rows,
    std::uint32_t cols,
    std::span<const float> input,
    std::span<float> output) {
    if (rows == 0 || cols == 0) {
        return false;
    }
    if (static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols) !=
        matrix_row_major.size()) {
        return false;
    }
    if (input.size() != cols || output.size() != rows) {
        return false;
    }

    for (std::uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < cols; ++c) {
            const float w = fp16_to_float(matrix_row_major[static_cast<std::size_t>(r) * cols + c]);
            sum += w * input[c];
        }
        output[r] = sum;
    }
    return true;
}

bool softmax_reference(std::span<float> scores) {
    if (scores.empty()) {
        return false;
    }

    // Find max for numerical stability
    float max_val = scores[0];
    for (const float v : scores) {
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (float& v : scores) {
        v = std::exp(v - max_val);
        sum += v;
    }

    // Normalize
    const float inv_sum = 1.0f / sum;
    for (float& v : scores) {
        v *= inv_sum;
    }
    return true;
}

bool softmax_2d_reference(std::span<float> scores, std::uint32_t rows, std::uint32_t cols) {
    if (static_cast<std::uint64_t>(rows) * cols != scores.size()) {
        return false;
    }
    for (std::uint32_t r = 0; r < rows; ++r) {
        auto row_span = scores.subspan(static_cast<std::size_t>(r) * cols, cols);
        if (!softmax_reference(row_span)) return false;
    }
    return true;
}

bool attention_gqa_reference(
    std::span<const float> q,
    std::span<const float> kv_cache_k,
    std::span<const float> kv_cache_v,
    std::uint32_t seq_len,
    std::uint32_t num_q_heads,
    std::uint32_t num_kv_heads,
    std::uint32_t head_dim,
    std::span<float> output) {

    if (seq_len == 0) {
        return false;
    }
    if (q.size() != static_cast<std::size_t>(num_q_heads) * head_dim) {
        return false;
    }
    const std::uint64_t expected_kv_size = static_cast<std::uint64_t>(seq_len) * num_kv_heads * head_dim;
    if (kv_cache_k.size() != expected_kv_size || kv_cache_v.size() != expected_kv_size) {
        return false;
    }
    if (output.size() != static_cast<std::size_t>(num_q_heads) * head_dim) {
        return false;
    }

    const std::uint32_t q_heads_per_kv = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Expand K: [seq_len, num_kv_heads, head_dim] -> [seq_len, num_q_heads, head_dim]
    std::vector<float> k_expanded(static_cast<std::size_t>(seq_len) * num_q_heads * head_dim);
    for (std::uint32_t pos = 0; pos < seq_len; ++pos) {
        for (std::uint32_t qh = 0; qh < num_q_heads; ++qh) {
            const std::uint32_t kh = qh / q_heads_per_kv;
            const std::size_t dst = (static_cast<std::size_t>(pos) * num_q_heads + qh) * head_dim;
            const std::size_t src = (static_cast<std::size_t>(pos) * num_kv_heads + kh) * head_dim;
            std::copy_n(kv_cache_k.data() + src, head_dim, k_expanded.data() + dst);
        }
    }

    // Expand V: same pattern
    std::vector<float> v_expanded(static_cast<std::size_t>(seq_len) * num_q_heads * head_dim);
    for (std::uint32_t pos = 0; pos < seq_len; ++pos) {
        for (std::uint32_t qh = 0; qh < num_q_heads; ++qh) {
            const std::uint32_t kh = qh / q_heads_per_kv;
            const std::size_t dst = (static_cast<std::size_t>(pos) * num_q_heads + qh) * head_dim;
            const std::size_t src = (static_cast<std::size_t>(pos) * num_kv_heads + kh) * head_dim;
            std::copy_n(kv_cache_v.data() + src, head_dim, v_expanded.data() + dst);
        }
    }

    // Compute attention scores: Q @ K.T -> [num_q_heads, seq_len]
    std::vector<float> scores(static_cast<std::size_t>(num_q_heads) * seq_len);
    for (std::uint32_t qh = 0; qh < num_q_heads; ++qh) {
        for (std::uint32_t pos = 0; pos < seq_len; ++pos) {
            float sum = 0.0f;
            for (std::uint32_t d = 0; d < head_dim; ++d) {
                sum += q[qh * head_dim + d] * k_expanded[(static_cast<std::size_t>(pos) * num_q_heads + qh) * head_dim + d];
            }
            scores[qh * seq_len + pos] = sum * scale;
        }
    }

    // Softmax
    std::vector<float> scores_softmax(scores);
    if (!softmax_2d_reference(scores_softmax, num_q_heads, seq_len)) {
        return false;
    }

    // Compute output: scores @ V -> [num_q_heads, head_dim]
    for (std::uint32_t qh = 0; qh < num_q_heads; ++qh) {
        for (std::uint32_t d = 0; d < head_dim; ++d) {
            float sum = 0.0f;
            for (std::uint32_t pos = 0; pos < seq_len; ++pos) {
                sum += scores_softmax[qh * seq_len + pos] * v_expanded[(static_cast<std::size_t>(pos) * num_q_heads + qh) * head_dim + d];
            }
            output[qh * head_dim + d] = sum;
        }
    }

    return true;
}

}  // namespace ondevai::custom