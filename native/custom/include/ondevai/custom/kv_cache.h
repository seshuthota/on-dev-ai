#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ondevai::custom {

struct KvCacheShape {
    std::uint32_t layer_count = 0;
    std::uint32_t kv_head_count = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t context_capacity = 0;
};

class KvCache {
public:
    explicit KvCache(KvCacheShape shape);

    [[nodiscard]] const KvCacheShape& shape() const;
    [[nodiscard]] std::size_t token_count() const;

    void reset();
    void set_token_count(std::size_t token_count);

    // Get K/V slice at a specific position
    [[nodiscard]] std::span<float> get_k(std::uint32_t layer_idx, std::uint32_t position);
    [[nodiscard]] std::span<const float> get_k(std::uint32_t layer_idx, std::uint32_t position) const;
    [[nodiscard]] std::span<float> get_v(std::uint32_t layer_idx, std::uint32_t position);
    [[nodiscard]] std::span<const float> get_v(std::uint32_t layer_idx, std::uint32_t position) const;

    // Append K/V at current token position, then increment token_count_
    void append(std::uint32_t layer_idx, std::span<const float> k_slice, std::span<const float> v_slice);

    // Get all cached K/V for a layer in range [start_pos, end_pos)
    [[nodiscard]] std::span<const float> get_k_range(std::uint32_t layer_idx, std::size_t start_pos, std::size_t end_pos) const;
    [[nodiscard]] std::span<const float> get_v_range(std::uint32_t layer_idx, std::size_t start_pos, std::size_t end_pos) const;

private:
    KvCacheShape shape_;
    std::size_t token_count_ = 0;
    std::vector<float> k_cache_;  // [layer, position, kv_head, head_dim]
    std::vector<float> v_cache_;  // [layer, position, kv_head, head_dim]
};

}  // namespace ondevai::custom
