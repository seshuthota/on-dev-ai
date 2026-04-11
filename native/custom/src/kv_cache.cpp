#include "ondevai/custom/kv_cache.h"

#include <algorithm>
#include <cstring>

namespace ondevai::custom {

KvCache::KvCache(KvCacheShape shape) : shape_(shape) {
    const std::size_t cache_size =
        static_cast<std::size_t>(shape_.layer_count) *
        shape_.context_capacity *
        static_cast<std::size_t>(shape_.kv_head_count) *
        shape_.head_dim;
    k_cache_.resize(cache_size, 0.0f);
    v_cache_.resize(cache_size, 0.0f);
}

const KvCacheShape& KvCache::shape() const {
    return shape_;
}

std::size_t KvCache::token_count() const {
    return token_count_;
}

void KvCache::reset() {
    token_count_ = 0;
    std::fill(k_cache_.begin(), k_cache_.end(), 0.0f);
    std::fill(v_cache_.begin(), v_cache_.end(), 0.0f);
}

void KvCache::set_token_count(const std::size_t token_count) {
    token_count_ = std::min(token_count, static_cast<std::size_t>(shape_.context_capacity));
}

std::span<float> KvCache::get_k(std::uint32_t layer_idx, std::uint32_t position) {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        position
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    return {k_cache_.data() + offset,
            static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim};
}

std::span<const float> KvCache::get_k(std::uint32_t layer_idx, std::uint32_t position) const {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        position
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    return {k_cache_.data() + offset,
            static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim};
}

std::span<float> KvCache::get_v(std::uint32_t layer_idx, std::uint32_t position) {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        position
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    return {v_cache_.data() + offset,
            static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim};
}

std::span<const float> KvCache::get_v(std::uint32_t layer_idx, std::uint32_t position) const {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        position
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    return {v_cache_.data() + offset,
            static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim};
}

void KvCache::append(std::uint32_t layer_idx, std::span<const float> k_slice, std::span<const float> v_slice) {
    if (token_count_ >= static_cast<std::size_t>(shape_.context_capacity)) {
        return;  // Cache full
    }
    auto k_span = get_k(layer_idx, static_cast<std::uint32_t>(token_count_));
    auto v_span = get_v(layer_idx, static_cast<std::uint32_t>(token_count_));
    std::copy(k_slice.begin(), k_slice.end(), k_span.begin());
    std::copy(v_slice.begin(), v_slice.end(), v_span.begin());
    // NOTE: token_count_ is incremented by Runtime::forward once per decode step,
    // not here, to avoid per-layer double counting.
}

std::span<const float> KvCache::get_k_range(std::uint32_t layer_idx, std::size_t start_pos, std::size_t end_pos) const {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        start_pos
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    const std::size_t count = (end_pos - start_pos) *
                              static_cast<std::size_t>(shape_.kv_head_count) *
                              shape_.head_dim;
    return {k_cache_.data() + offset, count};
}

std::span<const float> KvCache::get_v_range(std::uint32_t layer_idx, std::size_t start_pos, std::size_t end_pos) const {
    const std::size_t offset = (
        static_cast<std::size_t>(layer_idx) * shape_.context_capacity +
        start_pos
    ) * static_cast<std::size_t>(shape_.kv_head_count) * shape_.head_dim;
    const std::size_t count = (end_pos - start_pos) *
                              static_cast<std::size_t>(shape_.kv_head_count) *
                              shape_.head_dim;
    return {v_cache_.data() + offset, count};
}

}  // namespace ondevai::custom
