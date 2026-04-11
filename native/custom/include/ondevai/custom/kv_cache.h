#pragma once

#include <cstddef>
#include <cstdint>
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

private:
    KvCacheShape shape_;
    std::size_t token_count_ = 0;
};

}  // namespace ondevai::custom
