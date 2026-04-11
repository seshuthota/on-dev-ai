#pragma once

#include <cstdint>
#include <span>

namespace ondevai::custom {

class GreedySampler {
public:
    [[nodiscard]] std::uint32_t sample(std::span<const float> logits) const;
};

}  // namespace ondevai::custom
