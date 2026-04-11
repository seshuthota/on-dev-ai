#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ondevai::custom {

struct TokenizerConfig {
    std::uint32_t bos_token_id = 1;
    std::uint32_t eos_token_id = 2;
    std::uint32_t unk_token_id = 0;
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerConfig config = {});

    [[nodiscard]] const TokenizerConfig& config() const;

private:
    TokenizerConfig config_;
};

}  // namespace ondevai::custom
