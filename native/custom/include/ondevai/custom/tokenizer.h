#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ondevai::custom {

struct TokenizerConfig {
    std::uint32_t bos_token_id = 1;
    std::uint32_t eos_token_id = 2;
    std::uint32_t unk_token_id = 0;
    std::uint32_t vocab_size = 32000;
};

struct VocabEntry {
    std::string piece;
    std::uint32_t id;
    float score;
};

class Tokenizer {
public:
    // Load tokenizer from vocab.bin file
    static std::unique_ptr<Tokenizer> load(
        const std::string& vocab_bin_path,
        TokenizerConfig config = {}
    );

    explicit Tokenizer(TokenizerConfig config);

    // Encode text to token IDs. If add_bos=true, prepend BOS token.
    std::vector<std::uint32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode token IDs to text. Stops at EOS if encountered.
    std::string decode(const std::vector<std::uint32_t>& tokens) const;

    [[nodiscard]] const TokenizerConfig& config() const;

private:
    TokenizerConfig config_;
    std::vector<VocabEntry> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
};

}  // namespace ondevai::custom
