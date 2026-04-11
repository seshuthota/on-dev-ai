#include "ondevai/custom/tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <utility>

namespace ondevai::custom {

namespace {

constexpr std::uint32_t VOCAB_BIN_MAGIC = 0x4E494256;  // "VBIN" in little-endian

struct VocabBinHeader {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t vocab_size;
    std::uint32_t merges_count;
};

// Check if sequence starting at pos is U+2581 (3 bytes: E2 96 81)
inline bool is_u2581_at(const char* pos) {
    return static_cast<unsigned char>(pos[0]) == 0xE2 &&
           static_cast<unsigned char>(pos[1]) == 0x96 &&
           static_cast<unsigned char>(pos[2]) == 0x81;
}

// Normalize text per HuggingFace TinyLlama tokenizer:
// - If input starts with ASCII space: replace spaces with U+2581, no extra prefix
// - If input does not start with ASCII space: prepend U+2581, then replace spaces
// - Empty input yields empty result (unless add_bos is true in encode)
inline std::string normalize_text(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    std::string result;
    result.reserve(text.size() * 3 + 3);

    const bool starts_with_space = !text.empty() && text[0] == ' ';

    if (!starts_with_space) {
        result.push_back('\xE2');
        result.push_back('\x96');
        result.push_back('\x81');
    }

    for (char c : text) {
        if (c == ' ') {
            result.push_back('\xE2');
            result.push_back('\x96');
            result.push_back('\x81');
        } else {
            result.push_back(c);
        }
    }

    return result;
}

// Split text on U+2581 boundaries (pre-tokenization)
inline std::vector<std::string> split_on_u2581(const std::string& text) {
    std::vector<std::string> chunks;
    std::string current;

    const char* p = text.c_str();
    std::size_t len = text.size();
    std::size_t i = 0;

    while (i < len) {
        if (is_u2581_at(p + i)) {
            if (!current.empty()) {
                chunks.push_back(current);
                current.clear();
            }
            i += 3;
        } else {
            current.push_back(p[i]);
            i++;
        }
    }

    if (!current.empty()) {
        chunks.push_back(current);
    }

    return chunks;
}

// Apply BPE merges to a chunk, given vocab piece-to-id map and merge list
inline std::vector<std::uint32_t> apply_bpe(
    const std::string& chunk,
    const std::unordered_map<std::string, std::uint32_t>& piece_to_id,
    const std::vector<std::pair<std::string, std::string>>& merges,
    std::uint32_t unk_token_id
) {
    if (chunk.empty()) {
        return {};
    }

    // Split chunk into single-byte UTF-8 pieces
    std::vector<std::string> pieces;
    const char* data = chunk.c_str();
    std::size_t len = chunk.size();
    std::size_t i = 0;

    while (i < len) {
        unsigned char c = static_cast<unsigned char>(data[i]);
        if (c < 0x80) {
            pieces.emplace_back(std::string(1, data[i]));
            i += 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < len) {
            pieces.emplace_back(data + i, 2);
            i += 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < len) {
            pieces.emplace_back(data + i, 3);
            i += 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < len) {
            pieces.emplace_back(data + i, 4);
            i += 4;
        } else {
            pieces.emplace_back(std::string(1, data[i]));
            i += 1;
        }
    }

    // Apply BPE merges greedily based on merge rank order
    for (const auto& merge : merges) {
        const std::string& first = merge.first;
        const std::string& second = merge.second;

        bool changed = true;
        while (changed && pieces.size() > 1) {
            changed = false;
            for (std::size_t idx = 0; idx + 1 < pieces.size(); ++idx) {
                if (pieces[idx] == first && pieces[idx + 1] == second) {
                    pieces[idx] = first + second;
                    pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(idx) + 1);
                    changed = true;
                    break;
                }
            }
        }
    }

    // Convert pieces to token IDs
    std::vector<std::uint32_t> tokens;
    tokens.reserve(pieces.size());
    for (const auto& piece : pieces) {
        auto it = piece_to_id.find(piece);
        if (it != piece_to_id.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(unk_token_id);
        }
    }

    return tokens;
}

// Post-process decoded text: replace U+2581 with space
inline std::string postprocess_decode(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (std::size_t i = 0; i < text.size();) {
        if (i + 2 < text.size() && is_u2581_at(text.c_str() + i)) {
            result += ' ';
            i += 3;
        } else {
            result += text[i];
            ++i;
        }
    }

    // Strip leading space if present
    if (!result.empty() && result[0] == ' ') {
        result.erase(0, 1);
    }

    return result;
}

}  // namespace

Tokenizer::Tokenizer(TokenizerConfig config) : config_(std::move(config)) {}

const TokenizerConfig& Tokenizer::config() const {
    return config_;
}

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& vocab_bin_path, TokenizerConfig config) {
    std::ifstream f(vocab_bin_path, std::ios::binary);
    if (!f) {
        return nullptr;
    }

    auto tokenizer = std::make_unique<Tokenizer>(std::move(config));

    VocabBinHeader header{};
    f.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (f.gcount() != sizeof(header)) {
        return nullptr;
    }

    if (header.magic != VOCAB_BIN_MAGIC || header.version != 1) {
        return nullptr;
    }

    std::uint32_t vocab_section_size = 0;
    f.read(reinterpret_cast<char*>(&vocab_section_size), sizeof(vocab_section_size));
    if (f.gcount() != sizeof(vocab_section_size)) {
        return nullptr;
    }

    std::uint32_t vocab_size = header.vocab_size;
    tokenizer->vocab_.resize(vocab_size);

    for (std::uint32_t i = 0; i < vocab_size; ++i) {
        std::uint16_t piece_len = 0;
        f.read(reinterpret_cast<char*>(&piece_len), sizeof(piece_len));
        if (f.gcount() != sizeof(piece_len)) {
            return nullptr;
        }

        std::string piece(piece_len, '\0');
        f.read(piece.data(), piece_len);
        if (f.gcount() != static_cast<std::streamsize>(piece_len)) {
            return nullptr;
        }

        std::uint32_t id = 0;
        float score = 0.0f;
        f.read(reinterpret_cast<char*>(&id), sizeof(id));
        f.read(reinterpret_cast<char*>(&score), sizeof(score));

        tokenizer->vocab_[i] = {std::move(piece), id, score};
    }

    std::uint32_t merges_section_size = 0;
    f.read(reinterpret_cast<char*>(&merges_section_size), sizeof(merges_section_size));
    if (f.gcount() != sizeof(merges_section_size)) {
        return nullptr;
    }

    std::uint32_t merges_count = header.merges_count;
    tokenizer->merges_.reserve(merges_count);

    for (std::uint32_t i = 0; i < merges_count; ++i) {
        std::uint16_t p1_len = 0;
        f.read(reinterpret_cast<char*>(&p1_len), sizeof(p1_len));
        if (f.gcount() != sizeof(p1_len)) {
            return nullptr;
        }

        std::string p1(p1_len, '\0');
        f.read(p1.data(), p1_len);
        if (f.gcount() != static_cast<std::streamsize>(p1_len)) {
            return nullptr;
        }

        std::uint16_t p2_len = 0;
        f.read(reinterpret_cast<char*>(&p2_len), sizeof(p2_len));
        if (f.gcount() != sizeof(p2_len)) {
            return nullptr;
        }

        std::string p2(p2_len, '\0');
        f.read(p2.data(), p2_len);
        if (f.gcount() != static_cast<std::streamsize>(p2_len)) {
            return nullptr;
        }

        tokenizer->merges_.emplace_back(std::move(p1), std::move(p2));
    }

    return tokenizer;
}

std::vector<std::uint32_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<std::uint32_t> tokens;

    if (add_bos) {
        tokens.push_back(config_.bos_token_id);
    }

    // Normalize text: prepend U+2581, replace space with U+2581
    std::string normalized = normalize_text(text);

    // Build piece-to-id map
    std::unordered_map<std::string, std::uint32_t> piece_to_id;
    piece_to_id.reserve(vocab_.size());
    for (const auto& entry : vocab_) {
        piece_to_id[entry.piece] = entry.id;
    }

    // Apply BPE to the full normalized text (no pre-tokenization splitting)
    std::vector<std::uint32_t> result = apply_bpe(normalized, piece_to_id, merges_, config_.unk_token_id);
    tokens.insert(tokens.end(), result.begin(), result.end());

    return tokens;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    std::string result;

    std::unordered_map<std::uint32_t, std::string> id_to_piece;
    id_to_piece.reserve(vocab_.size());
    for (const auto& entry : vocab_) {
        id_to_piece[entry.id] = entry.piece;
    }

    for (std::uint32_t token : tokens) {
        if (token == config_.eos_token_id) {
            break;
        }
        if (token == config_.bos_token_id) {
            continue;
        }

        auto it = id_to_piece.find(token);
        if (it != id_to_piece.end()) {
            result += it->second;
        } else {
            result += '\xEF';
            result += '\xBF';
            result += '\xBD';
        }
    }

    return postprocess_decode(result);
}

}  // namespace ondevai::custom
