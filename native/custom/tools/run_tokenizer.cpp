#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>

#include "ondevai/custom/tokenizer.h"

int main(int argc, char* argv[]) {
    bool add_bos = false;
    std::string vocab_path;
    std::string text;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--add-bos") {
            add_bos = true;
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr, "Usage: %s [--add-bos] VOCAB_BIN TEXT...\n", argv[0]);
            std::fprintf(stderr, "  VOCAB_BIN   Path to vocab.bin file\n");
            std::fprintf(stderr, "  TEXT        Text to encode (joined with spaces if multiple args)\n");
            std::fprintf(stderr, "  --add-bos   Prepend BOS token\n");
            return 0;
        } else if (vocab_path.empty()) {
            vocab_path = arg;
        } else {
            if (!text.empty()) {
                text += ' ';
            }
            text += arg;
        }
    }

    if (vocab_path.empty() || text.empty()) {
        std::fprintf(stderr, "Usage: %s [--add-bos] VOCAB_BIN TEXT...\n", argv[0]);
        std::fprintf(stderr, "  VOCAB_BIN   Path to vocab.bin file\n");
        std::fprintf(stderr, "  TEXT        Text to encode (joined with spaces if multiple args)\n");
        std::fprintf(stderr, "  --add-bos   Prepend BOS token\n");
        return 1;
    }

    auto tokenizer = ondevai::custom::Tokenizer::load(vocab_path);
    if (!tokenizer) {
        std::fprintf(stderr, "Failed to load tokenizer from: %s\n", vocab_path.c_str());
        return 1;
    }

    auto tokens = tokenizer->encode(text, add_bos);

    std::printf("[");
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            std::printf(", ");
        }
        std::printf("%u", tokens[i]);
    }
    std::printf("]\n");

    return 0;
}
