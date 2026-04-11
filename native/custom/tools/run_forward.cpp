#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/tokenizer.h"
#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    std::string model_bin_path;
    std::string vocab_bin_path;
    std::string prompt = "Hello";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-bin" && i + 1 < argc) {
            model_bin_path = argv[++i];
        } else if (arg == "--vocab-bin" && i + 1 < argc) {
            vocab_bin_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " --model-bin PATH --vocab-bin PATH [OPTIONS]\n";
            std::cerr << "  --model-bin PATH   Path to model.bin (required)\n";
            std::cerr << "  --vocab-bin PATH   Path to vocab.bin (required)\n";
            std::cerr << "  --prompt TEXT      Prompt to process (default: Hello)\n";
            return 0;
        }
    }

    if (model_bin_path.empty()) {
        std::cerr << "error: --model-bin is required\n";
        return 1;
    }
    if (vocab_bin_path.empty()) {
        std::cerr << "error: --vocab-bin is required\n";
        return 1;
    }

    // Load model metadata
    std::cerr << "Loading model from: " << model_bin_path << "\n";
    auto model_result = ondevai::custom::load_packed_model_metadata(model_bin_path);
    if (!model_result.ok) {
        std::cerr << "error: failed to load model: " << model_result.error << "\n";
        return 1;
    }
    auto model = std::move(model_result.model);

    // Load tokenizer
    std::cerr << "Loading tokenizer from: " << vocab_bin_path << "\n";
    auto tokenizer = ondevai::custom::Tokenizer::load(vocab_bin_path);
    if (!tokenizer) {
        std::cerr << "error: failed to load tokenizer\n";
        return 1;
    }

    // Encode prompt
    auto tokens = tokenizer->encode(prompt, false);
    if (tokens.empty()) {
        std::cerr << "error: tokenizer produced no tokens\n";
        return 1;
    }

    std::cerr << "Prompt: \"" << prompt << "\" -> tokens: [";
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) std::cerr << ", ";
        std::cerr << tokens[i];
    }
    std::cerr << "]\n";

    // Create runtime
    ondevai::custom::RuntimeOptions runtime_opts;
    runtime_opts.context_length = 512;
    ondevai::custom::Runtime runtime(runtime_opts);

    // Load model into runtime
    std::cerr << "Loading model into runtime...\n";
    auto load_status = runtime.load_model(std::move(model), model_bin_path);
    if (!load_status.ok) {
        std::cerr << "error: failed to load model into runtime: " << load_status.message << "\n";
        return 1;
    }
    std::cerr << "Model loaded successfully\n";

    // Run forward for each token in the prompt
    std::vector<std::uint32_t> generated;
    generated.reserve(16);

    for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
        std::cerr << "Forward pass position " << pos << " (token " << tokens[pos] << ")...\n";
        std::uint32_t next_token = runtime.forward(tokens[pos], static_cast<std::uint32_t>(pos));
        generated.push_back(next_token);
        std::cerr << "  Next token: " << next_token << "\n";
    }

    // Decode generated tokens
    std::string decoded = tokenizer->decode(generated);
    std::cerr << "\nDecoded response: " << decoded << "\n";

    // Print top 5 logits for the last token
    std::cerr << "\nTop 5 next tokens (from last position):\n";
    // Note: We don't have direct access to logits here, but we could add a method to get them

    return 0;
}
