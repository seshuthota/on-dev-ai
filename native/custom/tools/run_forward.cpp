#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/tokenizer.h"
#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string get_git_commit() {
    FILE* fp = popen("git -C /home/curious/Documents/hermes-projects/OnDevAI rev-parse HEAD", "r");
    if (!fp) return "unknown";
    char buf[128] = {0};
    if (fgets(buf, sizeof(buf), fp)) {
        pclose(fp);
        std::string s = buf;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s;
    }
    pclose(fp);
    return "unknown";
}

std::string get_timestamp_utc() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

std::string read_manifest_field(const std::string& path, const std::string& field) {
    std::ifstream f(path);
    if (!f) return "";
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    // Simple JSON field extraction
    std::string search = "\"" + field + "\"";
    size_t pos = content.find(search);
    if (pos == std::string::npos) return "";
    pos = content.find(':', pos);
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '"')) ++pos;
    size_t end = pos;
    while (end < content.size() && content[end] != ',' && content[end] != '"' && content[end] != '\n') ++end;
    return content.substr(pos, end - pos);
}

std::string manifest_dir_from_model_path(const std::string& model_bin_path) {
    size_t last_slash = model_bin_path.rfind('/');
    if (last_slash == std::string::npos) return "";
    std::string dir = model_bin_path.substr(0, last_slash);
    // Check for manifest at dir/manifest.json
    std::string manifest = dir + "/manifest.json";
    std::ifstream f(manifest);
    if (f) return dir;
    // Also check dir/../manifest.json
    size_t parent_slash = dir.rfind('/');
    if (parent_slash == std::string::npos) return "";
    manifest = dir.substr(0, parent_slash) + "/manifest.json";
    f.open(manifest);
    if (f) return dir.substr(0, parent_slash);
    return "";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string model_bin_path;
    std::string vocab_bin_path;
    std::string prompt = "Hello";
    std::string output_jsonl_path;
    std::uint32_t max_new_tokens = 32;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-bin" && i + 1 < argc) {
            model_bin_path = argv[++i];
        } else if (arg == "--vocab-bin" && i + 1 < argc) {
            vocab_bin_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_new_tokens = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--output-jsonl" && i + 1 < argc) {
            output_jsonl_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " --model-bin PATH --vocab-bin PATH [OPTIONS]\n";
            std::cerr << "  --model-bin PATH   Path to model.bin (required)\n";
            std::cerr << "  --vocab-bin PATH  Path to vocab.bin (required)\n";
            std::cerr << "  --prompt TEXT     Prompt to process (default: Hello)\n";
            std::cerr << "  --max-tokens N    Max new tokens to generate (default: 32)\n";
            std::cerr << "  --output-jsonl PATH  Write JSONL benchmark record here\n";
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
    auto load_start = std::chrono::steady_clock::now();
    std::cerr << "Loading model from: " << model_bin_path << "\n";
    auto model_result = ondevai::custom::load_packed_model_metadata(model_bin_path);
    if (!model_result.ok) {
        std::cerr << "error: failed to load model: " << model_result.error << "\n";
        return 1;
    }
    auto model = std::move(model_result.model);
    auto load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - load_start).count();

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

    const std::uint32_t prompt_len = static_cast<std::uint32_t>(tokens.size());
    std::cerr << "Prompt: \"" << prompt << "\" -> " << prompt_len << " tokens\n";

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
    std::cerr << "Model loaded successfully (" << load_time_ms << "ms)\n";

    // Prefill phase
    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);

    auto generation_start = std::chrono::steady_clock::now();
    std::uint32_t current_token = 0;

    std::cerr << "\n=== PREFILL PHASE ===\n";
    for (std::uint32_t pos = 0; pos < prompt_len; ++pos) {
        current_token = runtime.forward(tokens[pos], pos);
        std::cerr << "  pos=" << pos << " token=" << tokens[pos] << " -> next=" << current_token << "\n";
    }
    generated.push_back(current_token);

    auto ttft_start = std::chrono::steady_clock::now();
    std::cerr << "\n=== DECODE LOOP ===\n";

    // Decode loop
    std::uint32_t position = prompt_len;
    while (generated.size() < max_new_tokens && current_token != tokenizer->config().eos_token_id) {
        current_token = runtime.forward(current_token, position++);
        generated.push_back(current_token);
        if (generated.size() <= 8 || generated.size() == max_new_tokens) {
            std::cerr << "  decoded token " << generated.size() << ": " << current_token << "\n";
        }
        if (current_token == tokenizer->config().eos_token_id) {
            std::cerr << "  EOS token (" << tokenizer->config().eos_token_id << ") reached at position " << generated.size() << "\n";
            break;
        }
    }

    auto total_elapsed = std::chrono::steady_clock::now() - generation_start;
    auto ttft_elapsed = std::chrono::steady_clock::now() - ttft_start;
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count();
    auto ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ttft_elapsed).count();

    const std::size_t decode_token_count = generated.size() - 1;  // exclude first prefill token
    const double decode_ms = static_cast<double>(total_ms - ttft_ms);
    const double decode_tok_per_sec = decode_ms > 0 ? (decode_token_count * 1000.0) / decode_ms : 0.0;
    const double prefill_tok_per_sec = ttft_ms > 0 ? (prompt_len * 1000.0) / ttft_ms : 0.0;

    std::cerr << "\n=== RESULTS ===\n";
    std::cerr << "Total generated: " << generated.size() << " tokens\n";
    std::cerr << "Prefill: " << prompt_len << " tokens in " << ttft_ms << "ms (" << prefill_tok_per_sec << " tok/s)\n";
    std::cerr << "Decode: " << decode_token_count << " tokens in " << decode_ms << "ms (" << decode_tok_per_sec << " tok/s)\n";
    std::cerr << "Total elapsed: " << total_ms << "ms\n";

    // Decode generated tokens
    std::string decoded = tokenizer->decode(generated);
    std::cerr << "\nDecoded response: " << decoded << "\n";

    // Print top 5 logits for last position
    auto logits = runtime.get_logits();
    if (!logits.empty()) {
        std::vector<std::pair<float, std::uint32_t>> sorted(logits.size());
        for (std::size_t i = 0; i < logits.size(); ++i) {
            sorted[i] = {logits[i], static_cast<std::uint32_t>(i)};
        }
        std::partial_sort(sorted.begin(), sorted.begin() + 5, sorted.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::cerr << "\nTop 5 next tokens (from last position):\n";
        for (int i = 0; i < 5; ++i) {
            std::cerr << "  " << sorted[i].second << ": " << sorted[i].first << "\n";
        }
    }

    // Write JSONL record
    if (!output_jsonl_path.empty()) {
        std::string manifest_dir = manifest_dir_from_model_path(model_bin_path);
        std::string model_checksum = "";
        std::string packed_checksum = "";
        if (!manifest_dir.empty()) {
            std::string manifest_path = manifest_dir + "/manifest.json";
            model_checksum = read_manifest_field(manifest_path, "source_weight_sha256");
            packed_checksum = read_manifest_field(manifest_path, "packed_model_sha256");
        }

        std::ofstream jsonl(output_jsonl_path, std::ios::app);
        jsonl << "{"
              << "\"schema_version\":1,"
              << "\"baseline_id\":\"tinyllama-v1-full-desktop\","
              << "\"git_commit\":\"" << get_git_commit() << "\","
              << "\"timestamp_utc\":\"" << get_timestamp_utc() << "\","
              << "\"device_serial\":null,"
              << "\"device_model\":null,"
              << "\"backend\":\"custom_cpu\","
              << "\"model_id\":\"TinyLlama-1.1B-Chat-v1.0\","
              << "\"model_checksum\":\"" << model_checksum << "\","
              << "\"packed_checksum\":\"" << packed_checksum << "\","
              << "\"prompt_id\":\"short_fact\","
              << "\"seed\":0,"
              << "\"context_length_cap\":512,"
              << "\"prompt_tokens\":" << prompt_len << ","
              << "\"generated_tokens\":" << generated.size() << ","
              << "\"load_time_ms\":" << load_time_ms << ","
              << "\"ttft_ms\":" << ttft_ms << ","
              << "\"prefill_tok_per_sec\":" << std::fixed << std::setprecision(2) << prefill_tok_per_sec << ","
              << "\"decode_tok_per_sec\":" << std::fixed << std::setprecision(2) << decode_tok_per_sec << ","
              << "\"elapsed_ms\":" << total_ms << ","
              << "\"peak_rss_mb\":null,"
              << "\"thermal_samples\":[],"
              << "\"status\":\"ok\","
              << "\"error\":null"
              << "}\n";
        std::cerr << "\nBenchmark record written to: " << output_jsonl_path << "\n";
    }

    return 0;
}
