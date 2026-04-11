#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/tokenizer.h"
#include "ondevai/custom/runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::array<const char*, 3> kValidPromptIds = {{
    "short_fact",
    "reasoning_small",
    "chat_medium",
}};

bool is_valid_prompt_id(const std::string& id) {
    return std::any_of(kValidPromptIds.begin(), kValidPromptIds.end(),
                       [&id](const char* valid) { return id == valid; });
}

std::string get_git_commit() {
    FILE* fp = popen("git -C /home/curious/Documents/hermes-projects/OnDevAI rev-parse HEAD 2>/dev/null", "r");
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

struct RunResult {
    std::uint32_t first_token = 0;
    std::size_t generated_count = 0;
    std::int64_t ttft_ms = 0;
    std::int64_t prefill_ms = 0;
    std::int64_t decode_ms = 0;
    double decode_tok_per_sec = 0.0;
    double prefill_tok_per_sec = 0.0;
    std::int64_t total_ms = 0;
    bool reached_eos = false;
};

RunResult run_decode(
    ondevai::custom::Runtime& runtime,
    const ondevai::custom::Tokenizer& tokenizer,
    const std::vector<std::uint32_t>& prompt_tokens,
    std::uint32_t max_new_tokens,
    bool verbose
) {
    RunResult result;
    const std::uint32_t prompt_len = static_cast<std::uint32_t>(prompt_tokens.size());
    std::vector<std::uint32_t> generated;
    generated.reserve(max_new_tokens);

    auto generation_start = std::chrono::steady_clock::now();
    std::uint32_t current_token = 0;

    // Prefill phase
    for (std::uint32_t pos = 0; pos < prompt_len; ++pos) {
        current_token = runtime.forward(prompt_tokens[pos], pos);
    }
    generated.push_back(current_token);
    result.first_token = current_token;

    auto ttft_elapsed = std::chrono::steady_clock::now() - generation_start;
    auto decode_start = std::chrono::steady_clock::now();

    // Decode loop
    std::uint32_t position = prompt_len;
    while (generated.size() < max_new_tokens && current_token != tokenizer.config().eos_token_id) {
        current_token = runtime.forward(current_token, position++);
        generated.push_back(current_token);
        if (verbose && generated.size() <= 4) {
            std::cerr << "  decoded " << generated.size() << ": " << current_token << "\n";
        }
        if (current_token == tokenizer.config().eos_token_id) {
            result.reached_eos = true;
            break;
        }
    }

    auto decode_elapsed = std::chrono::steady_clock::now() - decode_start;
    auto total_elapsed = std::chrono::steady_clock::now() - generation_start;

    result.ttft_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ttft_elapsed).count();
    result.prefill_ms = result.ttft_ms;
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(decode_elapsed).count();
    result.total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count();
    result.generated_count = generated.size();

    const std::size_t decode_token_count = generated.size() - 1;
    result.decode_tok_per_sec = result.decode_ms > 0
        ? (decode_token_count * 1000.0) / static_cast<double>(result.decode_ms) : 0.0;
    result.prefill_tok_per_sec = result.ttft_ms > 0
        ? (prompt_len * 1000.0) / static_cast<double>(result.ttft_ms) : 0.0;

    return result;
}

double compute_median(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    }
    return values[n / 2];
}

std::int64_t compute_median_ms(std::vector<std::int64_t>& values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    std::size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2;
    }
    return values[n / 2];
}

void write_jsonl_record(
    std::ostream& jsonl,
    const std::string& baseline_id,
    const std::string& git_commit,
    const std::string& build_id,
    const std::string& model_checksum,
    const std::string& packed_checksum,
    const std::string& prompt_id,
    std::uint32_t prompt_len,
    std::size_t generated_tokens,
    std::int64_t load_time_ms,
    std::int64_t ttft_ms,
    std::int64_t prefill_ms,
    double prefill_tok_per_sec,
    double decode_tok_per_sec,
    std::int64_t elapsed_ms,
    std::uint32_t warmup_count,
    std::uint32_t run_count,
    const std::string& timing_method,
    const std::vector<std::int64_t>& all_ttft_ms,
    const std::vector<double>& all_decode_toks,
    const std::string& status,
    const std::string& error,
    const std::vector<RunResult>& runs,
    const ondevai::custom::ProfilerResult* profiles
) {
    // Build all_ttft_ms JSON array
    std::ostringstream ttft_arr;
    for (std::size_t i = 0; i < all_ttft_ms.size(); ++i) {
        if (i > 0) ttft_arr << ",";
        ttft_arr << all_ttft_ms[i];
    }

    // Build all_decode_toks JSON array
    std::ostringstream dtok_arr;
    dtok_arr << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < all_decode_toks.size(); ++i) {
        if (i > 0) dtok_arr << ",";
        dtok_arr << all_decode_toks[i];
    }

    jsonl << "{"
           << "\"schema_version\":1,"
           << "\"baseline_id\":\"" << baseline_id << "\","
           << "\"git_commit\":\"" << git_commit << "\","
           << "\"build_id\":\"" << build_id << "\","
           << "\"timestamp_utc\":\"" << get_timestamp_utc() << "\","
           << "\"device_serial\":null,"
           << "\"device_model\":null,"
           << "\"backend\":\"custom_cpu\","
           << "\"model_id\":\"TinyLlama-1.1B-Chat-v1.0\","
           << "\"model_checksum\":\"" << model_checksum << "\","
           << "\"packed_checksum\":\"" << packed_checksum << "\","
           << "\"prompt_id\":\"" << prompt_id << "\","
           << "\"seed\":0,"
           << "\"context_length_cap\":512,"
           << "\"warmup_count\":" << warmup_count << ","
           << "\"run_count\":" << run_count << ","
           << "\"timing_method\":\"" << timing_method << "\","
           << "\"prompt_tokens\":" << prompt_len << ","
           << "\"generated_tokens\":" << generated_tokens << ","
           << "\"first_token\":" << (runs.empty() ? 0 : runs[0].first_token) << ","
           << "\"load_time_ms\":" << load_time_ms << ","
           << "\"ttft_ms\":" << ttft_ms << ","
           << "\"prefill_ms\":" << prefill_ms << ","
           << "\"prefill_tok_per_sec\":" << std::fixed << std::setprecision(2) << prefill_tok_per_sec << ","
           << "\"decode_tok_per_sec\":" << std::fixed << std::setprecision(4) << decode_tok_per_sec << ","
           << "\"elapsed_ms\":" << elapsed_ms << ","
           << "\"all_ttft_ms\":[" << ttft_arr.str() << "],"
           << "\"all_decode_tok_per_sec\":[" << dtok_arr.str() << "],"
           << "\"peak_rss_mb\":null,"
           << "\"thermal_samples\":[],"
           << "\"status\":\"" << status << "\","
           << "\"error\":" << (error.empty() ? "null" : "\"" + error + "\"") << ",";

    // Kernel profiles (last field)
    if (profiles) {
        jsonl << "\"kernel_profiles\":{"
              << "\"rmsnorm_input_us\":" << profiles->rmsnorm_input_us << ","
              << "\"matvec_qkv_us\":" << profiles->matvec_qkv_us << ","
              << "\"rope_q_us\":" << profiles->rope_q_us << ","
              << "\"rope_k_us\":" << profiles->rope_k_us << ","
              << "\"attention_us\":" << profiles->attention_us << ","
              << "\"matvec_o_us\":" << profiles->matvec_o_us << ","
              << "\"rmsnorm_post_us\":" << profiles->rmsnorm_post_us << ","
              << "\"matvec_mlp_us\":" << profiles->matvec_mlp_us << ","
              << "\"total_layer_us\":" << profiles->total_layer_us
              << "}}";
    } else {
        jsonl << "\"kernel_profiles\":null}";
    }
    jsonl << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string model_bin_path;
    std::string vocab_bin_path;
    std::string prompt = "Hello";
    std::string output_jsonl_path;
    std::string prompt_id = "short_fact";
    std::string build_id = "unknown";
    std::uint32_t max_new_tokens = 32;
    std::uint32_t warmup_count = 0;
    std::uint32_t run_count = 1;
    bool use_median = true;
    bool verify_determinism = false;
    bool enable_profiling = false;
    bool verbose = false;

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
        } else if (arg == "--prompt-id" && i + 1 < argc) {
            prompt_id = argv[++i];
        } else if (arg == "--build-id" && i + 1 < argc) {
            build_id = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_count = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--runs" && i + 1 < argc) {
            run_count = static_cast<std::uint32_t>(std::atoi(argv[++i]));
        } else if (arg == "--mean") {
            use_median = false;
        } else if (arg == "--verify-determinism") {
            verify_determinism = true;
        } else if (arg == "--profile") {
            enable_profiling = true;
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " --model-bin PATH --vocab-bin PATH [OPTIONS]\n";
            std::cerr << "  --model-bin PATH    Path to model.bin (required)\n";
            std::cerr << "  --vocab-bin PATH   Path to vocab.bin (required)\n";
            std::cerr << "  --prompt TEXT      Prompt (default: Hello)\n";
            std::cerr << "  --max-tokens N     Max new tokens (default: 32)\n";
            std::cerr << "  --prompt-id ID     short_fact|reasoning_small|chat_medium\n";
            std::cerr << "  --build-id ID      Build identifier for device runs\n";
            std::cerr << "  --warmup N         Warmup runs (default: 0)\n";
            std::cerr << "  --runs N           Measured runs (default: 1)\n";
            std::cerr << "  --mean             Use mean instead of median\n";
            std::cerr << "  --verify-determinism  Check first-token determinism\n";
            std::cerr << "  --profile          Enable per-kernel profiling\n";
            std::cerr << "  -v, --verbose      Verbose decode output\n";
            std::cerr << "  --output-jsonl PATH  Append benchmark record here\n";
            return 0;
        }
    }

    if (!is_valid_prompt_id(prompt_id)) {
        std::cerr << "error: --prompt-id must be one of: ";
        for (std::size_t i = 0; i < kValidPromptIds.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            std::cerr << kValidPromptIds[i];
        }
        std::cerr << "\n";
        return 1;
    }

    if (model_bin_path.empty()) {
        std::cerr << "error: --model-bin is required\n";
        return 1;
    }
    if (vocab_bin_path.empty()) {
        std::cerr << "error: --vocab-bin is required\n";
        return 1;
    }

    if (warmup_count > 0) {
        std::cerr << "Note: --warmup N will reset KV cache each run\n";
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

    // Enable profiler if requested
    ondevai::custom::ProfilerResult profiles_result;
    if (enable_profiling) {
        runtime.set_profiler_enabled(true);
        ondevai::custom::Profiler::instance().reset();
        std::cerr << "Profiling enabled\n";
    }

    // ========== WARMUP RUNS ==========
    for (std::uint32_t w = 0; w < warmup_count; ++w) {
        std::cerr << "[warmup " << (w + 1) << "/" << warmup_count << "]\n";
        runtime.reset_kv_cache();
        auto r = run_decode(runtime, *tokenizer, tokens, max_new_tokens, false);
        std::cerr << "  first_token=" << r.first_token << " generated=" << r.generated_count << "\n";
    }

    // ========== MEASURED RUNS ==========
    std::vector<RunResult> runs;
    runs.reserve(run_count);
    std::vector<std::uint32_t> first_tokens;
    first_tokens.reserve(run_count);

    // Reset profiler before measured runs (profiler state accumulates across warmup)
    if (enable_profiling) {
        ondevai::custom::Profiler::instance().reset();
    }

    for (std::uint32_t r = 0; r < run_count; ++r) {
        std::cerr << "[run " << (r + 1) << "/" << run_count << "]\n";
        runtime.reset_kv_cache();
        auto result = run_decode(runtime, *tokenizer, tokens, max_new_tokens, verbose);
        first_tokens.push_back(result.first_token);
        runs.push_back(result);
        std::cerr << "  first_token=" << result.first_token
                  << " ttft=" << result.ttft_ms << "ms"
                  << " decode=" << result.decode_tok_per_sec << " tok/s\n";
    }

    // Capture profiler results after all measured runs
    if (enable_profiling) {
        profiles_result = runtime.get_profiler_result();
        std::cerr << "\n[Kernel Profile (total across all layers and runs)]\n";
        std::cerr << "  rmsnorm_input_us: " << profiles_result.rmsnorm_input_us << "\n";
        std::cerr << "  matvec_qkv_us:    " << profiles_result.matvec_qkv_us << "\n";
        std::cerr << "  rope_q_us:        " << profiles_result.rope_q_us << "\n";
        std::cerr << "  rope_k_us:        " << profiles_result.rope_k_us << "\n";
        std::cerr << "  attention_us:     " << profiles_result.attention_us << "\n";
        std::cerr << "  matvec_o_us:      " << profiles_result.matvec_o_us << "\n";
        std::cerr << "  rmsnorm_post_us:  " << profiles_result.rmsnorm_post_us << "\n";
        std::cerr << "  matvec_mlp_us:    " << profiles_result.matvec_mlp_us << "\n";
    }

    // ========== AGGREGATE ==========
    std::vector<std::int64_t> all_ttft_ms;
    std::vector<double> all_decode_toks;
    std::vector<std::int64_t> all_total_ms;
    std::vector<std::int64_t> all_prefill_ms;

    for (const auto& r : runs) {
        all_ttft_ms.push_back(r.ttft_ms);
        all_decode_toks.push_back(r.decode_tok_per_sec);
        all_total_ms.push_back(r.total_ms);
        all_prefill_ms.push_back(r.prefill_ms);
    }

    std::string timing_method = use_median ? "median" : "mean";
    std::string status = "ok";
    std::string error;

    // Determinism check
    if (verify_determinism && run_count > 1) {
        bool all_same = std::all_of(first_tokens.begin(), first_tokens.end(),
                                    [ft = first_tokens[0]](std::uint32_t t) { return t == ft; });
        if (!all_same) {
            std::cerr << "DETERMINISM FAIL: first tokens differ: ";
            for (std::size_t i = 0; i < first_tokens.size(); ++i) {
                if (i > 0) std::cerr << ", ";
                std::cerr << first_tokens[i];
            }
            std::cerr << "\n";
            status = "non_deterministic";
            error = "first token inconsistent across runs";
        } else {
            std::cerr << "Determinism check: PASS (first token " << first_tokens[0] << " consistent)\n";
        }
    }

    double final_decode_toks;
    std::int64_t final_ttft_ms;
    std::int64_t final_total_ms;
    std::int64_t final_prefill_ms;
    double final_prefill_toks;

    if (use_median) {
        final_decode_toks = compute_median(all_decode_toks);
        final_ttft_ms = compute_median_ms(all_ttft_ms);
        final_total_ms = compute_median_ms(all_total_ms);
        final_prefill_ms = compute_median_ms(all_prefill_ms);
        std::cerr << "\n[Median of " << run_count << " runs]\n";
    } else {
        double sum_dtok = 0, sum_ttft = 0, sum_total = 0, sum_prefill = 0;
        for (std::size_t i = 0; i < runs.size(); ++i) {
            sum_dtok += all_decode_toks[i];
            sum_ttft += static_cast<double>(all_ttft_ms[i]);
            sum_total += static_cast<double>(all_total_ms[i]);
            sum_prefill += static_cast<double>(all_prefill_ms[i]);
        }
        final_decode_toks = sum_dtok / static_cast<double>(runs.size());
        final_ttft_ms = static_cast<std::int64_t>(sum_ttft / static_cast<double>(runs.size()));
        final_total_ms = static_cast<std::int64_t>(sum_total / static_cast<double>(runs.size()));
        final_prefill_ms = static_cast<std::int64_t>(sum_prefill / static_cast<double>(runs.size()));
        std::cerr << "\n[Mean of " << run_count << " runs]\n";
    }

    final_prefill_toks = final_ttft_ms > 0
        ? (prompt_len * 1000.0) / static_cast<double>(final_ttft_ms) : 0.0;

    std::cerr << "  ttft_ms median: " << final_ttft_ms << "ms\n";
    std::cerr << "  prefill tok/s: " << final_prefill_toks << "\n";
    std::cerr << "  decode tok/s median: " << final_decode_toks << "\n";
    std::cerr << "  total elapsed median: " << final_total_ms << "ms\n";

    // ========== WRITE JSONL ==========
    if (!output_jsonl_path.empty()) {
        std::ofstream jsonl(output_jsonl_path, std::ios::app);
        std::string baseline = (build_id == "unknown") ? "tinyllama-v1-full-desktop" : build_id;

        write_jsonl_record(
            jsonl,
            baseline,
            get_git_commit(),
            build_id,
            model_result.source_weight_sha256,
            model_result.packed_model_sha256,
            prompt_id,
            prompt_len,
            runs[0].generated_count,
            load_time_ms,
            final_ttft_ms,
            final_prefill_ms,
            final_prefill_toks,
            final_decode_toks,
            final_total_ms,
            warmup_count,
            run_count,
            timing_method,
            all_ttft_ms,
            all_decode_toks,
            status,
            error,
            runs,
            enable_profiling ? &profiles_result : nullptr
        );
        std::cerr << "\nBenchmark record written to: " << output_jsonl_path << "\n";
    }

    return status == "non_deterministic" ? 1 : 0;
}
