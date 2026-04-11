#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/packed_tensor_reader.h"
#include "ondevai/custom/reference_kernels.h"
#include "ondevai/custom/tokenizer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::string json_escape(const std::string& s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') {
            result.push_back('\\');
            result.push_back(c);
        } else if (c == '\n') {
            result.push_back('\\');
            result.push_back('n');
        } else if (c == '\r') {
            result.push_back('\\');
            result.push_back('r');
        } else if (c == '\t') {
            result.push_back('\\');
            result.push_back('t');
        } else {
            result.push_back(c);
        }
    }
    return result;
}

constexpr std::string_view kEmbedTokens = "model.embed_tokens.weight";
constexpr std::string_view kLayer0InputLayernorm = "model.layers.0.input_layernorm.weight";
constexpr std::string_view kLayer0QProj = "model.layers.0.self_attn.q_proj.weight";
constexpr std::string_view kLayer0KProj = "model.layers.0.self_attn.k_proj.weight";
constexpr std::string_view kLayer0VProj = "model.layers.0.self_attn.v_proj.weight";
constexpr std::string_view kLayer0OProj = "model.layers.0.self_attn.o_proj.weight";
constexpr std::string_view kLayer0PostAttnLayernorm = "model.layers.0.post_attention_layernorm.weight";
constexpr std::string_view kLayer0MlpGateProj = "model.layers.0.mlp.gate_proj.weight";
constexpr std::string_view kLayer0MlpUpProj = "model.layers.0.mlp.up_proj.weight";
constexpr std::string_view kLayer0MlpDownProj = "model.layers.0.mlp.down_proj.weight";
constexpr std::string_view kFinalNorm = "model.norm.weight";
constexpr std::string_view kLmHead = "lm_head.weight";

struct CliArgs {
    bool help = false;
    std::optional<std::filesystem::path> model_bin_path;
    std::optional<std::filesystem::path> vocab_bin_path;
    std::optional<std::filesystem::path> output_path;
    std::string prompt = "Hello";
    std::string error;
};

void print_usage() {
    std::cout << "Usage: dump_reference_tensors --model-bin PATH --vocab-bin PATH [OPTIONS]\n";
    std::cout << "Options:\n";
    std::cout << "  --model-bin PATH    Path to packed model.bin file (required)\n";
    std::cout << "  --vocab-bin PATH    Path to vocab.bin file (required)\n";
    std::cout << "  --prompt TEXT       Prompt to process (default: Hello)\n";
    std::cout << "  --output PATH       Output JSON path (required)\n";
    std::cout << "  --help, -h          Show this help\n";
}

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--model-bin") {
            if (i + 1 >= argc) {
                args.error = "missing value for --model-bin";
                return args;
            }
            args.model_bin_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--vocab-bin") {
            if (i + 1 >= argc) {
                args.error = "missing value for --vocab-bin";
                return args;
            }
            args.vocab_bin_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--prompt") {
            if (i + 1 >= argc) {
                args.error = "missing value for --prompt";
                return args;
            }
            args.prompt = argv[++i];
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                args.error = "missing value for --output";
                return args;
            }
            args.output_path = std::filesystem::path(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            args.help = true;
            return args;
        } else {
            args.error = "unknown argument: ";
            args.error += arg;
            return args;
        }
    }
    if (!args.model_bin_path) {
        args.error = "missing required --model-bin argument";
    }
    if (!args.vocab_bin_path) {
        args.error = "missing required --vocab-bin argument";
    }
    if (!args.output_path) {
        args.error = "missing required --output argument";
    }
    return args;
}

std::vector<float> fp16_to_float32(const std::vector<std::uint16_t>& fp16_data) {
    std::vector<float> result(fp16_data.size());
    for (std::size_t i = 0; i < fp16_data.size(); ++i) {
        result[i] = ondevai::custom::fp16_to_float(fp16_data[i]);
    }
    return result;
}

void write_json_header(
    std::ofstream& out,
    const std::string& prompt,
    std::uint32_t token_id,
    const std::filesystem::path& model_bin_path,
    const ondevai::custom::ModelConfig& config) {
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"prompt\": \"" << json_escape(prompt) << "\",\n";
    out << "  \"token_id\": " << token_id << ",\n";
    out << "  \"model_bin\": \"" << json_escape(model_bin_path.string()) << "\",\n";
    out << "  \"config\": {\n";
    out << "    \"hidden_size\": " << config.hidden_size << ",\n";
    out << "    \"intermediate_size\": " << config.intermediate_size << ",\n";
    out << "    \"num_attention_heads\": " << config.attention_head_count << ",\n";
    out << "    \"num_key_value_heads\": " << config.kv_head_count << ",\n";
    out << "    \"vocab_size\": " << config.vocab_size << ",\n";
    out << "    \"rope_theta\": " << config.rope_theta << ",\n";
    out << "    \"rms_norm_eps\": " << config.rms_norm_eps << "\n";
    out << "  },\n";
}

void write_tensor_dump(
    std::ofstream& out,
    const std::string& key,
    const std::vector<std::uint32_t>& shape,
    const std::vector<float>& values,
    bool is_last = false) {
    out << "  \"" << key << "\": {\n";
    out << "    \"shape\": [";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out << shape[i];
        if (i + 1 < shape.size()) out << ", ";
    }
    out << "],\n";
    out << "    \"values\": [";
    const std::size_t max_values = values.size();
    for (std::size_t i = 0; i < max_values; ++i) {
        out << std::setprecision(17) << values[i];
        if (i + 1 < max_values) out << ", ";
    }
    out << "]\n";
    out << "  }" << (is_last ? "\n" : ",\n");
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto args = parse_args(argc, argv);
    if (args.help) {
        print_usage();
        return 0;
    }
    if (!args.error.empty()) {
        std::cerr << "error: " << args.error << "\n";
        print_usage();
        return 1;
    }

    const auto model_result = ondevai::custom::load_packed_model_metadata(*args.model_bin_path);
    if (!model_result.ok) {
        std::cerr << "error: failed to load model: " << model_result.error << "\n";
        return 1;
    }

    const auto& config = model_result.model->config();
    const auto hidden_size = config.hidden_size;
    const auto num_attention_heads = config.attention_head_count;
    const auto num_kv_heads = config.kv_head_count;
    const auto intermediate_size = config.intermediate_size;
    const auto vocab_size = config.vocab_size;
    const auto rope_theta = config.rope_theta;
    const auto rms_norm_eps = config.rms_norm_eps;
    const auto head_dim = hidden_size / num_attention_heads;

    auto tokenizer = ondevai::custom::Tokenizer::load(args.vocab_bin_path->string());
    if (!tokenizer) {
        std::cerr << "error: failed to load tokenizer from: " << args.vocab_bin_path->string() << "\n";
        return 1;
    }

    auto tokens = tokenizer->encode(args.prompt, false);
    if (tokens.empty()) {
        std::cerr << "error: tokenizer produced no tokens for prompt: " << args.prompt << "\n";
        return 1;
    }
    const auto token_id = tokens[0];
    std::cout << "Prompt: \"" << args.prompt << "\" -> tokens: [" << token_id << ", ...]\n";
    std::cout << "Using first token: " << token_id << "\n";

    auto load_tensor = [&](const std::string& name) -> std::vector<float> {
        const auto result = ondevai::custom::load_fp16_tensor(*args.model_bin_path, *model_result.model, name);
        if (!result.ok) {
            std::cerr << "error: failed to load tensor " << name << ": " << result.error << "\n";
            throw std::runtime_error("tensor load failed");
        }
        return fp16_to_float32(result.fp16_data);
    };

    std::cout << "Loading tensors...\n";
    const auto embed_tokens_weight = load_tensor(std::string(kEmbedTokens));
    const auto input_layernorm_weight = load_tensor(std::string(kLayer0InputLayernorm));
    const auto q_proj_weight = load_tensor(std::string(kLayer0QProj));
    const auto k_proj_weight = load_tensor(std::string(kLayer0KProj));
    const auto v_proj_weight = load_tensor(std::string(kLayer0VProj));
    const auto o_proj_weight = load_tensor(std::string(kLayer0OProj));
    const auto post_attn_layernorm_weight = load_tensor(std::string(kLayer0PostAttnLayernorm));
    const auto gate_proj_weight = load_tensor(std::string(kLayer0MlpGateProj));
    const auto up_proj_weight = load_tensor(std::string(kLayer0MlpUpProj));
    const auto down_proj_weight = load_tensor(std::string(kLayer0MlpDownProj));
    const auto final_norm_weight = load_tensor(std::string(kFinalNorm));
    const auto lm_head_weight = load_tensor(std::string(kLmHead));

    std::cout << "Computing layer 0 forward pass...\n";

    std::vector<float> x(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
        x[i] = embed_tokens_weight[static_cast<std::size_t>(token_id) * hidden_size + i];
    }

    std::vector<float> norm1_out(hidden_size);
    if (!ondevai::custom::rmsnorm_reference(x, input_layernorm_weight, rms_norm_eps, norm1_out)) {
        std::cerr << "error: rmsnorm_reference failed for norm1\n";
        return 1;
    }

    std::vector<float> q(hidden_size);
    std::vector<float> k(num_kv_heads * head_dim);
    std::vector<float> v(num_kv_heads * head_dim);

    for (std::uint32_t r = 0; r < hidden_size; ++r) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < hidden_size; ++c) {
            sum += q_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * norm1_out[c];
        }
        q[r] = sum;
    }

    for (std::uint32_t r = 0; r < num_kv_heads * head_dim; ++r) {
        float sum_k = 0.0f;
        float sum_v = 0.0f;
        for (std::uint32_t c = 0; c < hidden_size; ++c) {
            sum_k += k_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * norm1_out[c];
            sum_v += v_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * norm1_out[c];
        }
        k[r] = sum_k;
        v[r] = sum_v;
    }

    std::vector<float> q_with_rope = q;
    std::vector<float> k_with_rope = k;
    if (!ondevai::custom::rope_reference(q_with_rope, 0, rope_theta, head_dim)) {
        std::cerr << "error: rope_reference failed for q\n";
        return 1;
    }
    if (!ondevai::custom::rope_reference(k_with_rope, 0, rope_theta, head_dim)) {
        std::cerr << "error: rope_reference failed for k\n";
        return 1;
    }

    std::vector<float> context(hidden_size, 0.0f);
    const std::uint32_t q_heads_per_kv = num_attention_heads / num_kv_heads;
    for (std::uint32_t qh = 0; qh < num_attention_heads; ++qh) {
        const std::uint32_t kv_head = qh / q_heads_per_kv;
        const std::size_t q_offset = static_cast<std::size_t>(qh) * head_dim;
        const std::size_t kv_offset = static_cast<std::size_t>(kv_head) * head_dim;
        for (std::uint32_t d = 0; d < head_dim; ++d) {
            context[q_offset + d] = v[kv_offset + d];
        }
    }

    std::vector<float> attn_out(hidden_size, 0.0f);
    for (std::uint32_t r = 0; r < hidden_size; ++r) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < hidden_size; ++c) {
            sum += o_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * context[c];
        }
        attn_out[r] = sum;
    }

    std::vector<float> residual1(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
        residual1[i] = x[i] + attn_out[i];
    }

    std::vector<float> norm2_out(hidden_size);
    if (!ondevai::custom::rmsnorm_reference(residual1, post_attn_layernorm_weight, rms_norm_eps, norm2_out)) {
        std::cerr << "error: rmsnorm_reference failed for norm2\n";
        return 1;
    }

    std::vector<float> gate(intermediate_size);
    std::vector<float> up(intermediate_size);
    for (std::uint32_t r = 0; r < intermediate_size; ++r) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        for (std::uint32_t c = 0; c < hidden_size; ++c) {
            sum_gate += gate_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * norm2_out[c];
            sum_up += up_proj_weight[static_cast<std::size_t>(r) * hidden_size + c] * norm2_out[c];
        }
        gate[r] = sum_gate;
        up[r] = sum_up;
    }

    std::vector<float> mlp_intermediate(intermediate_size);
    for (std::size_t i = 0; i < mlp_intermediate.size(); ++i) {
        mlp_intermediate[i] = silu(gate[i]) * up[i];
    }

    std::vector<float> mlp_out(hidden_size, 0.0f);
    for (std::uint32_t r = 0; r < hidden_size; ++r) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < intermediate_size; ++c) {
            sum += down_proj_weight[static_cast<std::size_t>(r) * intermediate_size + c] * mlp_intermediate[c];
        }
        mlp_out[r] = sum;
    }

    std::vector<float> layer_out(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
        layer_out[i] = residual1[i] + mlp_out[i];
    }

    std::vector<float> final_norm_out(hidden_size);
    if (!ondevai::custom::rmsnorm_reference(layer_out, final_norm_weight, rms_norm_eps, final_norm_out)) {
        std::cerr << "error: rmsnorm_reference failed for final_norm\n";
        return 1;
    }

    std::vector<float> logits(vocab_size, 0.0f);
    for (std::uint32_t r = 0; r < vocab_size; ++r) {
        float sum = 0.0f;
        for (std::uint32_t c = 0; c < hidden_size; ++c) {
            sum += lm_head_weight[static_cast<std::size_t>(r) * hidden_size + c] * final_norm_out[c];
        }
        logits[r] = sum;
    }

    std::cout << "Writing JSON output...\n";
    std::ofstream out(args.output_path->string());
    if (!out) {
        std::cerr << "error: cannot open output file: " << args.output_path->string() << "\n";
        return 1;
    }

    write_json_header(out, args.prompt, token_id, *args.model_bin_path, config);

    write_tensor_dump(out, "embedding", {hidden_size}, x, false);
    write_tensor_dump(out, "rmsnorm", {hidden_size}, norm1_out, false);
    write_tensor_dump(out, "q_projection", {hidden_size}, q, false);
    write_tensor_dump(out, "attention_output", {hidden_size}, attn_out, false);
    write_tensor_dump(out, "mlp_output", {hidden_size}, mlp_out, false);
    write_tensor_dump(out, "final_logits", {vocab_size}, logits, true);

    out << "}\n";
    out.close();

    std::cout << "SUCCESS: layer 0 forward pass completed\n";
    std::cout << "Output written to: " << args.output_path->string() << "\n";
    std::cout << "\nFirst few values of each tensor:\n";
    std::cout << "  embedding[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << x[i];
    std::cout << "\n";
    std::cout << "  rmsnorm[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << norm1_out[i];
    std::cout << "\n";
    std::cout << "  q_projection[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << q[i];
    std::cout << "\n";
    std::cout << "  attention_output[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << attn_out[i];
    std::cout << "\n";
    std::cout << "  mlp_output[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << mlp_out[i];
    std::cout << "\n";
    std::cout << "  final_logits[0:8]:";
    for (std::size_t i = 0; i < 8; ++i) std::cout << " " << std::setprecision(9) << logits[i];
    std::cout << "\n";

    return 0;
}