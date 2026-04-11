#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/packed_tensor_reader.h"
#include "ondevai/custom/reference_kernels.h"
#include "ondevai/custom/tokenizer.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

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

constexpr const char* DEFAULT_MODEL_BIN = "/data/local/tmp/model.bin";
constexpr const char* DEFAULT_VOCAB_BIN = "/data/local/tmp/vocab.bin";
constexpr const char* DEFAULT_OUTPUT = "/data/local/tmp/android_tensors.json";
constexpr const char* DEFAULT_PROMPT = "Hello";

struct CliArgs {
    std::string model_bin_path = DEFAULT_MODEL_BIN;
    std::string vocab_bin_path = DEFAULT_VOCAB_BIN;
    std::string output_path = DEFAULT_OUTPUT;
    std::string prompt = DEFAULT_PROMPT;
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--model-bin" && i + 1 < argc) {
            args.model_bin_path = argv[++i];
        } else if (arg == "--vocab-bin" && i + 1 < argc) {
            args.vocab_bin_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::fprintf(stderr, "Usage: %s [OPTIONS]\n", argv[0]);
            std::fprintf(stderr, "  --model-bin PATH   (%s)\n", DEFAULT_MODEL_BIN);
            std::fprintf(stderr, "  --vocab-bin PATH   (%s)\n", DEFAULT_VOCAB_BIN);
            std::fprintf(stderr, "  --output PATH     (%s)\n", DEFAULT_OUTPUT);
            std::fprintf(stderr, "  --prompt TEXT      (%s)\n", DEFAULT_PROMPT);
            std::exit(0);
        }
    }
    return args;
}

float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

int main(int argc, char* argv[]) {
    const auto args = parse_args(argc, argv);

    std::fprintf(stderr, "Loading model from: %s\n", args.model_bin_path.c_str());
    const auto model_result = ondevai::custom::load_packed_model_metadata(args.model_bin_path);
    if (!model_result.ok) {
        std::fprintf(stderr, "ERROR: failed to load model: %s\n", model_result.error.c_str());
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

    std::fprintf(stderr, "Loading tokenizer from: %s\n", args.vocab_bin_path.c_str());
    auto tokenizer = ondevai::custom::Tokenizer::load(args.vocab_bin_path);
    if (!tokenizer) {
        std::fprintf(stderr, "ERROR: failed to load tokenizer\n");
        return 1;
    }

    auto tokens = tokenizer->encode(args.prompt, false);
    if (tokens.empty()) {
        std::fprintf(stderr, "ERROR: tokenizer produced no tokens\n");
        return 1;
    }
    const auto token_id = tokens[0];
    std::fprintf(stderr, "Prompt: \"%s\" -> token_id=%u\n", args.prompt.c_str(), token_id);

    auto load_tensor = [&](const std::string& name) -> std::vector<float> {
        const auto result = ondevai::custom::load_fp16_tensor(args.model_bin_path, *model_result.model, name);
        if (!result.ok) {
            throw std::runtime_error("failed to load tensor: " + name);
        }
        std::vector<float> out(result.fp16_data.size());
        for (std::size_t i = 0; i < result.fp16_data.size(); ++i) {
            out[i] = ondevai::custom::fp16_to_float(result.fp16_data[i]);
        }
        return out;
    };

    std::fprintf(stderr, "Loading tensors...\n");
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

    std::fprintf(stderr, "Computing layer 0 forward pass...\n");

    std::vector<float> x(hidden_size);
    for (std::size_t i = 0; i < hidden_size; ++i) {
        x[i] = embed_tokens_weight[static_cast<std::size_t>(token_id) * hidden_size + i];
    }

    std::vector<float> norm1_out(hidden_size);
    if (!ondevai::custom::rmsnorm_reference(x, input_layernorm_weight, rms_norm_eps, norm1_out)) {
        std::fprintf(stderr, "ERROR: rmsnorm_reference failed for norm1\n");
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
        std::fprintf(stderr, "ERROR: rope_reference failed for q\n");
        return 1;
    }
    if (!ondevai::custom::rope_reference(k_with_rope, 0, rope_theta, head_dim)) {
        std::fprintf(stderr, "ERROR: rope_reference failed for k\n");
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
        std::fprintf(stderr, "ERROR: rmsnorm_reference failed for norm2\n");
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
        std::fprintf(stderr, "ERROR: rmsnorm_reference failed for final_norm\n");
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

    std::fprintf(stderr, "Writing output to: %s\n", args.output_path.c_str());
    FILE* out = fopen(args.output_path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "ERROR: cannot open output file: %s\n", args.output_path.c_str());
        return 1;
    }

    fprintf(out, "{\n");
    fprintf(out, "  \"prompt\": \"%s\",\n", args.prompt.c_str());
    fprintf(out, "  \"token_id\": %u,\n", token_id);
    fprintf(out, "  \"model_bin\": \"%s\",\n", args.model_bin_path.c_str());
    fprintf(out, "  \"config\": {\n");
    fprintf(out, "    \"hidden_size\": %u,\n", hidden_size);
    fprintf(out, "    \"intermediate_size\": %u,\n", intermediate_size);
    fprintf(out, "    \"num_attention_heads\": %u,\n", num_attention_heads);
    fprintf(out, "    \"num_key_value_heads\": %u,\n", num_kv_heads);
    fprintf(out, "    \"vocab_size\": %u,\n", vocab_size);
    fprintf(out, "    \"rope_theta\": %f,\n", rope_theta);
    fprintf(out, "    \"rms_norm_eps\": %f\n", rms_norm_eps);
    fprintf(out, "  },\n");

    auto write_tensor = [&](const char* key, const float* data, std::size_t size, bool is_last) {
        fprintf(out, "  \"%s\": {\"shape\": [%zu], \"values\": [", key, size);
        for (std::size_t i = 0; i < size; ++i) {
            if (i > 0) fprintf(out, ", ");
            fprintf(out, "%.17g", data[i]);
        }
        fprintf(out, "]}%s\n", is_last ? "" : ",");
    };

    write_tensor("embedding", x.data(), x.size(), false);
    write_tensor("rmsnorm", norm1_out.data(), norm1_out.size(), false);
    write_tensor("q_projection", q.data(), q.size(), false);
    write_tensor("attention_output", attn_out.data(), attn_out.size(), false);
    write_tensor("mlp_output", mlp_out.data(), mlp_out.size(), false);
    write_tensor("final_logits", logits.data(), logits.size(), true);

    fprintf(out, "}\n");
    fclose(out);

    std::fprintf(stderr, "SUCCESS: output written to %s\n", args.output_path.c_str());

    std::fprintf(stderr, "\nFirst few values:\n");
    std::fprintf(stderr, "  embedding[0:4]:");
    for (std::size_t i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", x[i]);
    std::fprintf(stderr, "\n  rmsnorm[0:4]:");
    for (std::size_t i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", norm1_out[i]);
    std::fprintf(stderr, "\n  q_projection[0:4]:");
    for (std::size_t i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", q[i]);
    std::fprintf(stderr, "\n  logits[0:4]:");
    for (std::size_t i = 0; i < 4; ++i) std::fprintf(stderr, " %.9g", logits[i]);
    std::fprintf(stderr, "\n");

    return 0;
}