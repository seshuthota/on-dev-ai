#include "ondevai/custom/packed_model_reader.h"
#include "ondevai/custom/packed_tensor_reader.h"
#include "ondevai/custom/reference_kernels.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kLayer0InputLayernormWeight = "model.layers.0.input_layernorm.weight";

struct CliArgs {
    bool help = false;
    std::optional<std::filesystem::path> model_bin_path;
    std::optional<std::filesystem::path> output_path;
    std::string error;
};

void print_usage() {
    std::cout << "Usage: run_layer0_rmsnorm --model-bin PATH --output PATH\n";
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
    if (!args.output_path) {
        args.error = "missing required --output argument";
    }
    return args;
}

std::vector<float> generate_deterministic_input(std::uint32_t hidden_size) {
    std::vector<float> input(hidden_size);
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        input[i] = 0.25f * std::sin(0.013f * static_cast<float>(i)) +
                   0.1f * std::cos(0.007f * static_cast<float>(i)) +
                   0.001f * static_cast<float>(i % 17);
    }
    return input;
}

constexpr std::array<std::size_t, 10> kSelectedIndices = {0, 1, 2, 3, 31, 127, 511, 1024, 1536, 2047};

void write_output_json(
    const std::filesystem::path& model_bin_path,
    const std::string& tensor_name,
    std::uint32_t hidden_size,
    float eps,
    const std::vector<float>& output,
    const std::filesystem::path& output_path) {
    std::ofstream out(output_path);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + output_path.string());
    }

    out << std::setprecision(17);
    out << "{\n";
    out << "  \"model_bin\": \"" << model_bin_path.string() << "\",\n";
    out << "  \"tensor_name\": \"" << tensor_name << "\",\n";
    out << "  \"hidden_size\": " << hidden_size << ",\n";
    out << "  \"eps\": " << eps << ",\n";
    out << "  \"selected_indices\": [";
    for (std::size_t i = 0; i < kSelectedIndices.size(); ++i) {
        out << kSelectedIndices[i];
        if (i + 1 < kSelectedIndices.size()) out << ", ";
    }
    out << "],\n";
    out << "  \"output_values\": [";
    for (std::size_t i = 0; i < kSelectedIndices.size(); ++i) {
        out << output[kSelectedIndices[i]];
        if (i + 1 < kSelectedIndices.size()) out << ", ";
    }
    out << "]\n";
    out << "}\n";
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

    const auto result = ondevai::custom::load_packed_model_metadata(*args.model_bin_path);
    if (!result.ok) {
        std::cerr << "error: " << result.error << "\n";
        return 1;
    }

    const auto& config = result.model->config();
    const auto hidden_size = config.hidden_size;
    const auto eps = config.rms_norm_eps;

    const auto weight_result = ondevai::custom::load_fp16_tensor(
        *args.model_bin_path,
        *result.model,
        kLayer0InputLayernormWeight);
    if (!weight_result.ok) {
        std::cerr << "error: " << weight_result.error << "\n";
        return 1;
    }

    if (weight_result.fp16_data.size() != hidden_size) {
        std::cerr << "error: weight tensor size " << weight_result.fp16_data.size()
                  << " != hidden_size " << hidden_size << "\n";
        return 1;
    }

    constexpr std::size_t kLargestSelectedIndex = kSelectedIndices.back();
    if (kLargestSelectedIndex >= hidden_size) {
        std::cerr << "error: largest selected index " << kLargestSelectedIndex
                  << " >= hidden_size " << hidden_size << "\n";
        return 1;
    }

    std::vector<float> weight_float(hidden_size);
    for (std::uint32_t i = 0; i < hidden_size; ++i) {
        weight_float[i] = ondevai::custom::fp16_to_float(weight_result.fp16_data[i]);
    }

    const auto input = generate_deterministic_input(hidden_size);

    std::vector<float> output(hidden_size, 0.0f);
    if (!ondevai::custom::rmsnorm_reference(input, weight_float, eps, output)) {
        std::cerr << "error: rmsnorm_reference failed\n";
        return 1;
    }

    try {
        write_output_json(*args.model_bin_path, std::string(kLayer0InputLayernormWeight),
                         hidden_size, eps, output, *args.output_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "SUCCESS: layer0 input RMSNorm completed\n";
    std::cout << "output written to: " << args.output_path->string() << "\n";
    std::cout << "tensor: " << kLayer0InputLayernormWeight << "\n";
    std::cout << "hidden_size: " << hidden_size << ", eps: " << eps << "\n";
    std::cout << "selected outputs:";
    for (const auto idx : kSelectedIndices) {
        std::cout << " " << idx << "=" << std::setprecision(9) << output[idx];
    }
    std::cout << "\n";

    return 0;
}
