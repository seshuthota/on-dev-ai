#include "ondevai/custom/packed_model_reader.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct CliArgs {
    bool help = false;
    std::optional<std::filesystem::path> model_bin_path;
    std::string error;
};

void print_usage() {
    std::cout << "Usage: run_reference --model-bin PATH\n";
}

std::string dtype_string(ondevai::custom::DType dtype) {
    using ondevai::custom::DType;
    switch (dtype) {
        case DType::fp16:
            return "fp16";
        case DType::fp32:
            return "fp32";
    }
    return "unknown";
}

std::string shape_string(const std::vector<std::uint32_t>& shape) {
    if (shape.empty()) {
        return "[]";
    }
    std::string s = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            s += ",";
        }
    }
    s += "]";
    return s;
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
    return args;
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
    const auto& tensors = result.model->tensors();

    std::cout << "model_id: " << config.model_id << "\n";
    std::cout << "model_family: " << config.model_family << "\n";
    std::cout << "hidden_size: " << config.hidden_size << "\n";
    std::cout << "layer_count: " << config.layer_count << "\n";
    std::cout << "context_cap: " << config.runtime_context_cap << "\n";
    std::cout << "tensor_count: " << tensors.size() << "\n";
    std::cout << "file_size: " << result.file_size << "\n";
    std::cout << "\n";

    for (const auto& tensor : tensors) {
        std::cout << tensor.name << " " << dtype_string(tensor.dtype) << " " << shape_string(tensor.shape) << " offset=" << tensor.offset << " bytes=" << tensor.byte_size << "\n";
    }

    return 0;
}
