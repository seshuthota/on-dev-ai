#include "ondevai/custom/packed_tensor_reader.h"

#include <fstream>
#include <span>

namespace ondevai::custom {
namespace {

const TensorInfo* find_tensor(const Model& model, std::string_view name) {
    for (const auto& tensor : model.tensors()) {
        if (tensor.name == name) {
            return &tensor;
        }
    }
    return nullptr;
}

}  // namespace

TensorLoadResult load_fp16_tensor(
    const std::filesystem::path& model_bin_path,
    const Model& model,
    const std::string_view tensor_name) {
    TensorLoadResult result;

    const auto* tensor = find_tensor(model, tensor_name);
    if (!tensor) {
        result.error = "tensor not found: ";
        result.error += tensor_name;
        return result;
    }

    if (tensor->dtype != DType::fp16) {
        result.error = "tensor ";
        result.error += tensor_name;
        result.error += " is not fp16";
        return result;
    }

    if (tensor->byte_size % 2 != 0) {
        result.error = "tensor ";
        result.error += tensor_name;
        result.error += " has odd byte_size";
        return result;
    }

    std::ifstream file(model_bin_path, std::ios::binary);
    if (!file) {
        result.error = "cannot open file: " + model_bin_path.string();
        return result;
    }

    file.seekg(static_cast<std::streamoff>(tensor->offset));
    if (!file) {
        result.error = "cannot seek to tensor offset for " + std::string(tensor_name);
        return result;
    }

    const std::size_t element_count = tensor->byte_size / 2;
    result.fp16_data.resize(element_count);

    if (!file.read(reinterpret_cast<char*>(result.fp16_data.data()), tensor->byte_size)) {
        result.error = "failed to read tensor payload for " + std::string(tensor_name);
        result.fp16_data.clear();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace ondevai::custom
