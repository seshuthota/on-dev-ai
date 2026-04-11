#include "ondevai/custom/packed_model_reader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ondevai::custom {
namespace {

constexpr std::string_view kMagic = "ODAI";
constexpr std::uint32_t kExpectedVersion = 1;
constexpr std::uint32_t kExpectedModelFamily = 1;
constexpr std::string_view kExpectedModelFamilyName = "tinyllama_v1";
constexpr std::string_view kExpectedModelId = "TinyLlama-1.1B-Chat-v1.0";
constexpr std::size_t kHeaderSize = 80;
constexpr std::size_t kMaxRank = 8;
constexpr std::uint64_t kDirectoryEntryFixedBytes = 20;

struct FileHeader {
    char magic[4];
    std::uint32_t version;
    std::uint32_t model_family;
    std::uint32_t hidden_size;
    std::uint32_t intermediate_size;
    std::uint32_t num_hidden_layers;
    std::uint32_t num_attention_heads;
    std::uint32_t num_key_value_heads;
    std::uint32_t vocab_size;
    std::uint32_t max_position_embeddings;
    std::uint32_t context_cap;
    std::uint32_t tensor_count;
    float rope_theta;
    float rms_norm_eps;
    std::uint64_t tensor_directory_offset;
    std::uint64_t tensor_data_offset;
    std::uint64_t file_size;
};

static_assert(sizeof(FileHeader) == kHeaderSize, "FileHeader size must equal kHeaderSize");

bool validate_header(const FileHeader& header, std::uint64_t actual_file_size) {
    if (std::memcmp(header.magic, kMagic.data(), 4) != 0) {
        return false;
    }
    if (header.version != kExpectedVersion) {
        return false;
    }
    if (header.model_family != kExpectedModelFamily) {
        return false;
    }
    if (header.file_size != actual_file_size) {
        return false;
    }
    if (header.tensor_directory_offset < kHeaderSize) {
        return false;
    }
    if (header.tensor_data_offset < header.tensor_directory_offset) {
        return false;
    }
    if (header.tensor_directory_offset > actual_file_size) {
        return false;
    }
    if (header.tensor_data_offset > actual_file_size) {
        return false;
    }
    return true;
}

bool has_remaining_directory_bytes(std::ifstream& file, const std::uint64_t directory_end, const std::uint64_t byte_count) {
    const auto pos = file.tellg();
    if (pos < 0) {
        return false;
    }
    const auto current_pos = static_cast<std::uint64_t>(pos);
    return current_pos <= directory_end && byte_count <= directory_end - current_pos;
}

bool expected_tensor_byte_size(const DType dtype, const std::vector<std::uint32_t>& shape, std::uint64_t& byte_size) {
    std::uint64_t element_count = 1;
    for (const auto dim : shape) {
        if (dim == 0 || element_count > std::numeric_limits<std::uint64_t>::max() / dim) {
            return false;
        }
        element_count *= dim;
    }

    const auto element_size = static_cast<std::uint64_t>(element_size_bytes(dtype));
    if (element_size == 0 || element_count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        return false;
    }

    byte_size = element_count * element_size;
    return true;
}

std::vector<std::string> required_tensors() {
    std::vector<std::string> names = {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    };
    const std::string prefix = "model.layers.0";
    names.insert(names.end(), {
        prefix + ".input_layernorm.weight",
        prefix + ".self_attn.q_proj.weight",
        prefix + ".self_attn.k_proj.weight",
        prefix + ".self_attn.v_proj.weight",
        prefix + ".self_attn.o_proj.weight",
        prefix + ".post_attention_layernorm.weight",
        prefix + ".mlp.gate_proj.weight",
        prefix + ".mlp.up_proj.weight",
        prefix + ".mlp.down_proj.weight",
    });
    return names;
}

bool validate_tensor_shapes(
    const std::vector<TensorInfo>& tensors,
    std::uint32_t hidden_size,
    std::uint32_t intermediate_size,
    std::uint32_t vocab_size,
    std::uint32_t attention_head_count,
    std::uint32_t kv_head_count
) {
    if (hidden_size == 0 || attention_head_count == 0 || hidden_size % attention_head_count != 0) {
        return false;
    }
    const std::uint32_t head_dim = hidden_size / attention_head_count;

    auto find_tensor = [&](const std::string& name) -> const TensorInfo* {
        for (const auto& t : tensors) {
            if (t.name == name) return &t;
        }
        return nullptr;
    };

    auto check_shape_1d = [&](const std::string& name, std::uint32_t expected_dim) -> bool {
        const auto* t = find_tensor(name);
        if (!t) return false;
        return t->shape.size() == 1 && t->shape[0] == expected_dim;
    };

    auto check_shape_2d = [&](const std::string& name, std::uint32_t dim0, std::uint32_t dim1) -> bool {
        const auto* t = find_tensor(name);
        if (!t) return false;
        return t->shape.size() == 2 && t->shape[0] == dim0 && t->shape[1] == dim1;
    };

    if (!check_shape_2d("model.embed_tokens.weight", vocab_size, hidden_size)) return false;
    if (!check_shape_2d("lm_head.weight", vocab_size, hidden_size)) return false;
    if (!check_shape_1d("model.norm.weight", hidden_size)) return false;
    if (!check_shape_1d("model.layers.0.input_layernorm.weight", hidden_size)) return false;
    if (!check_shape_1d("model.layers.0.post_attention_layernorm.weight", hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.self_attn.q_proj.weight", hidden_size, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.self_attn.o_proj.weight", hidden_size, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.self_attn.k_proj.weight", kv_head_count * head_dim, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.self_attn.v_proj.weight", kv_head_count * head_dim, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.mlp.gate_proj.weight", intermediate_size, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.mlp.up_proj.weight", intermediate_size, hidden_size)) return false;
    if (!check_shape_2d("model.layers.0.mlp.down_proj.weight", hidden_size, intermediate_size)) return false;

    return true;
}

}  // namespace

PackedModelLoadResult load_packed_model_metadata(const std::filesystem::path& model_bin_path) {
    PackedModelLoadResult result;

    if (!std::filesystem::exists(model_bin_path)) {
        result.error = "file not found: " + model_bin_path.string();
        return result;
    }

    std::uint64_t actual_file_size = 0;
    try {
        actual_file_size = std::filesystem::file_size(model_bin_path);
    } catch (const std::exception& e) {
        result.error = "cannot read file size: " + std::string(e.what());
        return result;
    }

    if (actual_file_size < kHeaderSize) {
        result.error = "file too small for header: " + std::to_string(actual_file_size) + " < " + std::to_string(kHeaderSize);
        return result;
    }

    std::ifstream file(model_bin_path, std::ios::binary);
    if (!file) {
        result.error = "cannot open file: " + model_bin_path.string();
        return result;
    }

    FileHeader header;
    if (!file.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        result.error = "failed to read header";
        return result;
    }

    if (!validate_header(header, actual_file_size)) {
        if (std::memcmp(header.magic, kMagic.data(), 4) != 0) {
            result.error = "invalid magic";
        } else if (header.version != kExpectedVersion) {
            result.error = "unsupported version: " + std::to_string(header.version);
        } else if (header.model_family != kExpectedModelFamily) {
            result.error = "unsupported model family: " + std::to_string(header.model_family);
        } else if (header.file_size != actual_file_size) {
            result.error = "file_size mismatch: header=" + std::to_string(header.file_size) + " actual=" + std::to_string(actual_file_size);
        } else if (header.tensor_directory_offset > actual_file_size) {
            result.error = "tensor_directory_offset exceeds file size";
        } else if (header.tensor_data_offset > actual_file_size) {
            result.error = "tensor_data_offset exceeds file size";
        } else if (header.tensor_data_offset < header.tensor_directory_offset) {
            result.error = "tensor_data_offset < tensor_directory_offset";
        } else {
            result.error = "header validation failed";
        }
        return result;
    }

    if (header.num_attention_heads == 0 || header.hidden_size == 0 || header.hidden_size % header.num_attention_heads != 0) {
        result.error = "invalid attention head configuration: hidden_size=" + std::to_string(header.hidden_size) + " num_attention_heads=" + std::to_string(header.num_attention_heads);
        return result;
    }

    if (header.tensor_count == 0 || header.tensor_count > 16384) {
        result.error = "invalid tensor_count: " + std::to_string(header.tensor_count);
        return result;
    }

    file.seekg(static_cast<std::streamoff>(header.tensor_directory_offset));
    if (!file) {
        result.error = "cannot seek to tensor directory";
        return result;
    }

    std::vector<TensorInfo> tensors;
    tensors.reserve(header.tensor_count);

    std::uint64_t directory_end = header.tensor_data_offset;
    for (std::uint32_t i = 0; i < header.tensor_count; ++i) {
        {
            std::uint64_t current_pos = static_cast<std::uint64_t>(file.tellg());
            if (current_pos >= directory_end) {
                result.error = "tensor directory overflow at index " + std::to_string(i);
                return result;
            }
        }

        std::uint16_t name_len = 0;
        if (!has_remaining_directory_bytes(file, directory_end, sizeof(name_len))) {
            result.error = "name length would overflow directory at tensor " + std::to_string(i);
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len))) {
            result.error = "failed to read name length for tensor " + std::to_string(i);
            return result;
        }

        if (!has_remaining_directory_bytes(file, directory_end, name_len)) {
            result.error = "name would overflow directory at tensor " + std::to_string(i);
            return result;
        }

        std::string name(name_len, '\0');
        if (!file.read(name.data(), name_len)) {
            result.error = "failed to read name for tensor " + std::to_string(i);
            return result;
        }

        std::uint8_t dtype_val = 0;
        std::uint8_t rank = 0;
        std::uint16_t reserved = 0;
        std::uint64_t offset = 0;
        std::uint64_t byte_size = 0;

        if (!has_remaining_directory_bytes(file, directory_end, kDirectoryEntryFixedBytes)) {
            result.error = "fixed tensor metadata would overflow directory at tensor " + std::to_string(i);
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&dtype_val), sizeof(dtype_val))) {
            result.error = "failed to read dtype for tensor " + std::to_string(i);
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&rank), sizeof(rank))) {
            result.error = "failed to read rank for tensor " + std::to_string(i);
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&reserved), sizeof(reserved))) {
            result.error = "failed to read reserved for tensor " + std::to_string(i);
            return result;
        }
        if (reserved != 0) {
            result.error = "nonzero reserved field in tensor " + name;
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&offset), sizeof(offset))) {
            result.error = "failed to read offset for tensor " + std::to_string(i);
            return result;
        }
        if (!file.read(reinterpret_cast<char*>(&byte_size), sizeof(byte_size))) {
            result.error = "failed to read byte_size for tensor " + std::to_string(i);
            return result;
        }

        DType dtype = DType::fp16;
        if (dtype_val == 1) {
            dtype = DType::fp16;
        } else if (dtype_val == 2) {
            dtype = DType::fp32;
        } else {
            result.error = "unknown dtype " + std::to_string(static_cast<int>(dtype_val)) + " for tensor " + name;
            return result;
        }

        if (rank == 0 || rank > kMaxRank) {
            result.error = "invalid rank " + std::to_string(static_cast<int>(rank)) + " for tensor " + name;
            return result;
        }

        const auto shape_byte_count = static_cast<std::uint64_t>(rank) * sizeof(std::uint32_t);
        if (!has_remaining_directory_bytes(file, directory_end, shape_byte_count)) {
            result.error = "shape would overflow directory for tensor " + name;
            return result;
        }
        std::vector<std::uint32_t> shape(rank);
        for (std::uint8_t d = 0; d < rank; ++d) {
            if (!file.read(reinterpret_cast<char*>(&shape[d]), sizeof(shape[d]))) {
                result.error = "failed to read shape dimension for tensor " + name;
                return result;
            }
        }

        {
            std::uint64_t current_pos = static_cast<std::uint64_t>(file.tellg());
            if (current_pos > directory_end) {
                result.error = "entry overflow after reading tensor " + name;
                return result;
            }
        }

        if (offset > actual_file_size) {
            result.error = "offset out of bounds for tensor " + name + ": " + std::to_string(offset) + " > " + std::to_string(actual_file_size);
            return result;
        }
        if (byte_size > actual_file_size) {
            result.error = "byte_size out of bounds for tensor " + name + ": " + std::to_string(byte_size) + " > " + std::to_string(actual_file_size);
            return result;
        }
        if (offset > actual_file_size - byte_size) {
            result.error = "payload overflow for tensor " + name + ": " + std::to_string(offset) + " + " + std::to_string(byte_size) + " > " + std::to_string(actual_file_size);
            return result;
        }

        if (offset < header.tensor_data_offset) {
            result.error = "tensor " + name + " payload offset " + std::to_string(offset) + " is before tensor_data_offset " + std::to_string(header.tensor_data_offset);
            return result;
        }

        std::uint64_t expected_byte_size = 0;
        if (!expected_tensor_byte_size(dtype, shape, expected_byte_size)) {
            result.error = "invalid shape byte-size product for tensor " + name;
            return result;
        }
        if (expected_byte_size != byte_size) {
            result.error = "byte_size mismatch for tensor " + name + ": expected " + std::to_string(expected_byte_size) + ", got " + std::to_string(byte_size);
            return result;
        }

        tensors.push_back({
            .name = std::move(name),
            .dtype = dtype,
            .shape = std::move(shape),
            .offset = offset,
            .byte_size = byte_size,
        });
    }

    {
        std::uint64_t current_pos = static_cast<std::uint64_t>(file.tellg());
        if (current_pos > directory_end) {
            result.error = "tensor directory exceeds tensor_data_offset after parsing";
            return result;
        }
    }

    const auto required = required_tensors();
    for (const auto& req : required) {
        if (std::none_of(tensors.begin(), tensors.end(), [&req](const TensorInfo& t) { return t.name == req; })) {
            result.error = "missing required tensor: " + req;
            return result;
        }
    }

    if (!validate_tensor_shapes(
            tensors,
            header.hidden_size,
            header.intermediate_size,
            header.vocab_size,
            header.num_attention_heads,
            header.num_key_value_heads)) {
        result.error = "tensor shape validation failed for layer 0 tensors";
        return result;
    }

    if (tensors.size() != header.tensor_count) {
        result.error = "tensor count mismatch: read " + std::to_string(tensors.size()) + " vs header " + std::to_string(header.tensor_count);
        return result;
    }

    ModelConfig config;
    config.model_id = std::string(kExpectedModelId);
    config.model_family = std::string(kExpectedModelFamilyName);
    config.hidden_size = header.hidden_size;
    config.intermediate_size = header.intermediate_size;
    config.layer_count = header.num_hidden_layers;
    config.attention_head_count = header.num_attention_heads;
    config.kv_head_count = header.num_key_value_heads;
    config.vocab_size = header.vocab_size;
    config.max_position_embeddings = header.max_position_embeddings;
    config.runtime_context_cap = header.context_cap;
    config.rope_theta = header.rope_theta;
    config.rms_norm_eps = header.rms_norm_eps;

    auto model = std::make_unique<Model>(std::move(config));
    for (auto& tensor : tensors) {
        model->add_tensor(std::move(tensor));
    }

    result.ok = true;
    result.model = std::move(model);
    result.file_size = actual_file_size;
    return result;
}

}  // namespace ondevai::custom
