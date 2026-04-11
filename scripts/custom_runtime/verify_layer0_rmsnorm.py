#!/usr/bin/env python3
"""
Independent Python verifier for layer-0 RMSNorm output.
Parses the same packed model.bin format and computes RMSNorm in Python.
Compares against C++ output with tolerance.
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path


MAGIC = b"ODAI"
HEADER_SIZE = 80
DIRECTORY_ENTRY_FIXED_BYTES = 20
DTYPE_FP16 = 1
DTYPE_FP32 = 2


def parse_header(f):
    data = f.read(HEADER_SIZE)
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small for header: {len(data)} < {HEADER_SIZE}")

    magic = data[0:4]
    if magic != MAGIC:
        raise ValueError(f"Invalid magic: {magic!r}")

    (
        version,
        model_family,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        vocab_size,
        max_position_embeddings,
        context_cap,
        tensor_count,
        rope_theta,
        rms_norm_eps,
        tensor_directory_offset,
        tensor_data_offset,
        file_size,
    ) = struct.unpack("<IIIIIIIIIIIffQQQ", data[4:])

    if version != 1:
        raise ValueError(f"Unsupported version: {version}")
    if model_family != 1:
        raise ValueError(f"Unsupported model family: {model_family}")

    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "context_cap": context_cap,
        "tensor_count": tensor_count,
        "rope_theta": rope_theta,
        "rms_norm_eps": rms_norm_eps,
        "tensor_directory_offset": tensor_directory_offset,
        "tensor_data_offset": tensor_data_offset,
        "file_size": file_size,
    }


def parse_tensor_directory(f, header, num_tensors):
    tensors = []
    f.seek(header["tensor_directory_offset"])

    for _ in range(num_tensors):
        name_len = struct.unpack("<H", f.read(2))[0]
        name = f.read(name_len).decode("utf-8")

        dtype_val, rank, reserved = struct.unpack("<BBH", f.read(4))
        if reserved != 0:
            raise ValueError(f"Non-zero reserved in tensor {name}")

        offset, byte_size = struct.unpack("<QQ", f.read(16))

        shape = list(struct.unpack(f"<{rank}I", f.read(4 * rank)))

        dtype_map = {DTYPE_FP16: "fp16", DTYPE_FP32: "fp32"}
        dtype_name = dtype_map.get(dtype_val, f"unknown({dtype_val})")

        tensors.append(
            {
                "name": name,
                "dtype": dtype_name,
                "dtype_val": dtype_val,
                "rank": rank,
                "shape": shape,
                "offset": offset,
                "byte_size": byte_size,
            }
        )

    return tensors


def fp16_to_float(bits):
    sign = (bits >> 15) & 0x1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF

    if exp == 0:
        if frac == 0:
            return -0.0 if sign else 0.0
        return ((-1) ** sign) * (frac / 1024.0) * (2**-14)
    if exp == 31:
        if frac == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")

    ieee_exp = exp - 15
    ieee_mantissa = frac << 13
    ieee_bits = (sign << 31) | ((ieee_exp + 127) << 23) | ieee_mantissa

    return struct.unpack("<f", struct.pack("<I", ieee_bits))[0]


def load_fp16_tensor(f, tensor_info):
    if tensor_info["dtype_val"] != DTYPE_FP16:
        raise ValueError(f"Tensor {tensor_info['name']} is not fp16")

    if tensor_info["byte_size"] % 2 != 0:
        raise ValueError(f"Tensor {tensor_info['name']} has odd byte_size")

    f.seek(tensor_info["offset"])
    data = f.read(tensor_info["byte_size"])

    if len(data) != tensor_info["byte_size"]:
        raise ValueError(f"Failed to read tensor payload for {tensor_info['name']}")

    num_elements = len(data) // 2
    result = []
    for i in range(num_elements):
        bits = struct.unpack("<H", data[i * 2 : (i + 1) * 2])[0]
        result.append(fp16_to_float(bits))

    return result


def generate_deterministic_input(hidden_size):
    inp = []
    for i in range(hidden_size):
        val = 0.25 * math.sin(0.013 * i) + 0.1 * math.cos(0.007 * i) + 0.001 * (i % 17)
        inp.append(val)
    return inp


def rmsnorm_python(input_vec, weight, eps):
    mean_sq = sum(v * v for v in input_vec) / len(input_vec)
    scale = 1.0 / math.sqrt(mean_sq + eps)

    output = [input_vec[i] * scale * weight[i] for i in range(len(input_vec))]
    return output


def main():
    parser = argparse.ArgumentParser(description="Verify layer-0 RMSNorm output")
    parser.add_argument(
        "--model-bin", required=True, type=Path, help="Path to model.bin"
    )
    parser.add_argument(
        "--cpp-output", required=True, type=Path, help="Path to C++ output JSON"
    )
    args = parser.parse_args()

    if not args.model_bin.exists():
        print(f"FAIL: model.bin not found: {args.model_bin}")
        return 1

    if not args.cpp_output.exists():
        print(f"FAIL: cpp_output not found: {args.cpp_output}")
        return 1

    with open(args.model_bin, "rb") as f:
        header = parse_header(f)
        tensors = parse_tensor_directory(f, header, header["tensor_count"])

    tensor_map = {t["name"]: t for t in tensors}

    target_tensor = "model.layers.0.input_layernorm.weight"
    if target_tensor not in tensor_map:
        print(f"FAIL: tensor {target_tensor} not found in model.bin")
        return 1

    with open(args.model_bin, "rb") as f:
        weight = load_fp16_tensor(f, tensor_map[target_tensor])

    hidden_size = header["hidden_size"]
    if len(weight) != hidden_size:
        print(f"FAIL: weight size mismatch: {len(weight)} != {hidden_size}")
        return 1

    eps = header["rms_norm_eps"]
    input_vec = generate_deterministic_input(hidden_size)
    output = rmsnorm_python(input_vec, weight, eps)

    with open(args.cpp_output, "r") as f:
        cpp_data = json.load(f)

    if cpp_data.get("hidden_size") != hidden_size:
        print(
            f"FAIL: hidden_size mismatch: {cpp_data.get('hidden_size')} != {hidden_size}"
        )
        return 1

    if abs(cpp_data.get("eps", 0) - eps) > 1e-10:
        print(f"FAIL: eps mismatch: {cpp_data.get('eps')} != {eps}")
        return 1

    selected_indices = cpp_data.get("selected_indices", [])
    cpp_values = cpp_data.get("output_values", [])

    if len(selected_indices) != len(cpp_values):
        print(f"FAIL: selected_indices and output_values length mismatch")
        return 1

    tolerance = 2e-5
    max_diff = 0.0
    max_idx = 0
    all_pass = True

    for idx, cpp_val in zip(selected_indices, cpp_values):
        py_val = output[idx]
        diff = abs(py_val - cpp_val)
        if diff > max_diff:
            max_diff = diff
            max_idx = idx
        if diff > tolerance:
            print(
                f"FAIL: mismatch at index {idx}: C++={cpp_val:.15e} Python={py_val:.15e} diff={diff:.15e}"
            )
            all_pass = False

    if all_pass:
        print(
            f"PASS: all {len(selected_indices)} values match within tolerance {tolerance}"
        )
        print(f"      max_diff={max_diff:.15e} at index {max_idx}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
