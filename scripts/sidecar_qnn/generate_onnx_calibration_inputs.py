#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx


def _dim_value(dim, symbol_map):
    if dim.HasField("dim_value") and dim.dim_value > 0:
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return int(symbol_map.get(dim.dim_param, 1))
    return 1


def _shape_for_input(value_info, symbol_map):
    t = value_info.type.tensor_type
    return [_dim_value(d, symbol_map) for d in t.shape.dim]


def _dtype_for_elem_type(elem_type: int):
    if elem_type == 1:
        return np.float32
    if elem_type == 6:
        return np.int32
    if elem_type == 7:
        return np.int64
    return np.float32


def _tensor_for_input(name: str, dtype, shape):
    if len(shape) == 0:
        if np.issubdtype(dtype, np.integer):
            return np.array(1, dtype=dtype)
        return np.array(0.0, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        if "attention_mask" in name:
            return np.ones(shape, dtype=dtype)
        if "position_ids" in name:
            return np.zeros(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a minimal calibration input-list + .npy tensors from an ONNX model signature."
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--out-dir", required=True, help="Output dir for .npy files and input_list.txt")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1)
    parser.add_argument("--past-sequence-length", type=int, default=1)
    parser.add_argument("--total-sequence-length", type=int, default=1)
    parser.add_argument("--num-logits-to-keep", type=int, default=1)
    args = parser.parse_args()

    onnx_path = Path(args.onnx).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_map = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "past_sequence_length": args.past_sequence_length,
        "total_sequence_length": args.total_sequence_length,
        "num_logits_to_keep": args.num_logits_to_keep,
    }

    model = onnx.load(str(onnx_path), load_external_data=False)
    pairs = []

    for idx, inp in enumerate(model.graph.input):
        name = inp.name
        shape = _shape_for_input(inp, symbol_map)
        dtype = _dtype_for_elem_type(inp.type.tensor_type.elem_type)
        arr = _tensor_for_input(name, dtype, shape)
        npy_path = out_dir / f"{idx:03d}_{name.replace('/', '_').replace(':', '_')}.npy"
        np.save(npy_path, arr)
        pairs.append((name, npy_path))

    input_list_path = out_dir / "input_list.txt"
    with input_list_path.open("w", encoding="utf-8") as f:
        line = " ".join(f"{name}:={path}" for name, path in pairs)
        f.write(line + "\n")

    print(f"[info] generated_inputs={len(pairs)}")
    print(f"[info] input_list={input_list_path}")


if __name__ == "__main__":
    main()
