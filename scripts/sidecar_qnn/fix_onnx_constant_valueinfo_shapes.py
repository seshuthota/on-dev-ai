#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper


def _save_model(model, output_path: Path):
    data_file = f"{output_path.name}.data"
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file,
        size_threshold=1024,
        convert_attribute=False,
    )


def _dims_from_vi(value_info):
    return [d.dim_value if d.HasField("dim_value") else None for d in value_info.type.tensor_type.shape.dim]


def _set_vi_dims(value_info, dims):
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for d in dims:
        dim = shape.dim.add()
        if d is None:
            dim.dim_param = "unk"
        else:
            dim.dim_value = int(d)


def main():
    parser = argparse.ArgumentParser(
        description="Fix value_info shapes for Constant-node outputs when they mismatch literal tensor shape."
    )
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", default="", help="Optional output path; default overwrites input.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path

    model = onnx.load(str(input_path))
    vi_map = {v.name: v for v in model.graph.value_info}

    fixes = 0
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        out_name = node.output[0]
        vi = vi_map.get(out_name)
        if vi is None:
            continue

        tensor = None
        for attr in node.attribute:
            if attr.name == "value":
                tensor = attr.t
                break
        if tensor is None:
            continue

        arr = numpy_helper.to_array(tensor)
        actual = list(arr.shape)
        declared = _dims_from_vi(vi)
        if all(d is not None for d in declared):
            declared_int = [int(d) for d in declared]
            if declared_int != actual:
                _set_vi_dims(vi, actual)
                fixes += 1

    _save_model(model, output_path)
    print(f"[info] wrote {output_path}")
    print(f"[info] constant_valueinfo_shape_fixes={fixes}")


if __name__ == "__main__":
    main()
