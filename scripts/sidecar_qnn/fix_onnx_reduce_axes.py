#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _read_axes_attr(node):
    for attr in node.attribute:
        if attr.name == "axes":
            return list(attr.ints)
    return None


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


def main():
    parser = argparse.ArgumentParser(
        description="For opset>=18 models, rewrite ReduceMean axes attr to axes input tensor."
    )
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", default="", help="Optional output path; default overwrites input.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else input_path

    model = onnx.load(str(input_path))
    default_opset = 0
    for oi in model.opset_import:
        if oi.domain == "":
            default_opset = oi.version
            break

    if default_opset < 18:
        print(f"[info] default opset={default_opset}; no fix required")
        if output_path != input_path:
            onnx.save(model, str(output_path))
        return

    existing_names = {init.name for init in model.graph.initializer}
    fixes = 0

    for node in model.graph.node:
        if node.op_type != "ReduceMean":
            continue
        axes = _read_axes_attr(node)
        if axes is None:
            continue
        if len(node.input) >= 2 and node.input[1]:
            continue

        base = (node.name or "reduce_mean").replace("/", "_")
        init_name = f"{base}_axes_const"
        idx = 0
        while init_name in existing_names:
            idx += 1
            init_name = f"{base}_axes_const_{idx}"
        existing_names.add(init_name)

        if len(axes) == 1:
            dims = []
            vals = [int(axes[0])]
        else:
            dims = [len(axes)]
            vals = [int(v) for v in axes]
        init = helper.make_tensor(name=init_name, data_type=TensorProto.INT64, dims=dims, vals=vals)
        model.graph.initializer.append(init)

        node.input.extend([init_name])
        kept = [a for a in node.attribute if a.name != "axes"]
        del node.attribute[:]
        node.attribute.extend(kept)
        fixes += 1

    _save_model(model, output_path)
    print(f"[info] wrote {output_path}")
    print(f"[info] reduce_mean_axes_fixed={fixes}")


if __name__ == "__main__":
    main()
