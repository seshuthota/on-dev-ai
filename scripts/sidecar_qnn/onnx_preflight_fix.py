#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


REDUCE_OPS_WITH_AXES_INPUT = {
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
}


def _shape_from_vi(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def _replace_shape(value_info, new_shape):
    shape = value_info.type.tensor_type.shape
    del shape.dim[:]
    for d in new_shape:
        shape.dim.add().dim_value = int(d)


def _const_output_shapes(model):
    out = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        tensor_attr = None
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                tensor_attr = attr.t
                break
        if tensor_attr is None:
            continue
        out[node.output[0]] = list(tensor_attr.dims)
    return out


def _fix_constant_value_info_shapes(model):
    const_shapes = _const_output_shapes(model)
    changes = 0
    for vi in model.graph.value_info:
        if vi.name not in const_shapes:
            continue
        expected = const_shapes[vi.name]
        actual = _shape_from_vi(vi)
        if actual != expected:
            _replace_shape(vi, expected)
            changes += 1
    return changes


def _fix_reduce_axes_attributes(model):
    changes = 0
    for node in model.graph.node:
        if node.op_type not in REDUCE_OPS_WITH_AXES_INPUT:
            continue
        axes_attr = None
        keep_attrs = []
        for attr in node.attribute:
            if attr.name == "axes":
                axes_attr = attr
            else:
                keep_attrs.append(attr)
        if axes_attr is None:
            continue

        axes = list(axes_attr.ints)
        if not axes and axes_attr.HasField("i"):
            axes = [int(axes_attr.i)]
        if not axes:
            continue

        axes_name = f"{node.name or node.output[0]}__axes_const"
        tensor = numpy_helper.from_array(np.asarray(axes, dtype=np.int64), name=axes_name)
        model.graph.initializer.append(tensor)
        node.input.append(axes_name)
        del node.attribute[:]
        node.attribute.extend(keep_attrs)
        changes += 1
    return changes


def _set_if_symbolic_or_unknown(dim, value):
    dim.dim_value = int(value)
    if dim.HasField("dim_param"):
        dim.ClearField("dim_param")


def _fix_static_input_dims(model, batch_size, prompt_length, context_length, past_seq_len):
    changed = 0
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            continue

        name = inp.name
        shape = tt.shape.dim
        lower = name.lower()

        if name == "input_ids" and len(shape) >= 2:
            _set_if_symbolic_or_unknown(shape[0], batch_size)
            _set_if_symbolic_or_unknown(shape[1], prompt_length)
            changed += 1
            continue

        if name == "attention_mask" and len(shape) >= 2:
            _set_if_symbolic_or_unknown(shape[0], batch_size)
            _set_if_symbolic_or_unknown(shape[1], context_length)
            changed += 1
            continue

        if name == "position_ids" and len(shape) >= 2:
            _set_if_symbolic_or_unknown(shape[0], batch_size)
            _set_if_symbolic_or_unknown(shape[1], prompt_length)
            changed += 1
            continue

        if "cache_position" in lower and len(shape) >= 1:
            _set_if_symbolic_or_unknown(shape[-1], prompt_length)
            changed += 1
            continue

        if "past_key_values" in lower or "past_value" in lower or "past_key" in lower:
            # Make cache sequence dimension deterministic for AR/CL context-length inference.
            seq_axis_set = False
            for i, d in enumerate(shape):
                if i == 0:
                    _set_if_symbolic_or_unknown(d, batch_size)
                    continue
                if d.HasField("dim_param"):
                    p = d.dim_param.lower()
                    if "past" in p or "cache" in p or "kv" in p:
                        _set_if_symbolic_or_unknown(d, past_seq_len)
                        seq_axis_set = True
                    elif "seq" in p or "token" in p or "length" in p:
                        _set_if_symbolic_or_unknown(d, prompt_length)
                        seq_axis_set = True

            # Some exported static ONNX models erase semantic dim names; pick a likely seq axis
            # among non-batch dims with value=1 to avoid AR/CL context=2 failures.
            if not seq_axis_set and len(shape) >= 3:
                candidates = []
                for i in range(1, len(shape)):
                    d = shape[i]
                    if d.HasField("dim_value") and int(d.dim_value) == 1:
                        candidates.append(i)
                if len(candidates) >= 2 and 1 in candidates:
                    candidates = [i for i in candidates if i != 1]
                if candidates:
                    _set_if_symbolic_or_unknown(shape[max(candidates)], past_seq_len)

            changed += 1

    return changed


def _get_attr(node, name, default=None):
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == onnx.AttributeProto.FLOAT:
            return float(attr.f)
        if attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
        if attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
    return default


def _lower_simplified_layernorm_to_layernorm(model):
    replaced = 0
    for node in model.graph.node:
        if node.op_type != "SimplifiedLayerNormalization":
            continue
        if (node.domain or "") != "":
            continue
        if len(node.input) < 2:
            continue

        epsilon = float(_get_attr(node, "epsilon", 1.0e-5))
        axis = int(_get_attr(node, "axis", -1))
        keep_attrs = [
            helper.make_attribute("axis", axis),
            helper.make_attribute("epsilon", epsilon),
        ]
        del node.attribute[:]
        node.attribute.extend(keep_attrs)
        node.op_type = "LayerNormalization"
        replaced += 1
    return replaced


def main():
    parser = argparse.ArgumentParser(description="Apply deterministic ONNX preflight fixes for QAIRT builder compatibility.")
    parser.add_argument("--input-onnx", required=True, help="Input ONNX path")
    parser.add_argument("--output-onnx", required=True, help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=1, help="Static batch size override")
    parser.add_argument("--prompt-length", type=int, default=1, help="Static prompt length override")
    parser.add_argument("--context-length", type=int, default=2048, help="Static context length override")
    parser.add_argument("--past-seq-len", type=int, default=0, help="Static past sequence length override")
    parser.add_argument(
        "--disable-simplified-layernorm-lowering",
        action="store_true",
        help="Disable lowering SimplifiedLayerNormalization -> LayerNormalization",
    )
    parser.add_argument("--run-checker", action="store_true", help="Run onnx.checker.check_model before save")
    args = parser.parse_args()

    input_path = Path(args.input_onnx).expanduser().resolve()
    output_path = Path(args.output_onnx).expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"input onnx not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path), load_external_data=True)
    reduce_fixes = _fix_reduce_axes_attributes(model)
    const_shape_fixes = _fix_constant_value_info_shapes(model)
    sln_lowered = 0
    if not args.disable_simplified_layernorm_lowering:
        sln_lowered = _lower_simplified_layernorm_to_layernorm(model)
    static_dim_fixes = _fix_static_input_dims(
        model=model,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        context_length=args.context_length,
        past_seq_len=args.past_seq_len,
    )
    if args.run_checker:
        onnx.checker.check_model(model)

    data_name = f"{output_path.name}.data"
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )

    print(f"[info] input={input_path}")
    print(f"[info] output={output_path}")
    print(f"[info] external_data={output_path.parent / data_name}")
    print(f"[info] fixes.reduce_axes={reduce_fixes}")
    print(f"[info] fixes.constant_value_info_shapes={const_shape_fixes}")
    print(f"[info] fixes.simplified_layernorm_lowered={sln_lowered}")
    print(f"[info] fixes.static_input_dims={static_dim_fixes}")


if __name__ == "__main__":
    main()
