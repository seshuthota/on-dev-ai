#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import onnx


def _normalize_gguf_scalar(v):
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", errors="replace")
    try:
        import numpy as np

        if isinstance(v, np.ndarray):
            if v.dtype == np.uint8:
                return bytes(v.tolist()).decode("utf-8", errors="replace")
            if v.size == 1:
                return _normalize_gguf_scalar(v.item())
            return str(v.tolist())
        if isinstance(v, np.generic):
            return _normalize_gguf_scalar(v.item())
    except Exception:
        pass
    return str(v)


def _build_gguf_report(gguf_path: Path):
    try:
        from gguf import GGUFReader
    except Exception as exc:
        return {
            "gguf_path": str(gguf_path),
            "gguf_parse_ok": False,
            "gguf_parse_error": str(exc),
            "qairt_gguf_architecture_unsupported": True,
        }

    reader = GGUFReader(str(gguf_path))
    arch = None
    model_name = None
    try:
        if "general.architecture" in reader.fields:
            f = reader.fields["general.architecture"]
            if f.data:
                i = f.data[0]
                v = f.parts[i]
                arch = _normalize_gguf_scalar(v)
        if "general.name" in reader.fields:
            f = reader.fields["general.name"]
            if f.data:
                i = f.data[0]
                v = f.parts[i]
                model_name = _normalize_gguf_scalar(v)
    except Exception:
        pass

    supported_architectures = None
    qairt_arch_unsupported = False
    try:
        from qti.aisw.converters.gguf_builder.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES

        supported_architectures = sorted(GGUF_SUPPORTED_ARCHITECTURES)
        if arch:
            qairt_arch_unsupported = arch not in set(GGUF_SUPPORTED_ARCHITECTURES)
    except Exception:
        # Conservative fallback based on observed failures in this repo.
        if arch:
            qairt_arch_unsupported = arch in {"qwen35"}

    return {
        "gguf_path": str(gguf_path),
        "gguf_parse_ok": True,
        "gguf_architecture": arch,
        "gguf_model_name": model_name,
        "qairt_gguf_supported_architectures": supported_architectures,
        "qairt_gguf_architecture_unsupported": bool(qairt_arch_unsupported),
    }


def _find_onnx(model_input: Path):
    if model_input.is_file() and model_input.suffix == ".onnx":
        return model_input
    if model_input.is_dir():
        direct = sorted(model_input.glob("*.onnx"))
        recursive = sorted(model_input.rglob("*.onnx"))
        matches = direct if direct else recursive
        if matches:
            priorities = {
                "model_q4_static.onnx": 0,
                "model_q4.onnx": 1,
                "decoder_model_merged_q4.onnx": 2,
                "model.onnx": 3,
            }

            def _key(p: Path):
                return (priorities.get(p.name, 100), len(str(p)))

            return sorted(matches, key=_key)[0]
    return None


def _find_encodings(onnx_path: Path):
    base = onnx_path.with_suffix(".encodings")
    if base.is_file():
        return str(base)
    matches = sorted(onnx_path.parent.glob("*.encodings"))
    return str(matches[0]) if len(matches) == 1 else None


def _shape_info(value_info):
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return dims


def _build_onnx_report(onnx_path: Path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    inputs = {i.name: _shape_info(i) for i in model.graph.input}
    op_counts = Counter(n.op_type for n in model.graph.node)
    domain_counts = Counter(n.domain or "" for n in model.graph.node)
    default_opset = 0
    opsets = {}
    for oi in model.opset_import:
        domain = oi.domain or ""
        opsets[domain] = int(oi.version)
        if domain == "":
            default_opset = int(oi.version)

    attention_mask_shape = inputs.get("attention_mask")
    attention_mask_last_static = (
        isinstance(attention_mask_shape[-1], int) if attention_mask_shape and attention_mask_shape else False
    )
    has_past_value_name = any("past_value" in name for name in inputs)
    has_past_key_values = any(name.startswith("past_key_values.") for name in inputs)
    has_simplified_layernorm = "SimplifiedLayerNormalization" in op_counts
    has_dynamic_dims = any(any(not isinstance(d, int) for d in shape) for shape in inputs.values())

    ar_cl_heuristic_risk = (not attention_mask_last_static) and (not has_past_value_name)
    sln_opset_risk = has_simplified_layernorm and default_opset >= 21
    onnx_file_size_bytes = os.path.getsize(onnx_path)
    ext_candidates = []
    ext_candidates.extend(sorted(onnx_path.parent.glob(f"{onnx_path.stem}.onnx_data*")))
    ext_candidates.extend(sorted(onnx_path.parent.glob(f"{onnx_path.name}.data*")))
    ext_sizes = {str(p): os.path.getsize(p) for p in ext_candidates if p.is_file()}
    ext_total_size = sum(ext_sizes.values())
    protobuf_serialize_risk = ext_total_size >= (1800 * 1024 * 1024)

    return {
        "onnx_path": str(onnx_path),
        "onnx_file_size_bytes": onnx_file_size_bytes,
        "onnx_external_data_files": ext_sizes,
        "onnx_external_data_total_bytes": ext_total_size,
        "default_opset": default_opset,
        "opsets": opsets,
        "node_count": len(model.graph.node),
        "input_count": len(model.graph.input),
        "output_count": len(model.graph.output),
        "top_ops": op_counts.most_common(30),
        "domains": domain_counts,
        "inputs": inputs,
        "attention_mask_last_dim_static": attention_mask_last_static,
        "has_past_value_name": has_past_value_name,
        "has_past_key_values": has_past_key_values,
        "has_simplified_layernorm": has_simplified_layernorm,
        "has_dynamic_dims": has_dynamic_dims,
        "qairt_ar_cl_context_heuristic_likely_fail": ar_cl_heuristic_risk,
        "qairt_simplified_layernorm_opset_risk": sln_opset_risk,
        "qairt_protobuf_serialization_risk": protobuf_serialize_risk,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate compatibility intake report for model inputs.")
    parser.add_argument("--model-input", required=True, help="Path to model input (ONNX/GGUF/dir)")
    parser.add_argument("--output-json", default="", help="Optional output report JSON path")
    parser.add_argument("--calibration-input-list", default="", help="Optional calibration input-list path")
    args = parser.parse_args()

    model_input = Path(args.model_input).expanduser().resolve()
    if not model_input.exists():
        raise SystemExit(f"model input not found: {model_input}")

    onnx_path = _find_onnx(model_input)
    report = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_input": str(model_input),
        "model_kind": "onnx" if onnx_path else ("gguf" if model_input.suffix == ".gguf" else "unknown"),
        "calibration_input_list": args.calibration_input_list or None,
        "calibration_input_list_exists": bool(args.calibration_input_list and Path(args.calibration_input_list).is_file()),
    }

    if onnx_path:
        onnx_report = _build_onnx_report(onnx_path)
        report.update(onnx_report)
        report["encodings_path"] = _find_encodings(onnx_path)
        report["encodings_found"] = bool(report["encodings_path"])
        report["requires_calibration_or_encodings"] = not (
            report["encodings_found"] or report["calibration_input_list_exists"]
        )
    elif report["model_kind"] == "gguf":
        gguf_report = _build_gguf_report(model_input)
        report.update(gguf_report)
        report["encodings_found"] = False
        report["requires_calibration_or_encodings"] = True
    else:
        report["encodings_found"] = False
        report["requires_calibration_or_encodings"] = True

    actions = []
    if report.get("qairt_ar_cl_context_heuristic_likely_fail"):
        actions.append("apply_static_shape_preflight_for_attention_mask_and_kv_cache")
    if report.get("qairt_simplified_layernorm_opset_risk"):
        actions.append("likely_requires_opset_or_operator_lowering_for_SimplifiedLayerNormalization")
    if report.get("qairt_protobuf_serialization_risk"):
        actions.append("high_risk_of_protobuf_serialize_failure_during_shape_inference")
    if report.get("qairt_gguf_architecture_unsupported"):
        actions.append("switch_to_supported_gguf_architecture_or_qairstack")
    if report.get("requires_calibration_or_encodings"):
        actions.append("provide_calibration_input_list_or_encodings")
    report["recommended_actions"] = actions

    text = json.dumps(report, indent=2)
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"[info] report={out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
