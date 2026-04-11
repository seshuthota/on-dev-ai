#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_tokenizer_path(exports_dir: Path):
    candidates = [
        exports_dir / "tokenizer.json",
        exports_dir / "tokenizer.model",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _infer_ids(exports_dir: Path):
    cfg = _load_json(exports_dir / "config.json")
    gen_cfg = _load_json(exports_dir / "generation_config.json")

    n_vocab = cfg.get("vocab_size")
    bos = gen_cfg.get("bos_token_id", cfg.get("bos_token_id"))
    eos = gen_cfg.get("eos_token_id", cfg.get("eos_token_id"))
    return n_vocab, bos, eos


def _collect_ctx_bins(container_dir: Path):
    models_dir = container_dir / "models"
    if models_dir.is_dir():
        bins = sorted(str(p.resolve()) for p in models_dir.rglob("*.bin") if p.is_file())
        if bins:
            return bins
    return sorted(str(p.resolve()) for p in container_dir.rglob("*.bin") if p.is_file())


def _resolve_model_root(model_input: Path):
    return model_input if model_input.is_dir() else model_input.parent


def _is_empty_activation_encodings(encodings_path: Path):
    if not encodings_path.is_file():
        return False
    try:
        with encodings_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        activations = data.get("activation_encodings", None)
        return isinstance(activations, list) and len(activations) == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Builds a QAIRT GenAI LLM container for HTP and writes a runtime manifest."
    )
    parser.add_argument(
        "--model-input",
        required=True,
        help="Model input path. Can be exports dir, ONNX file path, or GGUF file path.",
    )
    parser.add_argument("--container-dir", required=True, help="Output directory to save generated container.")
    parser.add_argument("--cache-root", default="", help="Optional cache root for resumable build.")
    parser.add_argument("--chipset", default="SM8750", help="Target chipset (default: SM8750).")
    parser.add_argument("--tokenizer-path", default="", help="Optional explicit tokenizer path override.")
    parser.add_argument("--config-path", default="", help="Optional explicit config path override.")
    parser.add_argument(
        "--calibration-input-list",
        default="",
        help="Optional calibration input-list path. Required for models without encodings.",
    )
    parser.add_argument(
        "--embedding-lut",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control embedding LUT preparation (default: auto).",
    )
    parser.add_argument("--qairt-tmp-dir", default="", help="Optional QAIRT_TMP_DIR.")
    args = parser.parse_args()

    model_input = Path(args.model_input).expanduser().resolve()
    container_dir = Path(args.container_dir).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else None
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve() if args.tokenizer_path else None
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else None
    calibration_input_list = (
        Path(args.calibration_input_list).expanduser().resolve() if args.calibration_input_list else None
    )

    if not model_input.exists():
        raise SystemExit(f"model input not found: {model_input}")
    if tokenizer_path and not tokenizer_path.exists():
        raise SystemExit(f"tokenizer path not found: {tokenizer_path}")
    if config_path and not config_path.exists():
        raise SystemExit(f"config path not found: {config_path}")
    if calibration_input_list and not calibration_input_list.is_file():
        raise SystemExit(f"calibration input list not found: {calibration_input_list}")

    if args.qairt_tmp_dir:
        os.environ["QAIRT_TMP_DIR"] = str(Path(args.qairt_tmp_dir).expanduser().resolve())

    try:
        from qairt.gen_ai_api.gen_ai_builder_factory import GenAIBuilderFactory
    except Exception as exc:
        raise SystemExit(f"failed to import QAIRT Python API: {exc}") from exc

    model_root = _resolve_model_root(model_input)

    print(f"[info] model_input={model_input}")
    print(f"[info] model_root={model_root}")
    print(f"[info] container_dir={container_dir}")
    if cache_root:
        print(f"[info] cache_root={cache_root}")
    print(f"[info] chipset={args.chipset}")
    print(f"[info] embedding_lut_mode={args.embedding_lut}")
    if tokenizer_path:
        print(f"[info] tokenizer_path={tokenizer_path}")
    if config_path:
        print(f"[info] config_path={config_path}")
    if calibration_input_list:
        print(f"[info] calibration_input_list={calibration_input_list}")
    if "QAIRT_TMP_DIR" in os.environ:
        print(f"[info] qairt_tmp_dir={os.environ['QAIRT_TMP_DIR']}")

    builder = GenAIBuilderFactory.create(
        str(model_input),
        "HTP",
        cache_root=str(cache_root) if cache_root else None,
        tokenizer_path=str(tokenizer_path) if tokenizer_path else None,
        config_path=str(config_path) if config_path else None,
    )
    if calibration_input_list:
        from qairt.api.converter.converter_config import CalibrationConfig, ConverterConfig

        builder.set_conversion_options(
            ConverterConfig(),
            CalibrationConfig(dataset=str(calibration_input_list), act_precision=16, bias_precision=32),
        )
        print("[info] conversion: calibration input-list configured")
    if args.embedding_lut == "off":
        setattr(builder, "_prepare_embedding_lut", False)
        print("[info] embedding_lut=disabled (explicit)")
    elif args.embedding_lut == "on":
        setattr(builder, "_prepare_embedding_lut", True)
        print("[info] embedding_lut=enabled (explicit)")
    else:
        enc_path_raw = getattr(builder, "encodings_path", None)
        if enc_path_raw:
            enc_path = Path(enc_path_raw)
            if _is_empty_activation_encodings(enc_path):
                setattr(builder, "_prepare_embedding_lut", False)
                print(f"[info] embedding_lut=disabled (auto; empty activation encodings at {enc_path})")
            else:
                print("[info] embedding_lut=enabled (auto)")
        else:
            print("[info] embedding_lut=enabled (auto; no encodings path)")
    builder.set_targets([f"chipset:{args.chipset}"])
    container = builder.build()
    container.save(str(container_dir), exist_ok=True)

    ctx_bins = _collect_ctx_bins(container_dir)
    if not ctx_bins:
        raise SystemExit(f"no ctx bins found under container models dir: {container_dir / 'models'}")

    tokenizer_path_inferred = str(tokenizer_path) if tokenizer_path else _pick_tokenizer_path(model_root)
    n_vocab, bos, eos = _infer_ids(model_root)

    manifest = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_input": str(model_input),
        "model_root": str(model_root),
        "container_dir": str(container_dir),
        "chipset": args.chipset,
        "tokenizer_json": tokenizer_path_inferred,
        "n_vocab": n_vocab,
        "bos_token": bos,
        "eos_token": eos,
        "ctx_bins": ctx_bins,
        "ctx_bin_count": len(ctx_bins),
    }

    manifest_path = container_dir / "htp_runtime_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"[info] ctx_bin_count={len(ctx_bins)}")
    print(f"[info] manifest={manifest_path}")


if __name__ == "__main__":
    main()
