#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Flatten a multimodal HF config into a text-only config for QAIRT builder."
    )
    parser.add_argument("--input-config", required=True, help="Path to source config.json")
    parser.add_argument("--output-config", required=True, help="Path to flattened config.json")
    args = parser.parse_args()

    in_path = Path(args.input_config).expanduser().resolve()
    out_path = Path(args.output_config).expanduser().resolve()

    with in_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = cfg.get("text_config")
    if not isinstance(text_cfg, dict):
        raise SystemExit("input config has no text_config object")

    flat = dict(text_cfg)
    flat["architectures"] = ["GemmaForCausalLM"]
    flat["bos_token_id"] = flat.get("bos_token_id", cfg.get("boa_token_id", cfg.get("eos_token_id")))
    flat["eos_token_id"] = flat.get("eos_token_id", cfg.get("eos_token_id"))
    flat["model_type"] = flat.get("model_type", "gemma")
    flat["transformers_version"] = cfg.get("transformers_version")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2)
        f.write("\n")

    print(f"[info] wrote {out_path}")


if __name__ == "__main__":
    main()
