#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path


MAGIC = b"ODAI"
VERSION = 1
DTYPE_FP16 = 1
ALIGNMENT = 64
HEADER_STRUCT = struct.Struct("<4sIIIIIIIIIIIffQQQ")


EXPECTED_CONFIG = {
    "model_type": "llama",
    "hidden_size": 2048,
    "intermediate_size": 5632,
    "num_hidden_layers": 22,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "vocab_size": 32000,
    "max_position_embeddings": 2048,
}


TOKENIZER_FILES = [
    "tokenizer.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def required_tensor_names(layer_count: int):
    names = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    for layer_idx in range(layer_count):
        prefix = f"model.layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            ]
        )
    return names


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align_up(value: int, alignment: int = ALIGNMENT):
    return ((value + alignment - 1) // alignment) * alignment


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict, context_cap: int):
    for key, expected in EXPECTED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            raise SystemExit(f"unexpected config {key}: expected {expected}, got {actual}")
    if context_cap > int(config["max_position_embeddings"]):
        raise SystemExit("context cap exceeds max_position_embeddings")


def read_safetensors_metadata(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def expected_shape(name: str, config: dict):
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    head_dim = hidden // int(config["num_attention_heads"])
    kv_dim = int(config["num_key_value_heads"]) * head_dim

    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return [vocab, hidden]
    if name == "model.norm.weight":
        return [hidden]
    if name.endswith(".input_layernorm.weight") or name.endswith(".post_attention_layernorm.weight"):
        return [hidden]
    if name.endswith(".self_attn.q_proj.weight") or name.endswith(".self_attn.o_proj.weight"):
        return [hidden, hidden]
    if name.endswith(".self_attn.k_proj.weight") or name.endswith(".self_attn.v_proj.weight"):
        return [kv_dim, hidden]
    if name.endswith(".mlp.gate_proj.weight") or name.endswith(".mlp.up_proj.weight"):
        return [intermediate, hidden]
    if name.endswith(".mlp.down_proj.weight"):
        return [hidden, intermediate]
    raise SystemExit(f"no expected shape rule for tensor: {name}")


def validate_tensors(metadata: dict, tensor_names: list[str], config: dict):
    missing = [name for name in tensor_names if name not in metadata]
    if missing:
        raise SystemExit("missing required tensors:\n" + "\n".join(missing))

    for name in tensor_names:
        meta = metadata[name]
        dtype = meta.get("dtype")
        shape = meta.get("shape")
        expected = expected_shape(name, config)
        if dtype != "BF16":
            raise SystemExit(f"unexpected dtype for {name}: expected BF16, got {dtype}")
        if shape != expected:
            raise SystemExit(f"unexpected shape for {name}: expected {expected}, got {shape}")


def directory_entry_bytes(entry: dict):
    name = entry["name"].encode("utf-8")
    shape = entry["shape"]
    return (
        struct.pack("<H", len(name))
        + name
        + struct.pack("<BBHQQ", DTYPE_FP16, len(shape), 0, entry["offset"], entry["byte_size"])
        + struct.pack(f"<{len(shape)}I", *shape)
    )


def build_directory_bytes(entries: list[dict]):
    return b"".join(directory_entry_bytes(entry) for entry in entries)


def write_packed_model(model_dir: Path, output_dir: Path, tensor_names: list[str], context_cap: int):
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit("packing requires torch and safetensors; use ONDEVAI_QAIRT_PYTHON or install deps") from exc

    config = load_json(model_dir / "config.json")
    weights_path = model_dir / "model.safetensors"
    metadata = read_safetensors_metadata(weights_path)
    validate_config(config, context_cap)
    validate_tensors(metadata, tensor_names, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for file_name in TOKENIZER_FILES:
        source = model_dir / file_name
        if not source.is_file():
            raise SystemExit(f"missing tokenizer file: {source}")
        shutil.copy2(source, tokenizer_dir / file_name)

    entries = []
    payloads = []
    current_offset = HEADER_STRUCT.size

    provisional_entries = [
        {
            "name": name,
            "shape": metadata[name]["shape"],
            "offset": 0,
            "byte_size": 2 * int(torch.tensor(metadata[name]["shape"]).prod().item()),
        }
        for name in tensor_names
    ]
    directory_size = len(build_directory_bytes(provisional_entries))
    data_offset = align_up(HEADER_STRUCT.size + directory_size)
    current_offset = data_offset

    with safe_open(str(weights_path), framework="pt", device="cpu") as tensors:
        for name in tensor_names:
            tensor = tensors.get_tensor(name).to(dtype=torch.float16).contiguous()
            raw = tensor.numpy().tobytes()
            current_offset = align_up(current_offset)
            entries.append(
                {
                    "name": name,
                    "dtype": "fp16",
                    "shape": list(tensor.shape),
                    "offset": current_offset,
                    "byte_size": len(raw),
                }
            )
            payloads.append((current_offset, raw))
            current_offset += len(raw)

    directory = build_directory_bytes(entries)
    model_bin = output_dir / "model.bin"
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        1,
        int(config["hidden_size"]),
        int(config["intermediate_size"]),
        int(config["num_hidden_layers"]),
        int(config["num_attention_heads"]),
        int(config["num_key_value_heads"]),
        int(config["vocab_size"]),
        int(config["max_position_embeddings"]),
        int(context_cap),
        len(entries),
        float(config["rope_theta"]),
        float(config["rms_norm_eps"]),
        HEADER_STRUCT.size,
        data_offset,
        current_offset,
    )

    with model_bin.open("wb") as f:
        f.write(header)
        f.write(directory)
        f.write(b"\0" * (data_offset - f.tell()))
        for offset, raw in payloads:
            if f.tell() > offset:
                raise SystemExit("internal packing error: overlapping tensor payload")
            f.write(b"\0" * (offset - f.tell()))
            f.write(raw)

    manifest = {
        "schema_version": 1,
        "model_id": "TinyLlama-1.1B-Chat-v1.0",
        "model_family": "tinyllama_v1",
        "source_model_path": str(model_dir),
        "source_weight_sha256": sha256_file(weights_path),
        "packed_model_sha256": sha256_file(model_bin),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "packer_version": 1,
        "runtime_context_cap": context_cap,
        "dtype": "fp16",
        "source_dtype": str(config.get("torch_dtype", "unknown")),
        "tensor_count": len(entries),
        "tokenizer_files": TOKENIZER_FILES,
        "tensors": entries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    return manifest


def print_dry_run(model_dir: Path, layer_limit: int | None, context_cap: int):
    config = load_json(model_dir / "config.json")
    weights_path = model_dir / "model.safetensors"
    metadata = read_safetensors_metadata(weights_path)
    validate_config(config, context_cap)

    layer_count = int(config["num_hidden_layers"]) if layer_limit is None else layer_limit
    tensor_names = required_tensor_names(layer_count)
    validate_tensors(metadata, tensor_names, config)

    print(f"model_dir={model_dir}")
    print(f"weights={weights_path}")
    print(f"source_weight_sha256={sha256_file(weights_path)}")
    print(f"selected_tensor_count={len(tensor_names)}")
    print(f"source_tensor_count={len(metadata)}")
    print(f"context_cap={context_cap}")
    for name in tensor_names[:24]:
        meta = metadata[name]
        print(f"{name} dtype={meta['dtype']} shape={meta['shape']}")
    if len(tensor_names) > 24:
        print(f"... {len(tensor_names) - 24} more selected tensors")


def main():
    parser = argparse.ArgumentParser(description="Pack TinyLlama HF safetensors into OnDevAI custom format v1.")
    parser.add_argument(
        "--model-dir",
        default="/home/curious/models/TinyLlama-1.1B-Chat-v1.0",
        help="TinyLlama Hugging Face model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=".artifacts/custom-runtime/tinyllama-v1",
        help="Output directory for model.bin, manifest.json, and tokenizer files.",
    )
    parser.add_argument("--context-cap", type=int, default=512)
    parser.add_argument(
        "--layer-limit",
        type=int,
        default=1,
        help="Number of layers to pack for early bring-up. Use 22 for full TinyLlama.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config/tensors and print selected metadata.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model dir not found: {model_dir}")
    if args.layer_limit < 1 or args.layer_limit > 22:
        raise SystemExit("--layer-limit must be between 1 and 22")

    if args.dry_run:
        print_dry_run(model_dir, args.layer_limit, args.context_cap)
        return

    config = load_json(model_dir / "config.json")
    tensor_names = required_tensor_names(args.layer_limit)
    manifest = write_packed_model(model_dir, output_dir, tensor_names, args.context_cap)
    print(f"wrote {output_dir}")
    print(f"tensor_count={manifest['tensor_count']}")
    print(f"packed_model_sha256={manifest['packed_model_sha256']}")
    print(f"hidden_size={config['hidden_size']} layers_packed={args.layer_limit}")


if __name__ == "__main__":
    main()
