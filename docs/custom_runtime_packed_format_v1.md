# Custom Runtime Packed Format v1

## Purpose

The packed format is the only model format the C++ runtime reads in v1. Hugging Face `safetensors` is source input for the Python packer, not a runtime dependency.

## Artifacts

```text
<packed-model-dir>/
  manifest.json
  model.bin
  tokenizer/
    tokenizer.model
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
```

## Ownership Boundary

`scripts/custom_runtime/pack_tinyllama.py` owns:

- reading `config.json`
- reading `model.safetensors`
- converting source BF16 tensors to FP16
- writing `model.bin`
- copying tokenizer artifacts
- writing `manifest.json`

C++ runtime owns:

- reading `model.bin`
- validating header and checksums
- looking up tensors by directory entry
- executing inference

C++ runtime must not parse `safetensors`, GGUF, ONNX, or QAIRT artifacts in v1.

## Packer Usage

Dry-run validation does not require `torch` or `safetensors`:

```bash
python3 scripts/custom_runtime/pack_tinyllama.py --dry-run
```

Writing `model.bin` requires `torch`, `numpy`, and `safetensors`. Use the external QAIRT Python environment or another Python with those packages:

```bash
../OnDevAI_external/.conda-qairt310/bin/python \
  scripts/custom_runtime/pack_tinyllama.py \
  --output-dir .artifacts/custom-runtime/tinyllama-v1-layer0
```

## Binary Rules

- byte order: little-endian
- tensor payload dtype: IEEE FP16
- tensor payload alignment: 64 bytes
- tensor names: UTF-8, null-terminated or length-prefixed
- offsets: absolute byte offsets from start of `model.bin`
- checksums: SHA-256 in manifest; fast per-tensor checksum in header may be added later

## Header v1

The v1 header is 80 bytes and matches Python `struct.Struct("<4sIIIIIIIIIIIffQQQ")`.

Required fields in order:

- `magic`: `ODAI`
- `version`: `1`
- `model_family`: `tinyllama_v1`
- `hidden_size`
- `intermediate_size`
- `layer_count`
- `attention_head_count`
- `kv_head_count`
- `vocab_size`
- `max_position_embeddings`
- `runtime_context_cap`
- `tensor_count`
- `rope_theta`
- `rms_norm_eps`
- `tensor_directory_offset`
- `tensor_data_offset`
- `file_size`

Keep this header simple. Add fields only when the runtime needs them.

## Tensor Directory v1

Each entry must include:

- tensor name
- dtype enum
- rank
- dimensions
- offset
- byte size

Expected first tensors:

- `model.embed_tokens.weight`
- `lm_head.weight`
- `model.norm.weight`
- `model.layers.0.input_layernorm.weight`
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.self_attn.k_proj.weight`
- `model.layers.0.self_attn.v_proj.weight`
- `model.layers.0.self_attn.o_proj.weight`
- `model.layers.0.post_attention_layernorm.weight`
- `model.layers.0.mlp.gate_proj.weight`
- `model.layers.0.mlp.up_proj.weight`
- `model.layers.0.mlp.down_proj.weight`

The packer should support all layers, but the first verification path may inspect layer 0 first.

## Manifest v1

`manifest.json` is for tooling, diagnostics, and benchmark records.

Required fields:

- `schema_version`
- `model_id`
- `source_model_path`
- `source_weight_sha256`
- `packed_model_sha256`
- `created_utc`
- `packer_version`
- `runtime_context_cap`
- `dtype`
- `source_dtype`
- `tensor_count`
- `tokenizer_files`

## Validation Rules

The packer must fail if:

- required config fields are missing
- required tokenizer files are missing
- required tensors are missing
- a tensor has an unexpected rank or shape
- `runtime_context_cap` exceeds source max position embeddings

The runtime must fail if:

- magic/version do not match
- model family is not `tinyllama_v1`
- tensor directory points outside the file
- required tensors are missing
- tensor dimensions do not match header hyperparameters

## Non-Goals

- Q4 or Q8 packing
- mmap-specific layout
- GPU-specific swizzling
- multiple model families
- paged KV metadata
- direct compatibility with external formats
