# Model Decision: Custom Runtime v1

## Decision

Use `TinyLlama-1.1B-Chat-v1.0` as the first custom runtime target.

Local source path:

```text
/home/curious/models/TinyLlama-1.1B-Chat-v1.0
```

Primary weight file:

```text
/home/curious/models/TinyLlama-1.1B-Chat-v1.0/model.safetensors
```

Weight checksum:

```text
sha256(model.safetensors)=6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933
```

## Why This Model

- It is a dense decoder-only `LlamaForCausalLM` model.
- It avoids multimodal, MoE, and new architecture friction.
- It is large enough to make performance work meaningful, but small enough for first bring-up.
- It has local Hugging Face artifacts already present: `config.json`, `model.safetensors`, `tokenizer.model`, `tokenizer.json`, and tokenizer metadata.

## Config Facts

- `model_type`: `llama`
- `hidden_size`: `2048`
- `num_hidden_layers`: `22`
- `num_attention_heads`: `32`
- `num_key_value_heads`: `4`
- `intermediate_size`: `5632`
- `vocab_size`: `32000`
- `max_position_embeddings`: `2048`
- `rope_theta`: `10000.0`
- `rms_norm_eps`: `1e-05`
- `bos_token_id`: `1`
- `eos_token_id`: `2`
- source dtype: `bfloat16`

## Runtime v1 Constraints

- Fixed context cap: `512`
- Decode mode: greedy
- Runtime tensor format: owned FP16 packed format
- Source loader: Python packer reads Hugging Face `safetensors`
- C++ runtime reads only the owned packed format
- No direct GGUF, ONNX, QAIRT, or QNN in the mainline runtime path

## Important Conversion Note

The source model declares `bfloat16`. Runtime format v1 will pack weights as FP16 for simplicity. This is acceptable for the correctness bring-up, but the oracle must use the same packed FP16 values when comparing desktop and Android outputs. Do not compare Android packed-FP16 logits against an untouched BF16/PyTorch run without accounting for conversion error.

## Rejected For v1

- `Qwen3.5`: current local evidence shows architecture/tooling friction.
- GGUF as runtime input: useful later, but it adds parser and layout complexity before the engine exists.
- Q4/Q8 first: quantization should start after the FP16 path proves correctness.
- HTP/QNN first: sidecar only until the owned runtime has benchmark records.

## Exit Criteria

This model decision can be revisited only after Milestone 3 in `RESET_PLAN_V2.md`, or earlier if `TinyLlama-1.1B-Chat-v1.0` fails for a reason specific to its artifacts rather than the runtime implementation.
