# Custom Runtime Benchmark Protocol v1

## Purpose

This protocol defines the first comparable benchmark record format for the custom runtime. It applies to owned CPU first, then future Vulkan and QNN sidecar comparisons.

## Fixed Settings

- model: `TinyLlama-1.1B-Chat-v1.0`
- runtime format: owned FP16 packed format
- context cap: `512`
- decode mode: greedy
- max new tokens: `64`
- warmup runs: `1`
- measured runs: `5`
- report statistic: median

## Prompt Set

The v1 benchmark uses exactly these prompts.

`short_fact`:

```text
Write one sentence about why fast on-device AI matters.
```

`reasoning_small`:

```text
A phone can process 9 tokens per second for 20 seconds. How many tokens can it process? Answer briefly.
```

`chat_medium`:

```text
Explain in two short paragraphs why measuring sustained performance is different from measuring peak performance.
```

## JSONL Schema

Each measured run writes one JSON object on one line.

Required fields:

- `schema_version`: `1`
- `baseline_id`: string, used when no git commit is available
- `git_commit`: string or `null`
- `timestamp_utc`: ISO-8601 string
- `device_serial`: string or `null`
- `device_model`: string or `null`
- `backend`: `custom_cpu`, `custom_vulkan`, or `qnn_sidecar`
- `model_id`: `TinyLlama-1.1B-Chat-v1.0`
- `model_checksum`: SHA-256 of source `model.safetensors`
- `packed_checksum`: SHA-256 of `model.bin`
- `prompt_id`: one of the fixed prompt ids
- `seed`: integer
- `context_length_cap`: integer
- `prompt_tokens`: integer
- `generated_tokens`: integer
- `load_time_ms`: number
- `ttft_ms`: number
- `prefill_tok_per_sec`: number
- `decode_tok_per_sec`: number
- `elapsed_ms`: number
- `peak_rss_mb`: number or `null`
- `thermal_samples`: array
- `status`: `ok` or `failed`
- `error`: string or `null`

## Thermal Samples

Thermal samples should be collected before, during, and after each measured run when Android APIs are available.

Each sample:

- `elapsed_ms`
- `thermal_status`
- `thermal_headroom` if available

Do not block Milestone 1 on thermal API wiring. If thermal data is unavailable, write an empty array and keep the field present.

## Output Location

Preferred app-internal location:

```text
files/benchmarks/custom_runtime_history.jsonl
```

Preferred host pull location:

```text
.artifacts/custom-runtime/benchmarks/
```

## Comparison Rule

Only compare rows when these fields match:

- `model_checksum`
- `packed_checksum`, except QNN sidecar where no packed model exists
- `prompt_id`
- `context_length_cap`
- `generated_tokens`
- `decode mode`

If any of those differ, the rows can be reported together but not treated as a direct speed comparison.
