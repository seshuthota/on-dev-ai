# llama.cpp Android Benchmark Baseline

**Device:** iQOO 13 (Snapdragon 8 Elite, arm64-v8a)
**llama.cpp:** `073bb2c20`
**Date:** 2026-04-11
**Protocol:** threads=6, cpu_mask=0xff (auto-big), warmup=1, runs=3, greedy, ctx=512, max_new_tokens=64

## Median Results

### Qwen2.5 1.5B Q4_K_M

| Prompt | TTFT (ms) | Decode (tok/s) | Generated |
|--------|-----------|----------------|-----------|
| short_fact | 103.25 | 53.79 | 63 |
| reasoning_small | 158.37 | 53.27 | 43 |
| chat_medium | 139.86 | 50.76 | 63 |
| **Median** | **133.83** | **~53** | |

### Qwen3.5 4B Q4_K_M

| Prompt | TTFT (ms) | Decode (tok/s) | Generated |
|--------|-----------|----------------|-----------|
| short_fact | 950.15 | 15.26 | 63 |
| reasoning_small | 1050.98 | 14.22 | 63 |
| chat_medium | 930.19 | 14.99 | 63 |
| **Median** | **~950** | **~15** | |

## Summary

| Model | Quant | Median Decode (tok/s) | Median TTFT (ms) |
|-------|-------|----------------------:|-----------------:|
| Qwen2.5 1.5B | Q4_K_M | ~53 | ~134 |
| Qwen3.5 4B | Q4_K_M | ~15 | ~950 |

**Best config:** threads=6, cpu_mask=0xff (all 8 cores)

## Benchmark Infrastructure Fixes

### Model Checksum Verification
The benchmark script now verifies model identity before reusing device artifacts:
- Host model SHA256 is pushed to `${TARGET_DIR}/model.sha256`
- `--reuse-device-artifacts` checks device checksum matches expected
- On mismatch: fails with clear error message

### Model Metadata Logging
JSONL rows now include `model_name_log`, `model_type_log`, `model_params_log`, `file_size_log` parsed from the actual llama-cli log output. This provides defensive verification that the correct model was loaded.

### Historical Bug
Early 4B benchmark runs used `--reuse-device-artifacts` without checksum verification. When the device had the 1.5B model loaded, the script labeled runs as "4B" but executed the 1.5B model, producing ~53 tok/s decode — identical to the actual 1.5B results. The fix ensures the on-device model matches the intended model before benchmarking.

## Notes

- 4B is **3.5x slower** than 1.5B decode (15 vs 53 tok/s) — expected given model size difference
- 4B TTFT (~950ms) is **7x higher** than 1.5B (~134ms) — the prefill phase scales roughly with parameter count
- CPU affinity `0xff` (all 8 cores) is the best setting for both models
- The decode throughput ratio (4B/1.5B ≈ 0.28) roughly matches the parameter count ratio (4B/1.5B ≈ 2.7), suggesting decode is bandwidth-bound but the larger model has proportionally higher memory traffic per token

## Benchmark Data

Raw results: `files/benchmarks/llama_cpp_android.jsonl`
Run artifacts: `.artifacts/llama-cpp/runs/<timestamp>/`

Query with filters:
```bash
# Latest valid results only
python3 scripts/benchmarks/compare_jsonl.py files/benchmarks/llama_cpp_android.jsonl --status ok --latest-only

# Specific model
python3 scripts/benchmarks/compare_jsonl.py files/benchmarks/llama_cpp_android.jsonl --model-id Qwen3.5-4B-Q4_K_M --status ok --latest-only
```
