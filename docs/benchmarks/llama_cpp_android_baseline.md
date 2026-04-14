# llama.cpp Android Benchmark Baseline

**Device:** iQOO 13 (Snapdragon 8 Elite, arm64-v8a)
**llama.cpp:** `073bb2c20`
**Date:** 2026-04-11 (updated 2026-04-12)
**Protocol:** threads=4, cpu_mask=0xff (all 8 cores), warmup=1, runs=3, greedy, ctx=512, max_new_tokens=64

## Corrected Baseline (2026-04-12)

> **Note:** Earlier runs (2026-04-11) used threads=6 and reported ~53 tok/s for 1.5B. The thread/mask sweep (2026-04-12) found that threads=4 with mask=0xff consistently outperforms threads=6, yielding a trimmed median of **49.41 tok/s** for 1.5B. The ~53 figure was a single-run outlier, not the stable baseline. All future testing should use threads=4.

### Qwen2.5 1.5B Q4_K_M — Corrected

| Prompt | TTFT (ms) | Decode (tok/s) | Generated |
|--------|-----------|----------------|-----------|
| short_fact | 109.04 | 49.47 | 63 |
| reasoning_small | 183.21 | 49.41 | 43 |
| chat_medium | 154.68 | 43.74 | 63 |
| **Median** | **~140** | **~49** | |

### Qwen3.5 4B Q4_K_M

| Prompt | TTFT (ms) | Decode (tok/s) | Generated |
|--------|-----------|----------------|-----------|
| short_fact | 950.15 | 15.26 | 63 |
| reasoning_small | 1050.98 | 14.22 | 63 |
| chat_medium | 930.19 | 14.99 | 63 |
| **Median** | **~950** | **~15** | |

## Thread/Mask Sweep Results (2026-04-12)

Full matrix: threads ∈ {4,5,6,7,8} × masks ∈ {0xff, 0xf0, 0xf8, 0x0}, 3 prompts, 3 runs each (trimmed median).

| Threads | Mask | Median tok/s |
|---------|------|-------------:|
| **4** | **0xff** | **49.41** |
| 6 | 0xff | 47.75 |
| 6 | 0xf0 | 47.74 |
| 6 | 0xf8 | 47.45 |
| 7 | 0x0 | 45.60 |
| 8 | 0xf0 | 45.46 |
| 6 | 0x0 | 45.10 |
| 8 | 0xf8 | 43.70 |
| 4 | 0xf0 | 43.47 |
| 7 | 0xff | 42.58 |
| 7 | 0xf8 | 42.43 |
| 8 | 0xff | 41.83 |
| 5 | 0x0 | 41.67 |
| 5 | 0xf8 | 41.65 |
| 5 | 0xf0 | 41.49 |
| 5 | 0xff | 41.45 |
| 4 | 0xf8 | 38.55 |
| 4 | 0x0 | 38.44 |
| 7 | 0xf0 | 37.80 |
| 8 | 0x0 | 35.09 |

**Best config: threads=4, cpu_mask=0xff (all 8 cores) at 49.41 tok/s median.**

Key findings:
- **4 threads beats 6-8 threads** with the same 0xff mask. Oversubscription hurts.
- **0xff (all cores) is best** even with 4 threads. The little cores help via context switching overlap.
- 0xf0 (prime+big, no little) is consistently 2-5 tok/s worse than 0xff.
- 0x0 (no affinity) is worst for 6-8 threads, but reasonable for 4-5 threads.

Per-prompt breakdown for best config (t=4, mask=0xff):
| Prompt | TTFT (ms) | Decode (tok/s) | Generated |
|--------|-----------|----------------|-----------|
| short_fact | 109.04 | 49.47 | 63 |
| reasoning_small | 183.21 | 49.41 | 43 |
| chat_medium | 154.68 | 43.74 | 63 |

## Summary

| Model | Quant | Median Decode (tok/s) | Median TTFT (ms) |
|-------|-------|----------------------:|-----------------:|
| Qwen2.5 1.5B | Q4_K_M | ~49 | ~134 |
| Qwen3.5 4B | Q4_K_M | ~15 | ~950 |

**Best config:** threads=4, cpu_mask=0xff (all 8 cores)

## Sustained 4B Check

Qwen3.5 4B Q4_K_M was also run with `max_new_tokens=256`, `threads=6`, `cpu_mask=0xff`, and verified 4B metadata in the logs.

| Prompt | TTFT (ms) | Decode (tok/s) | Generated | Elapsed (ms) |
|--------|-----------|----------------|-----------|--------------|
| short_fact | 1505.78 | 16.46 | 255 | 17104.16 |
| reasoning_small | 1420.30 | 15.44 | 163 | 12081.58 |
| chat_medium | 642.18 | 14.39 | 255 | 18542.84 |
| **Median** | **1420.30** | **15.44** | **255** | **17104.16** |

The sustained decode rate remains in the same range as the 64-token median pass, so there is no immediate evidence of severe short-run throttling at this duration.

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

## Profiler A/B Findings

**Device:** iQOO 13 (Snapdragon 8 Elite, arm64-v8a)
**Build:** profiler patch from `ondevai-profiler.patch`, compile with `-DGGML_PROFILER`

### Throughput Impact

| Build Variant | Decode (tok/s) | Notes |
|---------------|----------------|-------|
| clean llama-bench tg128 | ~42-44 | Baseline |
| profiler compiled, runtime off | ~32 | After reuse |
| profiler runtime on | ~27 | Active instrumentation overhead |

**Conclusion:** Profiler builds are for hotspot direction only, not production perf numbers.

### MUL_MAT Hotspot — Corrected Interpretation (2026-04-13)

**Method:** Kernel path inspection + standalone ARM NEON microbenchmark

**Key finding:** The `f16xf32` profiler label is graph-level, NOT the inner kernel type.

The profiler hook at `ggml_compute_forward_mul_mat` records `src0->type` and `src1->type` from the compute graph. For Q4_K_M models:
- `src0->type = Q4_K` (weight tensor's quantized type in the graph)
- `src1->type = F32` (activation tensor)
- `vec_dot_type = Q8_K` (internal conversion type for the inner kernel)

The actual inner kernel dispatched for Q4_K_M is `ggml_vec_dot_q4_K_q8_K`, which uses **ARM DOTPROD (vdotq_s32)** — not FP16 matvec. The `f16xf32` label is a misleading artifact of the graph-level profiler hook.

**Q4_K kernel verification (arch/arm/quants.c):**
- NEON INT8 path gated by `__ARM_FEATURE_MATMUL_INT8` (defined with `-march=armv8.7a`)
- DOTPROD path gated by `__ARM_FEATURE_DOTPROD` (mandatory baseline in armv8.7a)
- Generic scalar fallback for non-DOTPROD targets
- On this build: `vdotq_s32` is active for Q4_K × Q8_K dot

**Memory-bandwidth bound confirmed:**
- F32 matvec on device: ~7-11 GFLOPS (bandwidth-limited, not compute-limited)
- F16 matvec on device: ~7-8 GFLOPS (same bottleneck)
- Q4_K × Q8_K dot: ~0.6 GB/s effective bandwidth
- Raw sequential read: ~15.6 GB/s
- Adding `+fp16+i8mm+dotprod` to `-march=armv8.7a`: ~5% improvement on 1.5B

**Implication for optimization:**
The current Q4_K path IS already using the expected ARM DOTPROD implementation. Obvious SIMD enablement is not missing. The bottleneck is memory bandwidth, not compute. Optimization targets:
1. Reduce memory traffic (avoid Q8_K expansion per block)
2. Improve cache locality / reduce repeated reads
3. Thread scheduling around bandwidth pressure
4. NOT: rewriting the dot loop or enabling more SIMD flags

**Build comparison (Qwen2.5 1.5B Q4_K_M, llama-bench tg64, t=6):**

| Build | Flags | Decode (tok/s) |
|-------|-------|----------------|
| armv87-clean | `-march=armv8.7a` | 20.85 ± 1.02 |
| explicit-all | `-march=armv8.7a+fp16+i8mm+dotprod` | 21.87 ± 0.38 |

~5% improvement from explicit SIMD flags, with lower variance. The extra flags help marginally but don't change the memory-bandwidth-bound nature of the workload.

**Profile data location:** `.artifacts/llama-cpp/runs/20260412_154737/`

