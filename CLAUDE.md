# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OnDevAI is a **Snapdragon 8 Elite hardware-specific LLM inference app** focused on measuring and optimizing token generation performance. It combines a Kotlin/Compose Android UI with a native C++ inference engine.

**Current direction:** llama.cpp is the production inference engine. The custom runtime (`native/custom/`) is preserved as a correctness oracle and kernel lab. TinyLlama-1.1B-Chat-v1.0 is the target model for custom runtime work; Qwen2.5 1.5B and Qwen3.5 4B Q4_K_M are the llama.cpp benchmark models.

Target hardware: **arm64-v8a only, Snapdragon 8 Elite with Adreno GPU**

## Build Commands

```bash
# First time only: initialize git submodules (llama.cpp, Vulkan-Headers)
git submodule update --init --recursive

# Build debug APK (from project root) — includes native CMake build
./gradlew :app:assembleDebug

# Build with QNN runtime packaging enabled (sidecar only)
./gradlew :app:assembleDebug -Pondevai.enableQnn=true

# Install APK to connected device
./scripts/install_debug.sh

# View runtime diagnostics on device
adb logcat -s OnDevAI

# Run JVM unit tests
./gradlew :app:testDebugUnitTest

# Run Android instrumentation tests (requires device/emulator)
./gradlew :app:connectedDebugAndroidTest

# Run Android lint checks
./gradlew :app:lint
```

## llama.cpp Android Build & Benchmark Commands

```bash
# Build llama.cpp for Android arm64-v8a (outputs to .artifacts/llama-cpp/android-install/)
./scripts/llama_cpp/build_android.sh

# Run benchmarks on Android device
# Requires: --serial (adb device), --model (GGUF path), --model-id
./scripts/llama_cpp/benchmark_android.sh \
  --serial 1234567890 \
  --model /path/to/model.Q4_K_M.gguf \
  --model-id Qwen2.5-1.5B-Q4_K_M \
  --threads 6 \
  --cpu-mask auto-big \
  --runs 5 --warmup 1

# Reuse device artifacts (skip re-push of model + binary if already on device)
./scripts/llama_cpp/benchmark_android.sh ... --reuse-device-artifacts

# Compare benchmark results across JSONL files
python3 scripts/benchmarks/compare_jsonl.py files/benchmarks/llama_cpp_android.jsonl

# With filters (recommended for comparing latest runs):
python3 scripts/benchmarks/compare_jsonl.py files/benchmarks/llama_cpp_android.jsonl --status ok --latest-only
python3 scripts/benchmarks/compare_jsonl.py files/benchmarks/llama_cpp_android.jsonl --model-id Qwen3.5-4B-Q4_K_M --status ok --latest-only
```

## Architecture

```
Compose UI (Kotlin) → ViewModel (StateFlow) → NativeBridge (JNI) → llama.cpp (production)
                                                              → CustomRuntime (oracle/kernel lab)
```

### Key Source Locations

- **UI**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/`
- **ViewModel & State**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- **JNI Bridge**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- **llama.cpp**: `native/third_party/llama.cpp/` (git submodule)
- **Custom Runtime**: `native/custom/` — owned inference engine (kernel lab only)
- **Custom Runtime Headers**: `native/custom/include/ondevai/`
- **llama.cpp Build Script**: `scripts/llama_cpp/build_android.sh`
- **llama.cpp Benchmark Script**: `scripts/llama_cpp/benchmark_android.sh`
- **JSONL Normalizer**: `scripts/llama_cpp/normalize_llama_cli.py`
- **JSONL Comparator**: `scripts/benchmarks/compare_jsonl.py`
- **Packer Script**: `scripts/custom_runtime/pack_tinyllama.py`
- **Reference Tools**: `native/custom/tools/run_reference.cpp`

### Custom Runtime Structure (Kernel Lab, Not Production)

```
native/custom/
  include/ondevai/    # Headers: runtime.h, model.h, tensor.h, tokenizer.h, sampler.h, kv_cache.h
  src/                # Implementation: runtime.cpp, model_loader.cpp, layers.cpp, etc.
  tools/              # Desktop reference tools (run_reference.cpp, dump_reference_tensors.cpp)
  tests/              # Unit tests: test_rmsnorm.cpp, test_rope.cpp, test_matvec.cpp
```

### Custom Runtime Tools (Built via CMake from native/build-custom/)

```bash
./run_reference --model packed_tinyllama.bin --prompt "Hello world"
./test_rmsnorm
./test_rope
./test_matvec
./dump_reference_tensors --model packed_tinyllama.bin --prompt "test"
```

### Native Build Structure

- `native/build-llama/` — llama.cpp build tree (EXCLUDE_FROM_ALL)
- `native/build-custom/` — custom runtime build tree
- `native/jni/jni_bridge.cpp` — JNI entry point linking UI to native engines
- `native/engine/src/app_engine.cpp` — app engine implementation

ARM SIMD flags: `GGML_CPU_ARM_ARCH="armv8.5-a+fp16+i8mm+dotprod"` for optimal performance.

### Backend Status

**llama.cpp (production):** CPU-only on Android. OpenCL disabled due to Android 16 namespace isolation (`libOpenCL.so` dlopen failure).

**Custom runtime (lab):** Android CPU target for kernel experiments only.

**QNN (sidecar):** Preserved baseline. Not mainline until llama.cpp baseline is fully characterized.

## Benchmark Infrastructure

### Fixed Protocol

- 3 prompts in `docs/custom_runtime_benchmark_prompts.jsonl`: `short_fact`, `reasoning_small`, `chat_medium`
- Greedy decode, max new tokens = 64, context cap = 512
- Warmup runs = 1, measured runs = 5, report median

### JSONL Schema (schema_version 1)

Key fields: `backend`, `model_id`, `quant_format`, `prompt_id`, `ttft_ms`, `decode_tok_per_sec`, `generated_tokens`, `elapsed_ms`, `threads`, `cpu_mask`, `llama_cpp_commit`, `status`

### Benchmark Output Location

- `files/benchmarks/llama_cpp_android.jsonl` — llama.cpp Android results
- `files/benchmarks/custom_runtime_history.jsonl` — custom runtime results
- `.artifacts/llama-cpp/runs/<timestamp>/` — per-run log artifacts

### compare_jsonl.py

Supports `--model-id`, `--prompt-id`, `--status ok`, `--latest-only`, and `--since` filters. Use `--status ok --latest-only` to get clean latest results, excluding old failed rows and single runs.

## Current Benchmark Baselines (Snapdragon 8 Elite)

| Model | Quant | Median Decode | Median TTFT |
|-------|-------|---------------|-------------|
| Qwen2.5 1.5B | Q4_K_M | ~53 tok/s | ~134 ms |
| Qwen3.5 4B | Q4_K_M | ~15 tok/s | ~950 ms |

Best settings tested: `threads=6, cpu_mask=0xff` for both models.

## Model & Packed Format

**llama.cpp production models:** GGUF format, Q4_K_M primary quantization.

**Custom runtime format:** Own FP16 packed format in `model.bin` + `tokenizer/` + `manifest.json`. Created by `scripts/custom_runtime/pack_tinyllama.py`.

**Target model:** TinyLlama-1.1B-Chat-v1.0 for custom runtime kernel lab.

## Android 16 OpenCL Namespace Isolation (Known Issue)

On Android 16 / SDK 36, the Qualcomm vendor `libOpenCL.so` cannot be loaded:

```
dlopen failed: library "libcutils.so" not found: needed by ... libOpenCL.so in namespace clns-9
```

Resolution: `GGML_OPENCL=OFF`. CPU-only operation.

## Important Constraints

- **arm64-v8a only** — no other ABIs
- **minSdk 31 / targetSdk 35**
- App is a **performance lab**, not a consumer product
- llama.cpp is production inference; custom runtime is oracle/lab only
- Optimization work must beat the same-protocol baseline to earn its place
