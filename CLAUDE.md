# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OnDevAI is a **Snapdragon 8 Elite hardware-specific LLM inference app** focused on measuring and optimizing token generation performance. It combines a Kotlin/Compose Android UI with a native C++ inference engine.

**Current direction:** The custom runtime (under `native/custom/`) is the mainline. The llama.cpp engine is preserved but secondary. TinyLlama-1.1B-Chat-v1.0 is the target model.

Target hardware: **arm64-v8a only, Snapdragon 8 Elite with Adreno GPU**

## Build Commands

```bash
# Build debug APK (from project root)
./gradlew :app:assembleDebug

# Install APK to connected device
./scripts/install_debug.sh

# Push a GGUF model to the device
./scripts/adb_push_model.sh /path/to/model.gguf

# Build native library only (for fast iteration on C++ code)
./gradlew :app:assembleDebug

# View runtime diagnostics on device
adb logcat -s OnDevAI

# Run custom runtime tests (from native/build-llama or via cmake)
# Tests are in native/custom/tests/

# Pack TinyLlama to custom runtime format
python3 scripts/custom_runtime/pack_tinyllama.py /path/to/TinyLlama-1.1B-Chat-v1.0

# Run desktop reference tool
./native/build-llama/run_reference --model packed_tinyllama.bin --prompt "Hello world"
```

## Architecture

```
Compose UI (Kotlin) → ViewModel (StateFlow) → NativeBridge (JNI) → CustomRuntime (C++)
                                                              ↓
                                                        llama.cpp / ggml (legacy)
```

### Key Source Locations

- **UI**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/`
- **ViewModel & State**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- **JNI Bridge**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- **Custom Runtime**: `native/custom/` — owned inference engine
- **Custom Runtime Headers**: `native/custom/include/`
- **Packer Script**: `scripts/custom_runtime/pack_tinyllama.py`
- **Reference Tools**: `native/custom/tools/run_reference.cpp`

### Custom Runtime Structure

```
native/custom/
  include/           # Headers: runtime.h, model.h, tensor.h, tokenizer.h, sampler.h, kv_cache.h
  src/               # Implementation: runtime.cpp, model_loader.cpp, layers.cpp, etc.
  tools/             # Desktop reference tools (run_reference.cpp)
  tests/             # Unit tests: test_rmsnorm.cpp, test_rope.cpp, test_matvec.cpp
```

### Native Build

The native layer uses CMake with the build tree at `native/build-llama`. The custom runtime is built as part of the main CMake configuration. ARM SIMD flags are critical: `GGML_CPU_ARM_ARCH="armv8.5-a+fp16+i8mm+dotprod"` for optimal performance.

### Backend Selection

The custom runtime targets Android CPU first. The legacy llama.cpp path supports "cpu" or "opencl" backends via `AppEngine::FindOpenClDeviceLocked()`. OpenCL is disabled due to Android 16 namespace isolation (see below).

## Model & Packed Format

**Target Model:** TinyLlama-1.1B-Chat-v1.0 (`/home/curious/models/TinyLlama-1.1B-Chat-v1.0/`)

**Packed Format v1:** Own FP16 format (not GGUF). Created by `scripts/custom_runtime/pack_tinyllama.py`.

Format artifacts:
- `model.bin` — FP16 weights with header + tensor directory
- `tokenizer/` — tokenizer model/config
- `manifest.json` — metadata

## Performance Metrics

Benchmark results persisted to `files/benchmarks/history.jsonl` as JSON lines:
- **TTFT**: `ttft_ms`
- **Decode throughput**: `tok_per_sec`
- **Total elapsed**: `elapsed_ms`

Benchmark modes: `smoke` (1 short prompt), `standard` (3 prompts of varying length)

## CPU Benchmark Performance (Snapdragon 8 Elite, TinyLlama-1.1B-Q4_K_M via llama.cpp)

~1.0 tok/s with `GGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm+dotprod`. The custom runtime is the target for higher throughput.

## Android 16 OpenCL Namespace Isolation (Known Issue)

On Android 16 / SDK 36, the Qualcomm vendor `libOpenCL.so` cannot be loaded due to namespace isolation:

```
dlopen failed: library "libcutils.so" not found: needed by ... libOpenCL.so in namespace clns-9
```

Current resolution: `GGML_OPENCL=OFF` in the build. CPU-only operation.

## Important Constraints

- **arm64-v8a only** — no other ABIs are configured or tested
- **minSdk 31 / targetSdk 35**
- The app is a **performance lab**, not a consumer product