# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OnDevAI is a **Snapdragon 8 Elite hardware-specific LLM inference app** focused on measuring and optimizing token generation performance. It combines a Kotlin/Compose Android UI with a native C++ inference engine built on llama.cpp.

Target hardware: **arm64-v8a only, Snapdragon 8 Elite with Adreno GPU**
Primary acceleration path: **OpenCL via Adreno GPU**
Secondary/validation path: **CPU**

## Build Commands

```bash
# Build debug APK (from project root)
./gradlew :app:assembleDebug

# Install APK to connected device
./scripts/install_debug.sh

# Push a GGUF model to the device
./scripts/adb_push_model.sh /path/to/model.gguf

# Build native library only (for fast iteration on C++ code)
./gradlew :app:assembleDebug  # rebuilds native via CMake

# View runtime diagnostics on device
adb logcat -s OnDevAI
```

## Architecture

```
Compose UI (Kotlin) → ViewModel (StateFlow) → NativeBridge (JNI) → AppEngine (C++)
                                                              ↓
                                                        llama.cpp / ggml
                                                              ↓
                                                    CPU ←→ GPU_OPENCL (disabled) ←→ HTP (experimental)
```

### Key Source Locations

- **UI**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/`
- **ViewModel & State**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- **JNI Bridge**: `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- **Native Engine**: `native/engine/src/app_engine.cpp` — owns model lifecycle, generation loop, benchmark execution
- **JNI Entry Points**: `native/jni/jni_bridge.cpp`
- **Native Headers**: `native/engine/include/app_engine.h`

### Native Build

The native layer uses CMake with the build tree at `native/build-llama`. The `ondevai_native` shared library links against `llama.cpp` (built from `native/third_party/llama.cpp`). **GGML_OPENCL is currently OFF** in `native/CMakeLists.txt` due to Android 16 namespace isolation (see below).

**Critical ARM CPU flags:** `GGML_CPU_ARM_ARCH="armv8.5-a+fp16+i8mm+dotprod"` must be set for SIMD optimization. Without this (or with `GGML_NATIVE=OFF` alone), no ARM64 NEON/SIMD instructions are compiled, resulting in ~0.5 tok/s instead of ~1.0 tok/s on Snapdragon 8 Elite.

### Backend Selection

Backend target ("cpu" or "opencl") is passed at model load time. The `AppEngine::FindOpenClDeviceLocked()` method scans `ggml_backend_dev_count()` for a device named "GPUOpenCL" or registered under "OpenCL". If the requested backend is unavailable, model loading fails with a diagnostic message — no silent fallback occurs. The Settings screen detects whether OpenCL is in the backend inventory and disables the "Use OpenCL" button when unavailable.

### Model Storage

Models must be accessible from native code via internal app storage (`files/models/` under the app UID). The `MainViewModel::prepareModelForNativeLoad()` copies from external paths to internal storage to ensure reliable native access. Model path is printed in diagnostics at `app: loadStatus`.

## Performance Metrics

Benchmark results are persisted as JSON lines to `files/benchmarks/history.jsonl` in internal storage. Key metrics:
- **TTFT** (Time To First Token): `ttft_ms` in benchmark output
- **Decode throughput**: `tok_per_sec`
- **Total elapsed**: `elapsed_ms`

Benchmark modes: `smoke` (1 short prompt), `standard` (3 prompts of varying length)

## CPU Benchmark Performance (Snapdragon 8 Elite, Qwen3-2B-Q4_K_M)

With `GGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm+dotprod` and CPU at 2.4 GHz max:
- **smoke**: ~0.9-1.0 tok/s
- **standard short**: ~0.97 tok/s
- **standard medium**: ~0.90 tok/s
- **standard long**: ~0.86 tok/s

Without ARM SIMD flags (bare `GGML_NATIVE=OFF`), performance was ~0.45-0.5 tok/s — a 2x regression.

The ~1 tok/s ceiling is the genuine limit of the CPU-only path. GPU acceleration via Adreno is the target for higher throughput, but is blocked by Android 16 namespace isolation (see below).

## Android 16 OpenCL Namespace Isolation (Known Issue)

On the target device (Android 16 / SDK 36, custom ROM), the Qualcomm vendor `libOpenCL.so` (ICD loader) links against `libcutils.so` which is inaccessible in a third-party app's namespace:

```
dlopen failed: library "libcutils.so" not found: needed by ... libOpenCL.so in namespace clns-9
```

The vendor library stack (`/vendor/lib64/libOpenCL.so`, `libOpenCL_adreno.so`, `libcutils.so`) is present and self-contained, but the Android 16 namespace isolation prevents third-party apps from accessing `/system/lib64/` where `libcutils.so` lives.

Current resolution: `GGML_OPENCL=OFF` in the build. The app runs on CPU only. The Settings screen shows a clear diagnostic explaining the situation.

Potential fixes to investigate:
1. Use `android_dlopen_ext` from a native shim to load the vendor library with an explicit namespace that includes `/system/lib64`
2. Test on a device/ROM with more permissive namespace configuration
3. Root access to modify the namespace configuration

## Important Constraints

- **arm64-v8a only** — no other ABIs are configured or tested
- **minSdk 31 / targetSdk 35** (built against SDK 35)
- The app is a **performance lab**, not a consumer product — UI polish is intentionally minimal
