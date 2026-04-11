# CPU Performance Fix Plan — From ~1 tok/s to 8-20 tok/s

## Status

This file documents a completed CPU performance recovery phase.

It is a reference document, not the current active plan.

For the current active plan, use:

- `docs/current_execution_plan_2026-04-10.md`

Use this file for:

- root cause of the early CPU performance collapse
- fixes that recovered CPU throughput
- historical CPU optimization context

## Problem Statement

After enabling ARM SIMD flags (`GGML_CPU_ARM_ARCH="armv8.5-a+fp16+i8mm+dotprod"`), CPU inference
on the Snapdragon 8 Elite improved from ~0.46 tok/s to ~1.0 tok/s. This is still far below the
expected 8-20+ tok/s for a 2B Q4 model on Cortex-X4/A720 cores.

Root cause analysis identified three disabled build flags and one runtime misconfiguration that
together account for the ~10-20x performance gap.

---

## Fix Overview

| # | Fix | File | Expected Impact | Risk |
|---|-----|------|----------------|------|
| 1 | Enable `GGML_LLAMAFILE` | `native/CMakeLists.txt` | **3-5x** | Low |
| 2 | Enable `GGML_OPENMP` | `native/CMakeLists.txt` | **1.5-2.5x** | Low-Medium |
| 3 | Auto-resolve thread count | `MainViewModel.kt` | **~1.3x** | Low |
| 4 | Enable `GGML_CPU_KLEIDIAI` (stretch) | `native/CMakeLists.txt` | **Additional boost** | Medium |

**Combined expected result: ~5-15 tok/s minimum, potentially 15-20+ tok/s**

---

## Fix 1: Enable GGML_LLAMAFILE (Critical — Largest Single Impact)

### What it does

llamafile contributes optimized SGEMM (single-precision general matrix multiply) implementations
to llama.cpp. These are hand-tuned matrix multiplication kernels with specializations for ARM NEON.
Since matmul dominates the decode compute path (~70-85% of inference time), this is by far the
highest-impact single change.

Without GGML_LLAMAFILE, llama.cpp falls back to generic C implementations of key matmul operations,
which do not exploit the NEON pipeline efficiently even when SIMD compilation flags are present.

### What was wrong

```cmake
# native/CMakeLists.txt line 12
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
```

This was likely set OFF during initial bring-up to simplify the build and reduce the number of
moving parts. At the time, the priority was getting a working build, not performance tuning.

### The fix

```cmake
# Change from OFF to ON
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
```

### What changes in the build

When enabled, the CPU backend CMake includes two additional source files:
- `ggml-cpu/llamafile/sgemm.cpp` — the optimized SGEMM implementation
- `ggml-cpu/llamafile/sgemm.h` — header

And defines `GGML_USE_LLAMAFILE` which activates the optimized paths at compile time.

### Risk assessment

**Low.** The llamafile SGEMM is well-tested in llama.cpp and is the default on most platforms.
It compiles cleanly with the NDK clang for arm64-v8a. No runtime dependencies.

### Validation

- Build should succeed with no new warnings or errors
- Run smoke benchmark before and after
- Expected: decode tok/s jumps from ~1.0 to ~3-5

---

## Fix 2: Enable GGML_OPENMP (High Impact — Parallel Inner Loops)

### What it does

OpenMP enables parallel execution of inner computation loops within ggml tensor operations. While
llama.cpp has its own threading model (controlled by `n_threads`), many of the hot inner loops use
`#pragma omp parallel for` to distribute work across threads. Without OpenMP, these directives are
silently ignored and the loops run single-threaded.

This means even though we pass `threadCount=4`, the actual heavy computation (quantized dot products,
matmul accumulations) runs on a single core.

### What was wrong

```cmake
# native/CMakeLists.txt lines 10-11
set(LLAMA_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
```

OpenMP was disabled, likely to avoid an extra shared library dependency during initial bring-up.

### The fix

```cmake
# Change both from OFF to ON
set(LLAMA_OPENMP ON CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
```

### What changes in the build

- CMake will `find_package(OpenMP)` and link against `OpenMP::OpenMP_C` and `OpenMP::OpenMP_CXX`
- The NDK includes `libomp.so` — the LLVM OpenMP runtime for Android
- `GGML_USE_OPENMP` is defined, activating parallel regions in the ggml CPU backend
- The `libomp.so` from the NDK will be packaged into the APK alongside the native library

### Risk assessment

**Low-Medium.**

- The NDK (27.3) ships with full OpenMP support (`libomp`)
- The APK packaging should automatically include the NDK's `libomp.so` via the CMake build
- Potential issue: if the Gradle/CMake integration doesn't automatically bundle `libomp.so`, the
  app will crash at startup with `dlopen failed: library "libomp.so" not found`
- Fix for that: ensure the CMake shared library `ondevai_native` links against OpenMP, which should
  cause the NDK build system to include `libomp.so` in the APK

### Verification steps

1. Build the APK
2. Check that `libomp.so` is inside the APK:
   ```bash
   unzip -l app/android/app/build/outputs/apk/debug/app-debug.apk | grep libomp
   ```
   Should show `lib/arm64-v8a/libomp.so`
3. If missing, add to `build.gradle.kts`:
   ```kotlin
   packaging {
       jniLibs {
           useLegacyPackaging = true
       }
   }
   ```
4. Install and run, check logcat for OpenMP init
5. Run smoke benchmark — should see improvement from multi-core compute

### Expected impact

With 4+ threads actually utilized in inner loops: **1.5-2.5x** speedup.

---

## Fix 3: Auto-Resolve Thread Count (Moderate Impact)

### What it does

Uses the device's actual core count to determine the optimal number of inference threads, rather
than relying on a hardcoded value that was set during development.

### What was wrong

In `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`, lines 95 and 165:

```kotlin
threadCount = 4,
```

The thread count is hardcoded to 4 in two places (initial load and reload). The native
`ResolveThreadCount` function (app_engine.cpp:581) already handles auto-resolution:

```cpp
int AppEngine::ResolveThreadCount(const int requested_threads) const {
    if (requested_threads > 0) {
        return requested_threads;  // <-- 4 is passed, so this always returns 4
    }
    const long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpu_count <= 2) {
        return 2;
    }
    return static_cast<int>(std::max(2L, cpu_count - 2));  // 8 - 2 = 6 on S8 Elite
}
```

By passing 4, the auto-resolution is bypassed. Passing 0 would trigger auto-resolution to 6 threads
on the Snapdragon 8 Elite.

### The fix

Change both occurrences in MainViewModel.kt:
```kotlin
// Before
threadCount = 4,

// After
threadCount = 0,  // auto-resolve: uses (core_count - 2) in native code
```

### Optimal thread count considerations for Snapdragon 8 Elite

The S8 Elite core layout:
- 2x Cortex-X4 (prime cores, highest performance)
- 6x Cortex-A720 (performance cores)

The auto-resolve formula `cpu_count - 2 = 6` is reasonable. However, for decode (which is
memory-bandwidth-bound), having too many threads can cause cache thrashing. The optimal count
for decode is often lower than for prompt processing.

For now, auto-resolve (6 threads) is a good starting point. We can tune after measuring.

### Risk assessment

**Low.** The auto-resolve logic is already implemented and tested. Passing 0 simply lets it work.

### Expected impact

~1.3x improvement as more cores participate in compute.

---

## Fix 4: Enable GGML_CPU_KLEIDIAI (Stretch Goal — ARM-Optimized Microkernels)

### What it does

KleidiAI is ARM's official library of highly optimized compute microkernels, specifically tuned for
modern ARM cores including Cortex-X4 and A720. It provides hand-written assembly routines for:

- Q4 quantized matmul with dotprod
- Q4 quantized matmul with i8mm (Int8 matrix multiply)
- Pack/repack operations optimized for the cache hierarchy

These routines are written to exploit the exact microarchitecture of the cores in the S8 Elite.

### The fix

Add to `native/CMakeLists.txt`:
```cmake
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
```

### Build implications

- KleidiAI sources are fetched via CMake `FetchContent` from GitHub at build time
  (https://github.com/ARM-software/kleidiai, tag v1.22.0)
- This requires internet access during the first build
- Adds assembly (.S) files to the compilation for ARM-specific kernels
- The kernels are selected at build time based on the ARCH_FLAGS (our `armv8.5-a+fp16+i8mm+dotprod`
  would activate dotprod and i8mm variants)

### Risk assessment

**Medium.**

- Fetch at build time may fail in offline environments
- Assembly routines are architecture-specific — should be fine for arm64-v8a only
- The KleidiAI integration is relatively recent in llama.cpp; may have edge cases
- If build fails, easy to revert to OFF

### When to try

After Fixes 1-3 are validated and baseline is measured. This is an incremental optimization on top.

### Expected impact

Potentially another 1.5-2x on top of the llamafile SGEMM, particularly for Q4_0 quantization
with i8mm, where the ARM assembly routines can approach peak throughput.

---

## Execution Order

### Phase 1: Quick Wins (Fixes 1 + 3)

These are pure one-line changes with zero dependency risk:

1. Edit `native/CMakeLists.txt`: `GGML_LLAMAFILE OFF` → `ON`
2. Edit `MainViewModel.kt`: `threadCount = 4` → `threadCount = 0` (both occurrences)
3. Build: `./gradlew :app:assembleDebug`
4. Install: `./scripts/install_debug.sh`
5. Run smoke benchmark on device
6. Record results in benchmark history

**Expected result: ~3-5 tok/s**

### Phase 2: OpenMP (Fix 2)

Slightly higher risk due to the `libomp.so` packaging dependency:

1. Edit `native/CMakeLists.txt`: `GGML_OPENMP OFF` → `ON`, `LLAMA_OPENMP OFF` → `ON`
2. Build: `./gradlew :app:assembleDebug`
3. Verify `libomp.so` is in the APK: `unzip -l ... | grep libomp`
4. If missing, update `build.gradle.kts` packaging config
5. Install and run smoke benchmark
6. If startup crash (libomp not found), investigate NDK OpenMP packaging
7. Record results

**Expected result: ~5-12 tok/s**

### Phase 3: KleidiAI (Fix 4)

Stretch goal, try after Phase 1-2 baselines are established:

1. Edit `native/CMakeLists.txt`: add `GGML_CPU_KLEIDIAI ON`
2. Build (requires internet for first fetch)
3. Install and run standard benchmark
4. Compare against Phase 2 baseline
5. Record results

**Expected result: ~8-20 tok/s (realistic CPU ceiling)**

---

## Benchmark Protocol

For each phase, run the following sequence:

```
1. Clean build:       ./gradlew clean :app:assembleDebug
2. Install:           ./scripts/install_debug.sh
3. Wait 30 seconds for thermal stabilization
4. Run smoke benchmark from Developer screen
5. Run standard benchmark from Developer screen
6. Record: tok_per_sec, ttft_ms, elapsed_ms for each entry
7. Note: model_label and threads from benchmark output
```

Compare against current baseline:
- **Baseline (current):** ~0.9-1.0 tok/s, threadCount=4, LLAMAFILE=OFF, OPENMP=OFF
- **Phase 1 target:** ~3-5 tok/s
- **Phase 2 target:** ~5-12 tok/s
- **Phase 3 target:** ~8-20 tok/s

---

## Files to Modify

| File | Changes |
|------|---------|
| `native/CMakeLists.txt` | Lines 10-12: OPENMP ON, LLAMAFILE ON; optionally add KLEIDIAI ON |
| `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt` | Lines 95, 165: threadCount 4 → 0 |

Total: **2 files, 5 line changes** for the full plan.

---

## Rollback

If any fix causes issues:
- Each fix is independent and can be reverted individually
- The original values are documented above
- Build and test after each phase before moving to the next

---

## Model Considerations

The plan assumes a ~2B parameter Q4-quantized model (per the project plan's "1B to 1.5B for fast
iteration, 3B as stress"). If the actual model being tested is larger (e.g., 7B), the expected
tok/s numbers should be scaled down roughly proportionally. The relative improvement ratios remain
the same regardless of model size.

To verify which model is loaded, check the `model_label` field in benchmark output, or the
`llama_model_desc` output in logcat (`adb logcat -s OnDevAI`).
