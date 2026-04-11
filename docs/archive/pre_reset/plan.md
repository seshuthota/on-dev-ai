# Snapdragon 8 Elite LLM Performance App — Execution Plan

## Status

This is the original master plan and historical baseline document.

It is no longer the canonical active execution plan.

For the current active plan, use:

- `docs/current_execution_plan_2026-04-10.md`

Use this file for:

- original project intent
- CPU/OpenCL/Vulkan history
- baseline measurement history
- early milestone context

## 1. Objective

Build an Android app for a **single Snapdragon 8 Elite target class** that runs a local LLM and pushes **token generation performance** as far as the hardware will allow.

This project is not a generic on-device AI app.

This project is:

- a **hardware-specific inference app**
- a **performance lab with a usable chat UI**
- a place to make **device-specific runtime and backend changes**
- a framework for measuring whether each optimization actually improves TTFT, decode speed, and sustained behavior

The app exists to support optimization work. The UI is necessary, but performance engineering is the actual product goal.

---

## 2. Core Decision

We are optimizing for **one hardware family first**, not for portability.

Primary assumptions:

- target SoC: **Snapdragon 8 Elite**
- target OS: modern Android on a real device
- target ABI: **arm64-v8a only**
- primary acceleration path: **Adreno GPU via OpenCL**
- secondary validation path: **CPU**
- experimental path: **Hexagon / HTP**, only if measurement shows a real benefit

Non-goals for v1:

- broad Android device compatibility
- vendor-neutral backend abstractions for every chipset
- polished consumer product features
- shipping to arbitrary phones
- complex multi-model catalog support

If a design decision helps portability but slows down hardware-specific optimization, portability loses.

---

## 3. Success Criteria

The project is successful only if it produces a working app plus a reproducible optimization workflow.

Primary success metrics:

- low **TTFT**
- high **decode tok/s**
- stable sustained decode over **3 to 5 minute runs**
- acceptable thermal behavior during repeated runs
- minimal overhead between native generation and visible streamed output

Secondary metrics:

- prompt tok/s
- model load time
- memory footprint
- app responsiveness during generation

Hard rule:

- no optimization is considered real unless it is measured against a fixed benchmark configuration

---

## 3A. Progress Update

Current status as of the latest on-device validation:

- Sprint 1 completed
- Sprint 2 completed
- Sprint 3 completed
- Sprint 4 completed
- Sprint 5 completed
- CPU performance optimization completed
- GGML_VULKAN experiment completed (blocked by Adreno driver incompatibility)
- HTP work not started yet

What is working now:

- Android app shell builds and installs on the target device
- Compose UI has Chat, Settings, and Developer screens
- JNI bridge is stable enough for runtime init, model load, generation, stop, and benchmark calls
- `llama.cpp` is integrated into the native build
- one real GGUF model loads from internal app storage on device
- CPU-only generation works on device at **~28-30 tok/s** for Qwen3.5-2B Q4_K_M
- token streaming works live in the UI
- stop/cancel interrupts native generation cleanly
- smoke and standard benchmark modes exist in the app
- benchmark history is persisted locally as structured JSON lines
- the app exposes CPU backend selection in Settings (Vulkan and OpenCL UI is present but disabled by default)
- runtime diagnostics log to logcat via `adb logcat -s OnDevAI`

What was learned during execution:

- loading GGUF directly from Android external app storage was unreliable for native access on the target device
- the working solution is to load from internal app storage under the app UID
- the current target phone exposes `libOpenCL.so` and Adreno GPU libraries at the system level
- OpenCL is now loading on-device and `GPUOpenCL(QUALCOMM Adreno(TM) 830)` is visible in backend inventory
- the prior namespace failure (`libcutils.so` / `clns-9`) should be treated as environment-specific and not a universal Android 16 OpenCL block
- on this build (`targetSdk=35`), OpenCL initialization succeeds; current bottleneck moved from loader failure to runtime performance
- gaming apps using GPU acceleration (Vulkan/OpenGL ES) does not prove OpenCL path quality; API stacks and driver paths differ
- **Critical build lesson**: `./gradlew :app:assembleDebug` passes `CMAKE_BUILD_TYPE=Debug` to the NDK CMake toolchain, which compiles all native code with `-O0` (zero optimization). This caused a ~30x performance penalty. Fixed by adding `-DCMAKE_BUILD_TYPE=Release` to the cmake arguments in `build.gradle.kts`. Native code must always be compiled with optimization regardless of APK build type.
- Enabling `GGML_LLAMAFILE=ON` and `GGML_CPU_ARM_ARCH` flags had marginal impact under `-O0` but are necessary for full performance under `-O3`

Current measured CPU baseline (Qwen3.5-2B Q4_K_M, 6 threads, Snapdragon 8 Elite):

- cold-load (broadcast harness, CPU default + lazy accelerator loading): **~2.3-2.4 seconds**
- micro benchmark (8 decode): **22.53 tok/s**, TTFT 113ms, load 2344ms
- smoke benchmark (32 decode): **27.14 tok/s**, TTFT 163ms, load 2332ms
- standard benchmark (worker-run cold load): short **16.56 tok/s** (TTFT 141ms), medium **28.22 tok/s** (TTFT 186ms), long **26.75 tok/s** (TTFT 234ms), load 2949ms
- sustained decode rate: **~30-32 tok/s** (excluding TTFT)

CPU performance optimization progression:

- baseline (no SIMD, -O0): ~0.46 tok/s
- + ARM SIMD flags (-O0): ~1.0 tok/s
- + LLAMAFILE + auto threads (-O0): ~1.3 tok/s
- + CMAKE_BUILD_TYPE=Release (-O3): **~28-30 tok/s** (the dominant fix)

Current blocker to higher performance:

- OpenCL is functional but slower than CPU for this model/config because unsupported fused Gated Delta Net ops fall back to CPU, causing costly CPU<->GPU scheduling/copies
- latest OpenCL worker sweep (cold-start, ctx 2048):
  - micro (`8` decode):
    - `n_gpu_layers=4`: load 7376 ms, TTFT 287 ms, 8.11 tok/s
    - `n_gpu_layers=8`: load 7516 ms, TTFT 281 ms, 7.39 tok/s
    - `n_gpu_layers=12`: load 7915 ms, TTFT 306 ms, 6.88 tok/s
    - `n_gpu_layers=16`: load 8412 ms, TTFT 317 ms, 6.80 tok/s
  - smoke (`32` decode):
    - `n_gpu_layers=4`: load 7105 ms, TTFT 315 ms, 9.31 tok/s
    - `n_gpu_layers=8`: load 7518 ms, TTFT 432 ms, 7.15 tok/s
    - `n_gpu_layers=12`: load 8000 ms, TTFT 501 ms, 6.94 tok/s
    - `n_gpu_layers=16`: load 8304 ms, TTFT 559 ms, 6.70 tok/s
  - trend: best observed OpenCL point is currently `n_gpu_layers=4`; pushing more layers to GPU degrades both TTFT and tok/s
- non-broadcast long-run path is now implemented: `DEBUG_OPENCL_SMOKE` enqueues a `WorkManager` worker and receiver returns immediately
- reproducible runner script added: `scripts/adb_backend_benchmark_worker.sh` (worker-aware, timeout diagnostics, fallback signal counts)
- CPU `standard` now completes reliably through this worker path (no receiver lifetime drop)
- OpenCL `smoke` now completes across tested layer settings via worker (`n_gpu_layers=4/8/12/16`)
- OpenCL `standard` completes for lower offload settings:
  - `n_gpu_layers=4`: short/medium/long = 7.09 / 7.33 / 8.01 tok/s
  - `n_gpu_layers=8`: short/medium/long = 7.00 / 7.32 / 6.89 tok/s
- OpenCL `standard` is unstable for higher offload settings:
  - `n_gpu_layers=12` and `16` did not complete within 300s timeout windows and were interrupted/rescheduled by WorkManager
- `GGML_OPENCL_USE_ADRENO_KERNELS=ON` was tested and caused init hang at `ggml_opencl: loading OpenCL kernels`; reverted to `OFF`
- GGML_VULKAN remains blocked by Adreno 830 driver incompatibility (see Sprint 6 details below)

---

## 3B. Sprint 6 — GGML_VULKAN Experiment

**Status: completed, blocked by Adreno 830 driver incompatibility**

### Goal

Investigate GGML_VULKAN as an alternative GPU acceleration path on the iQOO 13 (Snapdragon 8 Elite, Android 16, custom ROM). Grok AI analysis confirmed Android 16 does not block Vulkan generally; the OpenCL issue is ROM-specific namespace hardening. Vulkan loader is treated as a standard graphics API and bypasses the ICD + libcutils.so namespace problem.

### Changes Made

**`native/CMakeLists.txt`** — Added Vulkan build configuration:
```cmake
set(GGML_VULKAN ON CACHE BOOL "" FORCE)
set(GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "${CMAKE_CURRENT_LIST_DIR}/cmake/host-Linux-x86_64.toolchain.cmake" CACHE FILEPATH "" FORCE)
set(Vulkan_GLSLC_EXECUTABLE "/path/to/ndk/.../glslc" CACHE FILEPATH "" FORCE)
set(Vulkan_FOUND TRUE CACHE BOOL "" FORCE)
set(Vulkan_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/Vulkan-Headers/include" CACHE PATH "" FORCE)
set(Vulkan_LIBRARY "..." CACHE FILEPATH "" FORCE)
# Advanced shader extensions disabled to avoid linker errors
set(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT OFF CACHE BOOL "" FORCE)
```

**`native/engine/src/app_engine.cpp`** — Added Vulkan backend support:
- `NormalizeBackendTarget()`: "vulkan" now maps to Vulkan backend (previously only "opencl" or "cpu")
- `BackendDisplayName()`: returns "Vulkan" for vulkan target
- `IsVulkanDevice()`: detects Vulkan backend devices by registry name "Vulkan" or device name "GPU"
- `FindVulkanDeviceLocked()`: finds the first available Vulkan device
- `AppEngine::LoadModel()`: Vulkan branch selects Vulkan device, sets `n_gpu_layers=-1`, enables KV offload to GPU
- `context_params`: `offload_kqv` and `op_offload` set true for Vulkan backend
- `app_engine.h`: declared `FindVulkanDeviceLocked()`

**`app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`** — Kotlin backend support:
- `normalizeBackendTarget()`: accepts "vulkan" as valid backend
- `backendDisplayName()`: returns "Vulkan" for vulkan target

**`app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/SnapdragonLabApp.kt`** — UI changes:
- Added "Use Vulkan" button in Settings screen (enabled when Vulkan backend detected in runtime info)
- Added `isVulkanAvailable` detection (checks `runtimeInfo.contains("Vulkan")`)
- Updated "Backend status" card to show both OpenCL and Vulkan availability
- Removed old "OpenCL status" diagnostic card

**`native/cmake/host-Linux-x86_64.toolchain.cmake`** — New file:
- Host toolchain for building `vulkan-shaders-gen` (which runs on Linux x86_64, not ARM)
- Sets `CMAKE_MAKE_PROGRAM` to the NDK's Ninja binary
- Sets host compilers to system gcc/g++

**`native/third_party/Vulkan-Headers/`** — New git clone:
```bash
git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git \
  native/third_party/Vulkan-Headers
```
Full repo required (not just vulkan.hpp) because it includes `vulkan_hpp_macros.hpp` which the single-header download lacked.

### Build Infrastructure Changes Required

The `vulkan-shaders-gen` sub-build (compiles SPIR-V shaders at build time) required several fixes:

1. **Host toolchain for cross-compilation**: ggml-vulkan uses `ExternalProject_Add()` for shader generation, which builds on the host machine. The NDK CMake toolchain needed an explicit `CMAKE_MAKE_PROGRAM` pointing to Ninja.

2. **Vulkan headers**: Android NDK only includes C Vulkan headers (`vulkan.h`). The C++ Vulkan bindings (`vulkan.hpp`) required the full Vulkan-Headers git repo.

3. **Vulkan library**: NDK provides `libvulkan.so` for API level 24+ at `toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/35/libvulkan.so`.

4. **glslc**: The NDK shader tools include `glslc` at `shader-tools/linux-x86_64/glslc`.

5. **Shader extension detection bypass**: `test_shader_extension_support()` in ggml-vulkan CMake runs `glslc` at CMake configure time to detect GPU capabilities. With `Vulkan_GLSLC_EXECUTABLE` set explicitly, this works without Vulkan SDK installed.

### What Worked

- Build succeeded with GGML_VULKAN=ON, all shaders compiled to SPIR-V
- App installed on device successfully
- Vulkan backend detected in `ggml_backend_dev_count()`: `backend_inventory=Vulkan0(Adreno (TM) 830, reg=Vulkan, type=igpu); CPU(CPU, reg=CPU, type=cpu)`
- Model loaded using Vulkan device: `using device Vulkan0 (Adreno (TM) 830)`
- KV cache allocated on Vulkan: `Vulkan0 KV buffer size = 24.00 MiB`
- GPU memory reserved: `Vulkan0 compute buffer size = 489.00 MiB`
- Graph scheduled: `graph nodes = 1377, graph splits = 2`

### Crash During Generation

On first token generation (first `llama_decode` call), the app crashed with:

```
SIGSEGV, code 1 (SEGV_MAPERR), fault addr 0x8
  #00 vkCmdBindPipeline in vulkan.adreno.so (qglinternal::vkCmdBindPipeline+4)
  #01 ggml_backend_sched_graph_compute_async in libondevai_native.so
  #02 llama_context::graph_compute in libondevai_native.so
  #03 llama_decode in libondevai_native.so
```

- `fault addr 0x8` = attempting to dereference offset 8 from a null pointer (null pipeline object passed to `vkCmdBindPipeline`)
- Crash occurred both **with** advanced shader extensions (coopmat, dotprod, bfloat16) and **without** them
- `AdrenoVK-0: Failed to link shaders` was logged before the first crash (shader linking failure)
- Second attempt (with advanced extensions disabled): shader linking appeared to succeed but `vkCmdBindPipeline` still crashed

### Root Cause

This is an **Adreno 830 driver-level bug or incompatibility** with ggml-vulkan's pipeline creation on this specific custom Android 16 ROM. The SIGSEGV happens inside the Qualcomm vendor driver (`vulkan.adreno.so`), not in our code or even in ggml-vulkan. The pipeline object created by ggml-vulkan appears to be invalid or incompatible with the driver's `vkCmdBindPipeline` implementation.

This is distinct from the OpenCL namespace isolation issue — it is a Vulkan driver compatibility problem specific to this ROM/kernel/driver combination.

### Outcome

- **GGML_VULKAN disabled** (`GGML_VULKAN=ON` removed from CMakeLists.txt)
- App reverted to CPU-only mode
- All Vulkan code (detection, device finding, UI button) remains in the codebase but is inactive
- Vulkan device detection and loading work correctly; the crash is purely at compute time

### How to Re-Enable Vulkan

When a fix is available (different ROM, driver update, or ggml-vulkan fix):
1. Add back the CMake Vulkan flags (see above)
2. The UI already has the "Use Vulkan" button
3. Rebuild and test generation

### Potential Fixes to Investigate

1. Test on stock OriginOS 6 (iQOO 13 with unmodified ROM) — the driver incompatibility may be specific to the custom ROM's kernel
2. Try with different `n_gpu_layers` values (partial offload instead of full)
3. Use `GGML_VULKAN_DEBUG=ON` to get more ggml-vulkan internal diagnostics
4. Check if a newer version of llama.cpp has Adreno 830 fixes
5. Root access + Magisk module to relax namespace isolation for clns-9/sphal (would enable OpenCL instead)

---

## 4. Product Shape

The app should stay intentionally narrow.

Required user-facing capabilities:

- local chat with streamed output
- start and stop generation
- fixed-model or limited-model selection
- backend selection for testing
- developer benchmark screen
- diagnostics and structured logs

Nice to have later:

- multi-model management
- conversation persistence
- runtime presets
- profile export/import

Not required for initial execution:

- prompt templates
- packaged model marketplace
- generic device capability discovery for many chipsets

---

## 5. Design Principle

Build the app as a **trusted experimentation platform** first.

That means the implementation order is:

1. bootable app
2. reliable JNI bridge
3. native CPU baseline
4. streaming generation
5. benchmark harness
6. OpenCL validation
7. OpenCL optimization
8. HTP experiments only after the GPU path is measured and stable

We do not start with advanced backend work before we can prove that a change improved a known baseline.

---

## 6. Architecture

```mermaid
flowchart TD
    UI[Compose UI] --> VM[ViewModel / StateFlow]
    VM --> JNI[JNI Bridge]
    JNI --> APPCORE[Native App Engine]
    APPCORE --> LLAMA[llama.cpp / ggml]
    APPCORE --> METRICS[Metrics + Benchmarking]
    APPCORE --> LOGS[Structured Diagnostics]

    LLAMA --> CPU[CPU Backend]
    LLAMA -.-> OCL[Adreno OpenCL Backend] -.dashed.-> "working, but slower than CPU (op fallback)"
    LLAMA -.-> VK[Adreno Vulkan Backend] -.dashed.-> "blocked (Adreno 830 driver)"
    LLAMA --> HTP[Hexagon HTP Experimental]
```

Architecture constraints:

- JNI stays thin
- app-owned engine API wraps llama.cpp
- backend selection and fallback happen in native code
- metrics collection is built into generation and benchmarking paths
- UI reflects native state, not inferred state

---

## 7. Repository Layout

```text
app/
  android/
    app/
      src/main/java/.../
      src/main/res/
      src/main/AndroidManifest.xml
      build.gradle.kts
    gradle/
    settings.gradle.kts
    build.gradle.kts

native/
  engine/
    include/
    src/
      app_engine.cpp
      runtime_config.cpp
      model_runner.cpp
      generation_session.cpp
      backend_selector.cpp
      benchmark_runner.cpp
      metrics_logger.cpp
      diagnostics.cpp
  jni/
    jni_bridge.cpp
  third_party/
    llama.cpp/

models/
  README.md
  profiles/
    primary_benchmark_model.json
    stress_model.json

scripts/
  android_build.sh
  adb_push_model.sh
  run_benchmark_suite.sh
  collect_device_logs.sh

docs/
  benchmark_protocol.md
  optimization_journal.md
  backend_notes.md
  target_device.md
```

---

## 8. Native-First Runtime Plan

The native layer is the center of the system.

Core native responsibilities:

- initialize runtime
- load model
- select backend
- execute prompt ingestion
- execute decode loop
- stream tokens to Kotlin
- stop generation safely
- expose runtime info
- run controlled benchmark suites
- record performance metrics
- record backend decisions and fallback reasons

Core objects:

### `RuntimeConfig`

Holds:

- model path
- backend type
- thread count
- context length
- batch size
- temperature
- top-p
- top-k
- repeat penalty
- `n_gpu_layers`
- benchmark flags

### `AppEngine`

Owns:

- runtime lifecycle
- active model state
- active generation session
- benchmark entry points
- diagnostics surface

### `GenerationSession`

Owns:

- prompt ingestion
- decode loop
- cancellation state
- token callback path
- per-request timing

### `BackendSelector`

Owns:

- requested backend validation
- OpenCL initialization
- CPU fallback rules
- backend capability reporting

### `BenchmarkRunner`

Owns:

- benchmark prompt definitions
- fixed runtime configs
- metric aggregation
- result serialization

### `MetricsLogger`

Records:

- TTFT
- prompt tok/s
- decode tok/s
- total generation time
- total generated tokens
- memory snapshots where available
- backend used
- model used

---

## 9. JNI Contract

JNI should expose a small surface and nothing more.

Suggested API:

```text
nativeInit(configJson)
nativeLoadModel(path, configJson)
nativeUnloadModel()
nativeGenerate(prompt, requestJson)
nativeStopGeneration()
nativeGetRuntimeInfo()
nativeRunBenchmark(benchmarkJson)
nativeListSupportedBackends()
```

JNI rules:

- no business logic in JNI
- no backend policy in JNI
- no state duplication between Kotlin and native
- token and metric callbacks must be explicit and timestampable

Config exchange format:

- JSON initially for speed of development
- compact structs later only if JNI overhead becomes measurable

---

## 10. Backend Strategy

### v1 backend order

1. `CPU`
2. `GPU_OPENCL`
3. `HTP` experimental only after GPU baseline exists

This order is intentional.

CPU is required first because:

- it validates model loading and generation correctness
- it isolates app and JNI overhead from GPU backend issues
- it gives us a trustworthy fallback

OpenCL is the real target because:

- it is the most practical path for device-specific optimization on Adreno
- it gives the best opportunity to tune decode performance directly
- it aligns with the project objective

HTP is deferred because:

- it is not required to prove the app architecture
- it can add major complexity before the benchmark system is mature
- it should be judged by measurement, not by assumption

---

## 11. Model Strategy

Model scope must stay narrow at the start.

Use:

- one **primary benchmark model** for most iteration
- one **secondary stress model** for memory and thermal exploration

Selection rules:

- dense model
- small enough to iterate quickly on device
- realistic chat decode behavior
- quantized format supported well by llama.cpp on the target path

Initial operating range:

- 1B to 1.5B for fast iteration
- 3B as a stress and scaling checkpoint

Model profile format:

```json
{
  "name": "Primary Benchmark Model",
  "path": "/sdcard/Models/primary-model.gguf",
  "quantization": "Q4",
  "recommended_backend": "GPU_OPENCL",
  "recommended_context": 4096,
  "recommended_threads": 4,
  "recommended_ngl": 99,
  "notes": "Primary Snapdragon benchmark target"
}
```

Important rule:

- do not spend early time building rich model management when one fixed benchmark model is enough to start optimization

---

## 12. Benchmarking Protocol

Benchmarking is a first-class subsystem.

Benchmark modes:

### Smoke

- one short prompt
- one short decode target
- fast validation after a code change

### Standard

- short prompt
- medium prompt
- long prompt
- fixed decode length

### Sustained

- repeated decode loop
- 3 to 5 minute duration
- thermal and stability observation

Each benchmark run must capture:

- timestamp
- git revision if available
- device name
- Android version
- model profile
- backend
- runtime config
- prompt length
- generated token count
- TTFT
- prompt tok/s
- decode tok/s
- total runtime
- failure status if any

Optional capture:

- battery before and after
- temperature note
- system trace reference

Rules:

- benchmark prompts must be fixed and versioned
- benchmark configs must be fixed and versioned
- one variable changes at a time during optimization work

---

## 13. Logging and Diagnostics

Logs must help explain performance outcomes.

Required log categories:

- app lifecycle
- model lifecycle
- backend initialization
- backend fallback
- generation lifecycle
- benchmark lifecycle
- native errors
- runtime config snapshot

Diagnostics screen should show:

- active device info
- active model info
- active backend
- backend initialization result
- last error
- recent benchmark summaries

Developer toggles:

- verbose native logs
- save benchmark result to file
- record raw timing events
- enable experimental backend flags

---

## 14. UI Scope

The UI exists to drive the runtime and expose measurements.

### Chat Screen

Must support:

- prompt entry
- send
- visible streaming output
- stop generation
- active backend badge
- active model badge
- current request timing summary

### Settings Screen

Must support:

- backend choice
- thread count
- context length
- batch size
- generation parameters
- `n_gpu_layers`

### Developer Screen

Must support:

- run smoke benchmark
- run standard benchmark
- run sustained benchmark
- view TTFT and tok/s
- inspect active config
- export benchmark results

Not required before performance work begins:

- polished navigation architecture
- conversation history
- consumer-grade chat UX

---

## 15. Concrete Milestones

### Milestone 0 — Repo Bootstrap

Status: `completed`

Deliverables:

- Android project created
- Compose app launches
- arm64-v8a build target configured
- NDK and CMake integration added

Exit criteria:

- app installs and launches on the target device
- native hello-world JNI call succeeds

### Milestone 1 — CPU Baseline

Status: `completed`

Deliverables:

- `llama.cpp` integrated
- one hardcoded GGUF model path supported
- single prompt generation works on CPU

Exit criteria:

- app can load model and generate a response on device
- native timings for load, prompt, and decode are emitted

### Milestone 2 — Streaming and Stop

Status: `completed`

Deliverables:

- token streaming callback
- UI updates incrementally
- stop and cancel path implemented

Exit criteria:

- tokens appear progressively in the chat screen
- stop button ends generation cleanly without app restart

### Milestone 3 — Benchmark Harness

Status: `completed`

Deliverables:

- smoke benchmark
- standard benchmark
- result persistence or structured file export
- developer screen shows latest runs

Exit criteria:

- one tap runs a fixed benchmark suite
- results are reproducible and comparable between builds

### Milestone 4 — OpenCL Bring-Up

Status: `completed (functional, underperforming)`

Deliverables:

- OpenCL backend selectable
- backend initialization diagnostics
- CPU fallback behavior

Exit criteria:

- OpenCL path runs at least one full prompt and decode benchmark on target device
- logs clearly show when fallback occurred and why

Outcome:

- OpenCL is compiled into the app and selectable in the UI (`GGML_OPENCL=ON`)
- fallback diagnostics are implemented and working
- device reports `GPUOpenCL(QUALCOMM Adreno(TM) 830)` and OpenCL model loading works
- GGML_VULKAN was investigated as an alternative but blocked by Adreno 830 driver crash during compute
- OpenCL decode is currently slower than CPU on this model due to operator fallback
- **Current best state**: CPU remains primary runtime (~28-30 tok/s standard runs), OpenCL remains experimental until throughput surpasses CPU
- immediate next work: reduce fallback-heavy graph paths and re-measure OpenCL with controlled `n_gpu_layers` sweeps

### Milestone 5 — OpenCL Optimization Passes

Status: `in progress`

Deliverables:

- first measured OpenCL baseline
- iterative performance patches
- optimization journal entries tied to benchmark deltas

Exit criteria:

- at least one measured improvement versus the original OpenCL baseline
- no regression in stability during sustained runs

### Milestone 6 — HTP Viability Check

Status: `not started`

Deliverables:

- experimental HTP integration branch or toggle
- measurement against CPU and OpenCL
- written conclusion on whether HTP is worth deeper work

Exit criteria:

- decision made using benchmark evidence, not intuition

---

## 16. Sprint Backlog

This is the execution order.

### Sprint 1

Status: `completed`

- create Android skeleton
- create Compose shell with Chat, Settings, Developer screens
- add JNI smoke path

Completed outcome:

- app boots on device
- JNI smoke path was replaced by the real runtime path in later sprints

### Sprint 2

Status: `completed`

- vendor or submodule `llama.cpp`
- build CPU-only native generation path
- hardcode one test model path

Completed outcome:

- CPU-only native generation works on device with a real GGUF model
- internal app storage is used for reliable native model access

### Sprint 3

Status: `completed`

- add streamed token callbacks
- add stop generation
- show request metrics in UI

Completed outcome:

- tokens stream live into the UI
- stop/cancel works during native decode
- TTFT and decode summary metrics are visible after generation

### Sprint 4

Status: `completed`

- add benchmark definitions
- add benchmark execution and export
- lock benchmark prompts and configs

Completed outcome:

- smoke and standard benchmarks run from the Developer screen
- benchmark results are persisted locally and recent history is visible in-app

### Sprint 5

Status: `completed`

- enable OpenCL backend selection
- validate actual OpenCL runtime behavior on device
- measure CPU versus OpenCL
- optimize CPU performance baseline

Outcome:

- backend selection is live in the app
- model reload correctly invalidates when switching backend targets
- runtime info and load status now report backend inventory and OpenCL failure reasons
- current device validation shows both `CPU` and `GPUOpenCL` are available
- OpenCL loads and runs on this build, but decode throughput is below CPU for current model/operator mix
- **CPU performance issues found and fixed (3 issues, ~60x total improvement)**:
  1. Missing `GGML_CPU_ARM_ARCH` flag — no ARM64 SIMD optimizations. Fix: `GGML_CPU_ARM_ARCH="armv8.5-a+fp16+i8mm+dotprod"`
  2. `GGML_LLAMAFILE=OFF` — disabled optimized SGEMM kernels. Fix: set to `ON`
  3. **`CMAKE_BUILD_TYPE=Debug` (-O0)** — the dominant issue. `assembleDebug` passes Debug to NDK CMake, compiling all native code with zero optimization. Fix: added `-DCMAKE_BUILD_TYPE=Release` to cmake arguments in `build.gradle.kts`. This single change gave **~28x improvement**.
  - Thread count auto-resolution also enabled (0 → native resolves to 6 on S8 Elite)
  - Final result: ~0.46 tok/s → **~28-30 tok/s** for Qwen3.5-2B Q4_K_M
- CPU baseline is now fully optimized and ready to serve as comparison target for GPU acceleration
- CPU default path now uses lazy accelerator loading so regular CPU runs avoid OpenCL startup overhead
- adb automation path upgraded: `DEBUG_OPENCL_SMOKE` now queues `WorkManager` work, so long benchmarks are no longer limited by `BroadcastReceiver` lifetime
- benchmark automation script now targets this path: `scripts/adb_backend_benchmark_worker.sh`

### Sprint 6 — GGML_VULKAN Experiment

Status: `completed (blocked)`

- Investigated GGML_VULKAN as alternative GPU acceleration path (bypasses OpenCL ICD + namespace issue)
- Build infrastructure resolved: host toolchain, Vulkan-Headers, glslc, libvulkan.so
- Vulkan device detected successfully on iQOO 13: `Vulkan0(Adreno (TM) 830, reg=Vulkan, type=igpu)`
- Model loaded and scheduled on Vulkan (489 MiB GPU memory reserved, 1377 graph nodes)
- **CRASH**: SIGSEGV in `vulkan.adreno.so`'s `vkCmdBindPipeline` during first decode
- Root cause: Adreno 830 driver incompatibility with ggml-vulkan pipeline creation on this custom ROM
- Advanced shader extensions (coopmat, dotprod, bfloat16) made no difference — same crash with/without
- Resolution: `GGML_VULKAN=ON` removed from CMakeLists.txt; app reverted to CPU-only
- All Vulkan code (detection, loading, UI) remains in codebase but inactive
- See Section 3B for full details

### Sprint 6+

Status: `pending`

- optimize decode path
- optimize buffer reuse and residency
- experiment with offload strategy
- explore HTP only if justified

### Sprint 7 — GPU Path Resolution

Status: `pending`

Three parallel paths to unblock useful GPU acceleration:

**Path A — OpenCL decode optimization on current ROM**:
- Keep CPU as default runtime while optimizing OpenCL as an experimental path
- Non-broadcast benchmark runner is now in place via `WorkManager` (`startForegroundService` from adb broadcast remains blocked on this ROM)
- `n_gpu_layers=4` is the current best OpenCL setting; use it as the optimization baseline
- Profile fallback-heavy decode ops and re-test after each optimization pass
- Expected: OpenCL micro/smoke becomes reproducible first, then throughput improvements can be measured credibly

**Path B — Stock ROM cross-check (optional but high value)**:
- Flash stock OriginOS 6 and run the same benchmark matrix
- Compare OpenCL load time, TTFT, and decode throughput against custom ROM
- Expected: isolate whether current OpenCL underperformance is mostly ggml graph/fallback behavior or ROM/driver overhead

**Path C — ggml-vulkan fix**:
- Monitor llama.cpp for Adreno 830 / Snapdragon 8 Elite fixes
- Retry on newer firmware/driver drops when available
- Try partial GPU offload (`n_gpu_layers`) once driver stability improves
- Expected: variable, depends on root cause

---

## 17. Hardware-Specific Optimization Workstreams

These workstreams are the real point of the project.

### Workstream A — OpenCL Decode Path

Questions:

- which ops dominate decode latency
- where dequantization costs are paid
- whether kernel launch overhead is hurting small-batch decode
- whether tensor layout or buffer movement is limiting throughput

### Workstream B — Offload Strategy

Questions:

- full versus partial GPU offload
- prompt path versus decode path configuration
- which layer split gives the best sustained decode

### Workstream C — Session Residency

Questions:

- how much model and KV state can remain resident
- where reinitialization overhead happens
- whether multi-turn reuse is leaving performance on the table

### Workstream D — App Overhead

Questions:

- how much time is lost in JNI callbacks
- whether token batching improves perceived streaming without harming UX
- whether UI update cadence is too expensive

### Workstream E — Thermal Stability

Questions:

- when throttling begins
- which configs produce stable throughput over time
- when a slower nominal config wins over sustained runs

### Workstream F — HTP Reality Check

Questions:

- whether HTP helps prompt, decode, or neither
- whether integration cost is justified by measured gains
- whether hybrid scheduling is worth the complexity

---

## 18. Testing Strategy

Unit tests:

- config parsing
- model profile parsing
- benchmark result formatting
- runtime preset validation if presets are added

Integration tests:

- model load and unload
- generation start and stop
- token stream callback integrity
- backend fallback behavior

Manual test matrix:

- cold app launch then first prompt
- repeated prompt loop
- long prompt
- sustained decode run
- model reload
- background then foreground during idle and during generation

---

## 19. Risks

### Risk: chasing backend work before the app is measurable

Mitigation:

- benchmark harness by Milestone 3
- fixed prompt suite
- fixed baseline config

### Risk: OpenCL path works but does not outperform CPU enough

Mitigation:

- treat bring-up and optimization as separate milestones
- inspect per-stage timings before changing kernels blindly

### Risk: JNI or UI streaming overhead contaminates results

Mitigation:

- timestamp native token events
- compare raw native throughput against visible UI throughput

### Risk: thermal throttling invalidates short benchmarks

Mitigation:

- sustained benchmark mode
- compare burst and sustained results separately

### Risk: project scope drifts into generic app development

Mitigation:

- keep user-facing scope narrow
- reject non-essential features until OpenCL optimization work is producing results

---

## 20. Immediate Build Order

Start here:

1. create Android app shell
2. add JNI hello path
3. integrate `llama.cpp`
4. run one model on CPU
5. stream tokens to UI
6. add benchmark harness
7. enable OpenCL selection
8. measure OpenCL
9. optimize OpenCL
10. evaluate HTP

This order is mandatory because it keeps correctness, visibility, and measurement ahead of optimization experiments.

---

## 21. Immediate Next Action

Build the **minimum viable performance app**:

- one chat screen
- one settings screen
- one developer screen
- one JNI bridge
- one hardcoded model path
- one CPU generation path
- one stop button
- one streaming output panel
- one benchmark button

Only after that should backend-specific optimization begin.

---

## 22. Final Principle

The app is not the end goal.

The end goal is a repeatable system for answering this question:

**What change on this Snapdragon hardware actually improves local LLM token generation?**

Every architecture decision, every UI decision, and every native code change should be judged by whether it helps answer that question faster and more accurately.
