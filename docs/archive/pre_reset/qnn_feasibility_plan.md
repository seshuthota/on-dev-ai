# QNN Feasibility Plan For OnDevAI

## Status

This file is the QNN feasibility and bring-up record.

It contains the detailed history of the QNN investigation and implementation work, but it is not the canonical active execution plan anymore.

For the current active plan, use:

- `docs/current_execution_plan_2026-04-10.md`

Use this file for:

- QNN workstream history
- validation results
- packaging/runtime findings
- Genie bring-up notes

## 1. Objective

Determine whether a Qualcomm QNN / QAIRT NPU path can deliver a meaningful throughput jump on the target device without destabilizing the current app.

This is a feasibility plan, not a migration plan.

The current CPU and OpenCL paths remain the production benchmark baseline until QNN proves all of the following:

- it can load and run a real model on the target phone
- it can be benchmarked using the same prompts and reporting conventions
- it can beat the current CPU baseline by a meaningful margin
- it can sustain repeated runs without breaking app startup, packaging, or backend fallback logic

## 2. Why This Exists

The current OpenCL path looks like a local optimum, not a path to a large jump:

- CPU is already roughly `28-30 tok/s`
- OpenCL is functional but materially slower on the current model
- higher `n_gpu_layers` makes results worse, not better
- fused Gated Delta Net fallback is a structural backend mismatch, not a simple tuning issue

That makes QNN the right next spike if the goal is a step-change rather than another 5-10 percent.

## 3. Source-Backed Constraints

These are the facts this plan is built around:

- Official `llama.cpp` currently lists `OpenCL` for `Adreno GPU`, but `Hexagon` is still marked `In Progress` for `Snapdragon`.
- Official `llama.cpp` build docs include Android + OpenCL instructions, but do not currently provide a stable Android + Hexagon/QNN build flow.
- Qualcomm AI Hub Models lists `Qualcomm AI Engine Direct` as supporting `Android`, `Linux`, and `Windows`.
- Qualcomm AI Hub Models lists Snapdragon `8 Elite` among supported chipsets and shows NPU support for `FP16`, `INT16`, and `INT8`.
- Qualcomm QIDK explicitly provides Android sample apps that use `Qualcomm AI Engine Direct SDK` / `QNN`.
- The current QIDK README references a public `Qualcomm AI Runtime Community` SDK download and also shows newer QAIRT-based example solutions.

Inference from these sources:

- A serious QNN path is plausible on this device class.
- It is not currently a drop-in flag inside our existing `llama.cpp` Android integration.
- We should expect a separate SDK, separate model export path, and separate runtime packaging work.

## 3A. Local SDK Status

The repo already contains a local QAIRT/QNN SDK drop:

- version: `2.45.0.260326`
- root: `/home/curious/Documents/hermes-projects/OnDevAI/v2.45.0.260326/qairt/2.45.0.260326`

Validated locally:

- QNN headers exist under `include/QNN`
- Genie headers exist under `include/Genie`
- Android-target shared libraries exist under `lib/aarch64-android`
- HTP-specific Android libraries are present, including:
  - `libQnnHtp.so`
  - `libQnnHtpPrepare.so`
  - `libQnnHtpV68Stub.so`
  - `libQnnHtpV73Stub.so`
  - `libQnnHtpV75Stub.so`
  - `libQnnHtpV79Stub.so`
  - `libQnnHtpV81Stub.so`
- Genie runtime libraries are present, including:
  - `libGenie.so`
  - `libQnnGenAiTransformer.so`
  - `libQnnGenAiTransformerModel.so`
- SDK docs are present under `docs/QAIRT-Docs`

Practical consequence:

- Workstream A no longer starts with "obtain SDK"
- Workstream A now starts with "validate and pin this SDK layout in the repo"
- Workstream C can proceed as soon as host/tool invocation and minimal runtime loading are verified

## 3B. Current Feasibility Status

Status as of `2026-04-09`:

- Workstream A setup artifacts are now in the repo:
  - `docs/qnn_sdk_setup.md`
  - `.env.example`
  - `scripts/check_qnn_sdk.sh`
- The repo-local validator confirms the pinned SDK tree, Android runtime libraries, and connected device metadata.
- The official QAIRT Android `qnn-net-run` sample path has already completed on the target phone for:
  - `cpu`
  - `htp-v79`
- The target phone is reachable over adb and identifies as:
  - model `I2401`
  - SoC `SM8750`
  - Android `16`

What this means:

- QNN runtime packaging on this ROM is viable.
- HTP loading is viable on this device class.
- The next real risk is no longer "can QNN run on-device at all".
- The next real risk is model compatibility and export for our current baseline model.

## 4. Non-Goals

This spike does not try to do these things up front:

- replace the current app runtime immediately
- switch the project baseline to a new model
- optimize OpenCL and QNN at the same time
- chase social-media performance claims without reproducing them locally
- promise `100+ tok/s` before we have even loaded one model through QNN on-device

## 5. Key Decision

The spike stays on the current model first if conversion/export is possible.

If the current GGUF model cannot be moved into a QNN-compatible format in a reasonable amount of time, use a temporary reference LLM only to validate the QNN path itself. If that happens:

- the reference model is for backend validation only
- the current model remains the project benchmark baseline
- no throughput claims are accepted unless the same benchmark harness is used

## 6. Entry And Exit Gates

### Entry Gate

Start this spike only because the current OpenCL path has already been characterized and is underperforming relative to CPU.

### Exit Gate: Continue

Continue investment only if all of the following are true:

- QNN runtime libraries load on the phone
- one real text-generation model runs end-to-end
- `micro` and `smoke` benchmark equivalents are measurable
- decode throughput beats CPU by at least `1.5x` on one stable configuration

### Exit Gate: Stop

Stop after the feasibility spike if any of the following are true:

- SDK/runtime packaging is too brittle on the target ROM
- model conversion/export path requires unacceptable model changes
- throughput does not materially beat CPU
- integration cost is high enough that it would stall the current app for weeks

## 7. Workstreams

### Workstream A: SDK And Runtime Acquisition

Goal:

- prove we can legally and reproducibly obtain the QAIRT/QNN toolchain needed for Android targeting

Actions:

- create branch `qnn-spike-sdk`
- document this pinned SDK in a new setup note:
  - version `2.45.0.260326`
  - local root path
  - required `lib/aarch64-android` libraries
  - required DSP/HTP companion files
- document host prerequisites in the same note:
  - Qualcomm account / SDK access
  - host OS
  - NDK version
  - Java / Gradle compatibility
- record exact package name and version in the repo
- add a validator that checks this exact SDK tree before any QNN build starts

Repo changes:

- add `docs/qnn_sdk_setup.md`
- add `.env.example` entries for `QAIRT_SDK_ROOT` or equivalent
- add `scripts/check_qnn_sdk.sh`

Success criteria:

- one command verifies that required host tools and target Android libraries exist
- exact SDK version is pinned in docs
- validator confirms `lib/aarch64-android/libQnnHtp.so` and related runtime files are present

Failure criteria:

- access is blocked
- Android-target runtime libraries required for HTP are missing
- local SDK tree is incomplete or not usable for Android packaging

### Workstream B: Model Compatibility Audit

Goal:

- determine whether the current model can realistically be brought into a QNN-compatible path

Actions:

- inspect the current baseline model architecture and quantization
- determine whether there is a supported export route to QNN/QAIRT
- if direct export is unrealistic, choose one temporary validation model that is close enough to prove the backend path without rewriting the whole app

Decision order:

1. Current model
2. Closely related Qwen/Llama class model with existing Qualcomm path
3. Abort if the only working path requires an unrelated demo model

Deliverable:

- compatibility table in the plan with:
  - model name
  - source format
  - export path
  - expected precision
  - blocker notes

Success criteria:

- one candidate model is selected with a concrete conversion/export path

### Workstream B: Current Audit Status

Current production model facts gathered from the on-device GGUF header:

- file: `files/models/primary-model.gguf`
- size: `1274396256` bytes
- `general.architecture=qwen35`
- `general.size_label=1.9B`
- `general.quantized_by=Unsloth`
- tokenizer pre: `qwen35`
- architecture-specific fields include:
  - `qwen35.ssm.conv_kernel`
  - `qwen35.ssm.state_size`
  - `qwen35.ssm.group_count`
  - `qwen35.ssm.time_step_rank`
  - `qwen35.ssm.inner_size`

Interpretation:

- this is not just a standard transformer checkpoint repacked as GGUF
- it includes state-space / SSM-specific structure
- it is the same model family that currently triggers fused Gated Delta Net fallback in the OpenCL path

Current compatibility assessment:

| Candidate | Source format today | Evidence for Qualcomm path | Risk | Decision |
| --- | --- | --- | --- | --- |
| Current `qwen35` 1.9B GGUF | GGUF only in app storage | No direct public Qualcomm AI Hub listing identified for `qwen35`; current path is `llama.cpp`-specific | High | Keep as stretch target, do not make it the first QNN integration target |
| `Qwen3-4B` | Official AI Hub model | Qualcomm AI Hub lists it for phone/tablet, Snapdragon 8 Elite, with a Genie SDK tutorial and minimum QNN SDK `2.42.0` | Medium | Best Qwen-family validation candidate if we want family proximity |
| `Llama-v3.2-3B-Instruct` | Official AI Hub model | Qualcomm AI Hub lists it for Snapdragon 8 Elite and links an Android `LLM Chat` sample app using Genie APIs | Low | Best control model for first Android app-side QNN integration |

Working decision:

1. Do not block the QNN spike on exporting the current `qwen35` GGUF immediately.
2. Use `Llama-v3.2-3B-Instruct` as the first control model for Android-side Genie/QNN app integration because Qualcomm already publishes an Android chat sample around this path.
3. Use `Qwen3-4B` as the next family-adjacent validation candidate if we want a Qwen-branded path on Snapdragon 8 Elite.
4. Return to the current `qwen35` baseline only after the app-side QNN path is proven and we have a credible export route.

Operational update:

- Worker benchmarks now support explicit model routing via `model_name` / `model_path`.
- Multi-model matrix runner added:
  - `scripts/adb_multi_model_benchmark_matrix.sh`
- Existing single-run worker script now accepts optional model argument:
  - `scripts/adb_backend_benchmark_worker.sh <serial> <backend> <mode> <gpu_layers> [timeout_sec] [model_name_or_path]`
- Matrix script now supports repeat-count execution and median rollups:
  - `scripts/adb_multi_model_benchmark_matrix.sh <serial> <backend> <mode> <gpu_layers> <models_csv> [timeout_sec] [report_file|repeat_count] [repeat_count]`
  - emits `model_repeat_*` lines and final `model_median*` lines per model

This allows us to compare control-model and baseline-model behavior with the same backend/mode harness instead of hardcoding `primary-model.gguf`.

### Workstream B: Multi-Model Micro Baseline (2026-04-09)

Device: `I2401` (`SM8750`) over adb `192.168.29.11:41993`, `ctx=2048`, `threads=6`.

Run artifacts:

- `/tmp/ondevai_matrix_cpu_micro_models_v3.txt`
- `/tmp/ondevai_matrix_opencl_micro_models_v3.txt`

Models installed in app internal storage:

- `primary-model.gguf` (qwen35 2B class)
- `qwen35-2b-q4_k_m.gguf` (qwen35 2B class control copy)
- `qwen35-4b-q4_k_m.gguf` (qwen35 4B class control)

CPU micro (`n_gpu_layers=-1`):

- `primary-model.gguf`: `23.39 tok/s`, `TTFT 94 ms`, `load 1866 ms`
- `qwen35-2b-q4_k_m.gguf`: `24.84 tok/s`, `TTFT 98 ms`, `load 2225 ms`
- `qwen35-4b-q4_k_m.gguf`: `6.76 tok/s`, `TTFT 337 ms`, `load 3453 ms`

OpenCL micro (`n_gpu_layers=4`):

- `primary-model.gguf`: `8.75 tok/s`, `TTFT 227 ms`, `load 2631 ms`, fallback signal count `2`
- `qwen35-2b-q4_k_m.gguf`: `8.64 tok/s`, `TTFT 233 ms`, `load 2545 ms`, fallback signal count `2`
- `qwen35-4b-q4_k_m.gguf`: `2.82 tok/s`, `TTFT 1673 ms`, `load 5349 ms`, fallback signal count `2`

Interpretation:

- 2B-class models stay in the previously observed range.
- 4B-class model scales down sharply on both CPU and OpenCL.
- OpenCL remains slower than CPU for the tested qwen35 models and still shows fused Gated Delta Net fallback signals.
- This reinforces the decision to keep QNN/Genie as the step-change path, not incremental OpenCL tuning for this model family.

### Workstream B: Multi-Model Smoke Baseline (2026-04-09)

Run artifacts:

- `/tmp/ondevai_matrix_cpu_smoke_models_v1.txt`
- `/tmp/ondevai_matrix_opencl_smoke_models_v1.txt`

CPU smoke (`n_gpu_layers=-1`, 32 decode):

- `primary-model.gguf` (qwen35 2B class): `27.44 tok/s`, `TTFT 143 ms`, `load 2345 ms`
- `qwen35-4b-q4_k_m.gguf` (qwen35 4B class): `13.79 tok/s`, `TTFT 334 ms`, `load 3599 ms`

OpenCL smoke (`n_gpu_layers=4`, 32 decode):

- `primary-model.gguf` (qwen35 2B class): `9.54 tok/s`, `TTFT 333 ms`, `load 2423 ms`, fallback signal count `2`
- `qwen35-4b-q4_k_m.gguf` (qwen35 4B class): `5.50 tok/s`, `TTFT 732 ms`, `load 4120 ms`, fallback signal count `2`

Interpretation:

- 4B class roughly halves decode speed relative to 2B class on CPU.
- OpenCL remains behind CPU for both tested sizes and keeps the same fallback signature.
- Model scaling behavior is now measured and reproducible with the same worker harness, which is enough to proceed to QNN control-model integration without waiting for more OpenCL parameter sweeps.

### Workstream C: Minimal Native QNN Harness

Goal:

- prove QNN on-device independently of the app UI

Actions:

- create a tiny standalone native harness outside the chat app path
- keep this harness separate from `llama.cpp` integration at first
- compile and run a minimal model load + single inference + timing log

Repo changes:

- add `native/qnn/`
- add `native/qnn/CMakeLists.txt`
- add `native/qnn/src/qnn_smoke.cpp`
- add `scripts/build_qnn_smoke.sh`
- add `scripts/run_qnn_smoke_android.sh`

Implementation rules:

- no UI changes yet
- no ViewModel changes yet
- no backend selector changes yet
- all success/failure should be visible through adb/logcat and exit codes

Success criteria:

- harness runs on device
- QNN runtime initializes
- one inference completes and logs latency

Failure criteria:

- packaging or runtime loading fails
- HTP delegate/backend cannot initialize on the device

### Workstream C: Implementation Status (2026-04-09)

Added artifacts:

- `native/qnn/CMakeLists.txt`
- `native/qnn/src/qnn_smoke.cpp`
- `scripts/build_qnn_smoke.sh`
- `scripts/run_qnn_smoke_android.sh`

Harness behavior:

- `qnn_smoke` is a standalone Android CLI binary.
- It probes required runtime libs with `dlopen` before inference.
- It then runs one `qnn-net-run` invocation and reports `qnn_smoke_result ... inference_ms=<N>`.
- Runner script exits non-zero on backend/device failures and always prints log/output paths.

On-device results on `192.168.29.11:41993`:

- `cpu`: pass, `qnn_smoke_result status=ok exit_code=0 inference_ms=192`
- `htp-v79`: pass, `qnn_smoke_result status=ok exit_code=0 inference_ms=583`
- `htp-v81`: fail, `Device Creation failure`, `qnn_smoke_result status=fail exit_code=11 inference_ms=25`

Interpretation:

- Workstream C success criteria are met for `cpu` and `htp-v79`.
- `htp-v79` remains the correct first HTP target for this `SM8750` device.
- `htp-v81` should be treated as an optional fallback probe, not default.

### Workstream D: Android Packaging And Loader Integration

Goal:

- make QNN libraries load reproducibly inside the Android app process

Actions:

- mirror the QIDK-style Android packaging approach
- identify exactly which shared libraries and skeleton files are required
- add imported CMake targets and Gradle packaging rules
- keep all QNN additions behind a build option until validated

Likely files to change:

- `native/CMakeLists.txt`
- `app/android/app/build.gradle.kts`
- `app/android/app/src/main/AndroidManifest.xml`
- possibly new `native/cmake/FindQNN.cmake`

Recommended feature flag:

- `ONDEVAI_ENABLE_QNN`

Success criteria:

- app installs with QNN bits included
- no regression to CPU/OpenCL startup when QNN is disabled
- QNN can be enabled at build time without affecting current userspace path

### Workstream D: Implementation Status (2026-04-09)

Implemented build-time switch:

- Gradle property: `ondevai.enableQnn` (default `false`)
- Optional SDK override: `ondevai.qairtSdkRoot=/abs/path/to/qairt/...`
- Native CMake flag wired: `-DONDEVAI_ENABLE_QNN=ON|OFF`

Implemented packaging behavior when enabled:

- Stages selected runtime libs into APK `lib/arm64-v8a/`:
  - `libQnnSystem.so`, `libQnnCpu.so`, `libQnnGpu.so`, `libQnnHtp.so`, `libQnnHtpPrepare.so`
  - `libQnnHtpV79Stub.so`, `libQnnHtpV81Stub.so`
  - `libGenie.so`, `libQnnGenAiTransformer.so`
- Stages HTP skels into APK assets:
  - `assets/qnn/hexagon-v79/libQnnHtpV79Skel.so`
  - `assets/qnn/hexagon-v81/libQnnHtpV81Skel.so`
- Staging is validated before build; missing SDK artifacts fail the build with explicit paths.

Validation results:

- `./gradlew :app:assembleDebug` (QNN disabled): pass
- `./gradlew :app:assembleDebug -Pondevai.enableQnn=true`: pass
- `./gradlew :app:installDebug`: pass
- `./gradlew :app:installDebug -Pondevai.enableQnn=true`: pass
- APK inspection confirms QNN libs/assets appear only when enabled.

Notes:

- This is packaging/load plumbing only. App runtime does not select or execute QNN yet.
- Workstream E remains the step where backend selection and engine integration are added.

### Workstream E: App Engine Integration

Goal:

- expose QNN as a controlled experimental backend in the existing engine

Actions:

- add backend target normalization for `qnn`
- add QNN runtime inventory reporting
- implement a dedicated load path instead of overloading the OpenCL path
- keep the current CPU/OpenCL code untouched as much as possible

Likely files to change:

- `native/engine/include/app_engine.h`
- `native/engine/src/app_engine.cpp`
- `native/jni/jni_bridge.cpp`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/SnapdragonLabApp.kt`

Rules:

- QNN remains hidden behind an experimental toggle until benchmarked
- fallback to CPU must be explicit and logged
- do not delete or rewrite the OpenCL path during this spike

Success criteria:

- app can select `qnn`
- model load status and runtime info show `qnn` distinctly
- failures are diagnosable from logcat

### Workstream E: Implementation Status (2026-04-09)

Implemented behavior:

- Kotlin/UI backend normalization now accepts `qnn` only when the app is built with `ONDEVAI_ENABLE_QNN`.
- Settings screen exposes an experimental `Use QNN` target in QNN-enabled builds.
- Native runtime inventory now reports QNN separately from ggml backends:
  - `qnn_compiled=true|false`
  - `qnn_packaged_libs=...`
  - `qnn_packaged_count=...`
  - `qnn_status=...`
- `LoadModel(..., backend_target="qnn")` no longer silently normalizes to CPU.
- Current QNN load path fails fast with an explicit message that GGUF app-side QNN execution is not implemented yet.

Validated behavior:

- `:app:assembleDebug` passes with QNN disabled and enabled.
- `:app:installDebug -Pondevai.enableQnn=true` passes on device.
- Debug worker with `backend=qnn` logs an explicit Workstream E failure path:
  - `backend_target=qnn`
  - `backend_loaded=QNN (not loaded)`
  - `last_error=QNN backend selected but GGUF app-side QNN execution is not implemented yet`
- Failure is visible in app runtime info and logcat, with no silent fallback to CPU.

Current limitation:

- App-side QNN model execution still does not exist.
- App-side runtime packaging is now fully visible from the app process after two fixes:
  - manifest declares `libcdsprpc.so` as an optional native library
  - `NativeBridge` explicitly preloads packaged QNN libs with `System.loadLibrary(...)`
- After those fixes, app runtime inventory reaches `qnn_packaged_count=7/7` and `qnn_status=runtime_packaged` on device.
- The remaining gap is no longer runtime packaging; it is the absence of a real in-app QNN model execution path.
- A control-model subprocess experiment was also completed:
  - app packages `qnn_smoke`, `qnn-net-run`, sample model libs, and sample inputs as assets
  - app can extract those assets into internal storage
  - Android app sandbox blocks executing the extracted ELF binaries with `error=13 Permission denied`
- Conclusion: app-side subprocess execution from internal storage is not the right path; the next viable path is in-process native/JNI QNN execution.

### Workstream F: In-Process JNI Control Runner Status (2026-04-09)

Implemented behavior:

- Added a native in-process QNN runner linked into `libondevai_native.so`.
- Added a JNI bridge so the app can run a control-model smoke test without `ProcessBuilder`.
- `QnnControlRunner` now extracts:
  - control model `.so` assets into app files storage
  - `hexagon-v79` skel assets into app files storage for HTP runs
- `NativeBridge` now configures `ADSP_LIBRARY_PATH` before QNN preload.

Validated on `192.168.29.11:41993`:

- `cpu`: pass, `success=true`, `total_ms=5`
- `htp-v79`: pass, `success=true`, `total_ms=196`, `graph_compose_ms=75`, `graph_finalize_ms=19`

Root cause of the earlier app-side HTP `14001` failure:

- The app initially used `:` as the `ADSP_LIBRARY_PATH` separator.
- QAIRT FastRPC on Android expects `;` in this path, matching the official `android-qnn-net-run.sh` sample.
- With `:`, FastRPC treated the whole search path as one literal filename and could not locate `libQnnHtpV79Skel.so`.
- After switching `ADSP_LIBRARY_PATH` to `;` separators and extracting the `v79` skel before preload, `deviceCreate` and `contextCreate` succeeded in-process.

Interpretation:

- Android 16 app exec restrictions were only blocking the subprocess path.
- In-process QNN HTP execution is viable inside the app for this device.
- `htp-v79` is now validated in three ways on this device:
  - official QAIRT Android sample
  - standalone repo `qnn_smoke`
  - in-process app/JNI control-model runner

Next recommended step:

- integrate the in-process QNN path into the app engine as a real experimental execution path, starting with control-model style loading semantics rather than GGUF.

### Workstream G: Experimental App Engine QNN Path Status (2026-04-09)

Implemented behavior:

- The app now prepares QNN control assets and HTP environment during startup in QNN-enabled builds.
- `LoadModel(..., backend_target="qnn")` no longer fails as an unimplemented placeholder.
- The native engine now treats QNN as a real experimental control-model load path:
  - resolves `filesDir/qnn/control/libqnn_model_8bit_quantized.so`
  - validates it in-process through `qnn_runner`
  - records `backend_loaded=QNN (HTP v79 control)`
- `GenerateStreaming(...)` remains explicitly unsupported for this path and returns a precise message instead of pretending chat is available.
- `RunBenchmark("smoke")` now returns a structured JSON result for a control-model QNN run when the QNN control path is loaded.

Validated on `192.168.29.11:41993`:

- debug action `ai.ondev.snapdragonlab.DEBUG_QNN_ENGINE_SMOKE`: pass
- engine load:
  - `QNN control model loaded in 250 ms`
  - `backend_loaded=QNN (HTP v79 control)`
- engine runtime info:
  - `model_loaded=true`
  - `backend_target=qnn`
  - `qnn_status=runtime_packaged`
- engine benchmark:
  - `elapsed_ms=474`
  - `success=true`

Interpretation:

- `qnn` is no longer just a UI/runtime-info placeholder.
- The app now has a real engine-visible experimental QNN execution path.
- The remaining gap is specifically text-generation semantics and real LLM model export/integration, not app-process HTP execution.

### Workstream F: Benchmark Parity

Goal:

- compare QNN to CPU/OpenCL using the same harness logic

Actions:

- extend worker path and script path to understand backend `qnn`
- preserve the same benchmark modes:
  - `micro`
  - `smoke`
  - `standard`
- preserve the same output fields:
  - load time
  - TTFT
  - tok/s
  - success/failure

Likely files to change:

- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugReceiver.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugBenchmarkWorker.kt`
- `scripts/adb_backend_benchmark_worker.sh`

Success criteria:

- one command can benchmark `cpu`, `opencl`, and `qnn`
- results are comparable without changing prompts or parsing logic

### Workstream G: Kill Test

Goal:

- decide quickly whether QNN is worth full integration

Required report after the first complete pass:

- model used
- export path used
- QNN load time
- QNN micro/smoke tok/s
- CPU micro/smoke tok/s on the same build
- implementation pain points
- ROM/runtime issues

Decision rule:

- if QNN is not clearly faster than CPU on short decode, stop
- if QNN is faster but unstable, keep only if instability looks solvable within one more week

## 8. Execution Order

1. Lock current baseline results in the current branch.
2. Create `qnn-spike-sdk`.
3. Add SDK setup docs and validation scripts.
4. Complete model compatibility audit.
5. Build standalone native QNN smoke harness.
6. Run QNN smoke harness on device.
7. If successful, add build-flagged Android packaging.
8. If packaging works, integrate `qnn` as experimental backend in app engine.
9. Extend worker benchmark path.
10. Run parity matrix against CPU and OpenCL.
11. Make continue/stop decision.

## 9. Practical Success Metrics

Minimum acceptable result:

- QNN runs a real model end-to-end on the phone
- QNN beats CPU on `micro`

Meaningful result:

- QNN beats CPU on `micro` and `smoke`
- QNN remains stable across repeated runs

Full win:

- QNN clearly outperforms CPU and OpenCL on the same benchmark modes
- app integration remains maintainable

### Workstream H: First Real LLM Through Genie (2026-04-10)

What was completed:

- Host-side QAIRT Python toolchain is now usable in a project-local Python 3.10 env:
  - `.conda-qairt310`
  - required deps installed (`PyYAML`, `aenum`, `pydantic`, `sentencepiece`, `transformers`, etc.)
  - `qnn-genai-transformer-composer --help` works
- Real LLM composer path validated:
  - model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - output: `.artifacts/qnn-genai/tinyllama_1p1b_q4.bin` (~656 MB)
- Added Android runner script:
  - `scripts/run_genie_cpu_real_llm_android.sh`
  - pushes `genie-t2t-run`, runtime libs, model, tokenizer, generated dialog config, and runs on-device.

On-device result (192.168.29.11:41993):

- TinyLlama Q4 bin runs end-to-end and returns generated text successfully via `genie-t2t-run`.
- This proves first real LLM execution with QAIRT Genie artifacts on-device from this repo.

Current limitation:

- Runtime logs show `QNN_CPU` execution (not `QNN_HTP`) for the current `QnnGenAiTransformer` flow.
- The repo now separates this from the HTP-targeted path:
  - `scripts/run_genie_cpu_real_llm_android.sh`
  - `scripts/run_genie_htp_real_llm_android.sh`
  - `scripts/run_genie_t2t_android.sh` remains only as a compatibility wrapper
- So LLM execution is integrated and working, but NPU/HTP acceleration for this LLM path is not yet proven.

## 10. Risks

- The current GGUF model may not map cleanly into a QNN/QAIRT workflow.
- SDK setup may be more complex than backend performance tuning.
- The ROM may introduce runtime-loader or DSP-access quirks.
- A temporary validation model may prove the backend but not prove the final target model.
- A high-performance demo from another implementation may depend on a different model, precision, scheduler, or proprietary prep pipeline.

## 11. Immediate Next Actions

- create the new branch for the spike
- document the pinned local SDK and add a validator script
- add host-side validation script
- audit the current model for export feasibility
- do not modify the current app runtime until the native QNN smoke harness works

## 12. Source Notes

These links informed the plan:

- Official `llama.cpp` supported backends and current backend status:
  - https://github.com/ggml-org/llama.cpp
- Official `llama.cpp` build docs for Android/OpenCL:
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- Qualcomm AI Hub Models runtime/device support:
  - https://github.com/qualcomm/ai-hub-models
- Qualcomm AI Hub Apps Android `ChatApp` / `Genie SDK` deployment path:
  - https://github.com/qualcomm/ai-hub-apps
- Qualcomm QIDK Android/QNN examples and SDK acquisition notes:
  - https://github.com/quic/qidk
