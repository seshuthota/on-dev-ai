# OnDevAI Progress Deep Dive (2026-04-10)

## Purpose

This document records the actual project progress from the original performance goal to the current QNN / Genie state, grounded in the repo's plans, scripts, docs, and implementation files.

This is not a marketing summary. It is an engineering status document.

## 1. Original Goal

The project started as a Snapdragon 8 Elite-specific Android inference app whose real product goal is performance engineering, not generic portability.

Source of truth:

- `plan.md`

Core intent from the start:

- optimize for one hardware family first
- maximize local LLM inference performance on Snapdragon 8 Elite
- treat the app as a performance lab with a usable chat UI
- measure every optimization against TTFT, decode tok/s, sustained behavior, and thermal stability

Initial backend posture in `plan.md`:

- primary acceleration path: Adreno GPU via OpenCL
- secondary path: CPU
- experimental path: Hexagon / HTP only if measurement justified it

## 2. Baseline Before QNN

The repo already had substantial app and llama.cpp integration work completed before the QNN feasibility spike started.

What was already working:

- Android app shell builds and installs
- Compose UI for Chat, Settings, and Developer screens
- JNI bridge for init, load, generate, stop, and benchmark actions
- `llama.cpp` integrated into the native build
- real GGUF model loading from app internal storage
- live token streaming in the UI
- structured benchmark history logging

Key baseline lesson:

- the biggest CPU performance problem was not ARM flags, but build mode
- `./gradlew :app:assembleDebug` was propagating `CMAKE_BUILD_TYPE=Debug`
- that compiled native code with `-O0` and caused an enormous performance collapse

Documented in:

- `plan.md`
- `cpu_performance_fix_plan.md`

CPU optimization progression captured in the repo:

- baseline with poor native optimization: about `0.46 tok/s`
- after ARM SIMD flags: about `1.0 tok/s`
- after additional flags but still under `-O0`: about `1.3 tok/s`
- after forcing native Release / `-O3`: about `28-30 tok/s`

Current CPU baseline recorded in `plan.md` for Qwen3.5-2B Q4_K_M:

- cold load roughly `2.3-2.4s`
- micro benchmark about `22.53 tok/s`
- smoke benchmark about `27.14 tok/s`
- sustained decode about `30-32 tok/s`

## 3. OpenCL Path And Why It Stopped Being The Main Bet

The original optimization path focused on Adreno OpenCL. That path is still important context because it is the reason the repo pivoted toward QNN.

What was learned:

- OpenCL was not fundamentally blocked by Android 16
- on this phone and build, OpenCL can initialize and detect `GPUOpenCL(QUALCOMM Adreno(TM) 830)`
- the older namespace failure was environment-specific, not a universal Android 16 OpenCL ban

What went wrong anyway:

- OpenCL was consistently slower than CPU on the current model
- unsupported fused Gated Delta Net ops were falling back to CPU
- that created expensive CPU/GPU scheduling and copy overhead
- higher `n_gpu_layers` made performance worse instead of better

Measured OpenCL pattern in `plan.md`:

- best observed point was around `n_gpu_layers=4`
- pushing more layers to GPU degraded both TTFT and decode tok/s
- higher offload settings also became unstable in worker-driven runs

Additional OpenCL issues:

- `GGML_OPENCL_USE_ADRENO_KERNELS=ON` caused initialization hangs and had to be reverted
- worker-based benchmark infrastructure had to be added to avoid receiver lifetime problems and support long-running GPU tests

Supporting repo artifacts:

- `scripts/adb_backend_benchmark_worker.sh`
- `scripts/adb_multi_model_benchmark_matrix.sh`
- `scripts/adb_install_model_internal.sh`
- `scripts/adb_list_internal_models.sh`

Conclusion from this phase:

- OpenCL was functioning, but looked like a local optimum rather than a path to a step-change
- the current model/backend mix had a structural mismatch
- that is the direct reason `qnn_feasibility_plan.md` exists

## 4. Vulkan Experiment And Why It Was Abandoned

Before QNN, the repo also tried GGML Vulkan as another GPU path.

What was implemented:

- Vulkan build plumbing in native CMake
- host shader build toolchain support
- Vulkan backend selection in the engine and Kotlin UI
- required Vulkan headers vendored into `native/third_party/Vulkan-Headers`

What worked:

- build succeeded
- Vulkan backend enumerated on-device
- model loading reached Vulkan device selection and GPU memory allocation

What failed:

- first token generation crashed inside Qualcomm's vendor Vulkan driver
- fault occurred in `vulkan.adreno.so` during `vkCmdBindPipeline`
- advanced shader extension toggles did not fix it

Documented in:

- `plan.md`

Practical conclusion:

- this was treated as an Adreno 830 driver / ROM compatibility issue, not an app bug
- Vulkan was disabled and the project returned to CPU + OpenCL while evaluating QNN

## 5. Why The Repo Pivoted To QNN / QAIRT

The QNN pivot was not random. It followed directly from the evidence above.

Reasons captured in `qnn_feasibility_plan.md`:

- CPU was already decent
- OpenCL was slower than CPU for the current model
- Vulkan had driver-level instability
- Snapdragon 8 Elite is a device class where Qualcomm's QAIRT / QNN path is plausible
- official Qualcomm runtime and model ecosystem exists for Android + HTP

The repo explicitly reframed the question from:

- "Can OpenCL be tuned a bit more?"

to:

- "Can Qualcomm QNN / QAIRT deliver a meaningful throughput jump without destabilizing the app?"

## 6. QNN Feasibility Workstream: What Path We Took

The QNN work happened in clear stages, and the repo now contains the artifacts for each stage.

### 6.1. Workstream A: Pin And Validate The SDK

The repo already contained a local QAIRT SDK drop:

- version `2.45.0.260326`
- path `v2.45.0.260326/qairt/2.45.0.260326`

What was added:

- `docs/qnn_sdk_setup.md`
- `scripts/check_qnn_sdk.sh`
- `.env.example` entries for QAIRT-related setup

What this achieved:

- pinned the SDK version and layout
- documented required libraries and tooling
- validated that the target phone is reachable and appropriate for the spike

### 6.2. Workstream B: Model Compatibility Audit

The repo audited the existing production model and found that it was a poor first QNN target.

Key finding from `qnn_feasibility_plan.md`:

- the current baseline GGUF is `qwen35`-family and includes SSM/state-space structure
- it is not just a standard transformer checkpoint repacked as GGUF
- it already caused fused Gated Delta Net fallback pain on OpenCL

Decision recorded in the repo:

- do not block the QNN spike on direct export of the current GGUF
- use a control model first to prove the backend
- prefer Qualcomm-supported Llama / Qwen candidates before returning to the current production model

### 6.3. Workstream C: Minimal Native QNN Harness

The repo then proved QNN outside the app UI first.

Added:

- `native/qnn/src/qnn_smoke.cpp`
- `native/qnn/CMakeLists.txt`
- `scripts/build_qnn_smoke.sh`
- `scripts/run_qnn_smoke_android.sh`

What was validated on the device:

- CPU QNN sample path passed
- HTP `v79` sample path passed
- HTP `v81` failed

Interpretation:

- QNN runtime loading on the phone was viable
- HTP on this phone/ROM worked through `v79`
- `v81` was not the right target for this device state

### 6.4. Workstream D: Android App Packaging For QNN

After standalone validation, the repo wired QNN packaging into the Android app behind a build flag.

Implemented in:

- `app/android/app/build.gradle.kts`
- `native/CMakeLists.txt`

What changed:

- QNN libs and HTP skels are packaged only when `-Pondevai.enableQnn=true`
- default CPU/OpenCL path remains unchanged when QNN is off

### 6.5. Workstream E: Experimental App Backend Exposure

The app was then taught to recognize `qnn` as an experimental backend target.

Key files:

- `native/engine/src/app_engine.cpp`
- `native/engine/include/app_engine.h`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/ui/SnapdragonLabApp.kt`

Important behavior choice:

- `qnn` no longer silently normalized to CPU
- it reported itself explicitly
- it failed fast with a clear not-implemented message rather than pretending to work

This was an important engineering choice because it prevented false positives.

### 6.6. App-Side Packaging And Namespace Issues

Once QNN was packaged in-app, runtime library issues showed up.

Problems found:

- packaged HTP libraries did not fully load in the app namespace
- app-side HTP loading needed `libcdsprpc.so`

Fixes made:

- explicit QNN `System.loadLibrary(...)` preloads
- `<uses-native-library android:name="libcdsprpc.so" android:required="false" />`

Relevant files:

- `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- `app/android/app/src/main/AndroidManifest.xml`

This got app-side runtime state to `qnn_status=runtime_packaged`.

### 6.7. Subprocess Path Failed Under App Sandbox

The repo tried using packaged executables like `qnn_smoke` from app-private storage.

That failed with:

- `error=13 Permission denied`

Meaning:

- the problem was not QNN itself
- the problem was Android app execution policy for extracted binaries

This was a major fork point. It forced the repo away from `ProcessBuilder` and toward in-process JNI execution.

### 6.8. In-Process JNI QNN Runner

This was the correct pivot.

Implemented in:

- `native/qnn/src/qnn_runner.cpp`
- `native/qnn/include/qnn_runner.h`
- `native/jni/jni_bridge.cpp`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/QnnControlRunner.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`

What this changed:

- QNN APIs are loaded in-process
- control model loading and execution happen through JNI
- the app no longer depends on spawning executables from internal storage

### 6.9. The `14001` HTP Failure And The Real Root Cause

After the in-process runner was added, the app hit HTP error `14001`.

Initial suspicion:

- namespace isolation
- DSP unavailability
- wrong stub version

Actual root cause found in the repo work:

- `ADSP_LIBRARY_PATH` formatting was wrong
- QAIRT FastRPC expected `;` separators
- the app path was using `:`

This was one of the most important debugging wins in the project.

Files involved:

- `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/QnnControlRunner.kt`

After fixing the separator and extracting the HTP skel into app-private storage:

- app-side `htp-v79` worked in-process
- `deviceCreate` and `contextCreate` succeeded

This proved the issue was not "Android 16 blocks QNN" and not "NPU unavailable". It was app-side runtime setup.

### 6.10. Experimental Engine Integration

Once the in-process control runner was stable, the repo integrated it into the actual app engine.

Implemented in:

- `native/engine/src/app_engine.cpp`
- `native/engine/include/app_engine.h`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainActivity.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugBenchmarkWorker.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugReceiver.kt`

What this gave the project:

- `LoadModel(..., backend_target="qnn")` became a real control-path validation
- engine state reported `QNN (HTP v79 control)`
- debug smoke actions could validate the engine-backed control path on-device

Important limitation:

- this was still not text generation for a real LLM
- it was a control-model path proving in-process QNN execution inside the app

## 7. Host Toolchain Bring-Up For Real LLM Export

After the control model path worked, the repo moved to a real LLM export / execution workflow through Qualcomm Genie.

### 7.1. Why Python 3.10 Became Necessary

The pinned QAIRT GenAI toolchain is ABI-sensitive.

Observed issues:

- existing `llm` conda env was Python `3.12`
- QAIRT GenAI pieces required `libpython3.10.so.1.0`

What was done:

- created project-local env `.conda-qairt310`
- resolved missing runtime dependencies and Python packages incrementally

Key dependency issues fixed during bring-up:

- missing `PyYAML`
- missing `aenum`
- missing `typing_extensions`
- missing `pydantic`
- missing `sentencepiece`
- missing `transformers`
- missing `tokenizers`
- missing `safetensors`
- missing `huggingface_hub`
- missing `paramiko`
- missing `tqdm`
- host runtime linker issues for `libpython3.10.so.1.0`, `libc++.so.1`, `libunwind.so.1`, and `libGenie.so`

One local compatibility workaround that mattered:

- the environment only had `libunwind.so.8`
- QAIRT wanted `libunwind.so.1`
- a local env symlink was used to make the extension load

This work is reflected in:

- `.conda-qairt310`
- `docs/qnn_sdk_setup.md`
- the fact that `qnn-genai-transformer-composer` now works in this repo

## 8. Real LLM Export Attempts And What Failed

### 8.1. Qwen3.5 Local Model Attempt

The repo tried to stay close to the current Qwen-family work by using local Qwen checkpoints first.

Local candidates found:

- `~/models/Qwen3.5-4B-F16`
- `~/models/Qwen3.5-4B-FP16-full`

What blocked them:

1. The model config is multimodal:
   - architecture `Qwen3_5ForConditionalGeneration`
   - contains `text_config` and `vision_config`
   - this was not auto-mapped by the bundled QAIRT composer config logic

2. The local shard naming did not match the composer's expected pattern:
   - repo had `model.safetensors-00001-of-00002.safetensors`
   - composer looked for `model-00001-of-*.safetensors`

3. Even after creating a symlinked workaround directory, the model layout still failed composer assumptions:
   - key mismatch such as `KeyError: 'model.language_model.embed_tokens.weight'`

Conclusion:

- the current Qwen3.5 local model is not a practical first Genie control target in this repo state
- it remains a follow-up conversion problem, not the first path to real LLM-on-device proof

### 8.2. GPT-2 Control Attempt

The repo then used a fully cached local `gpt2` Hugging Face snapshot as a quick control model.

What succeeded:

- composer produced:
  - `.artifacts/qnn-genai/gpt2_124m_q4.bin`
  - `.artifacts/qnn-genai/gpt2_auto_q4.bin`

What failed:

- on-device Genie load reached graph finalization and then failed
- `finalizeGraphs FAILED!`

Meaning:

- composer path itself worked
- runtime compatibility for that model / config was not good enough
- it was not a dead end for Genie overall, but it was not a clean validation model

## 9. Real LLM Export Attempt That Worked: TinyLlama

The first complete LLM success in the repo came from TinyLlama.

Model used:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

What was done:

- direct weight download with `curl` was used after Hugging Face helper-based downloading proved unreliable for large files in this environment
- composer successfully produced:
  - `.artifacts/qnn-genai/tinyllama_1p1b_q4.bin`

Supporting runner added:

- `scripts/run_genie_cpu_real_llm_android.sh`

What the runner does:

- pushes `genie-t2t-run`
- pushes required QAIRT runtime libraries
- pushes model bin and tokenizer
- generates a dialog config
- runs generation on the phone via adb

This is a significant milestone because it moved the repo from:

- "QNN control graph works"

to:

- "a real LLM generated text on the phone through QAIRT Genie artifacts"

## 10. What Is Working Right Now

As of `2026-04-10`, the repo can do all of the following:

- run the Android app with stable CPU inference
- benchmark CPU and OpenCL paths through worker-based scripts
- validate QNN CPU and HTP `v79` outside the app with standalone smoke tooling
- package QNN runtime into the Android app behind a build flag
- run in-process QNN control models inside the app
- expose an experimental app-engine-visible `qnn` path for control-model validation
- compose a real LLM into Genie / QNN bin format on the host
- run TinyLlama text generation on the phone through `genie-t2t-run`

Concrete generated artifacts currently present:

- `.artifacts/qnn-genai/gpt2_124m_q4.bin`
- `.artifacts/qnn-genai/gpt2_auto_q4.bin`
- `.artifacts/qnn-genai/tinyllama_1p1b_q4.bin`

## 11. Supplementary Plan Status: What Was Actually Implemented

`suplementary-plan.md` influenced direction, but it was not followed literally. The actual repo path diverged based on measurements, device behavior, and Qualcomm toolchain constraints.

### 11.1. Supplementary-Plan Items That Were Implemented

1. Benchmark automation was expanded materially.
   - `scripts/adb_backend_benchmark_worker.sh` became the stable app-side benchmark path
   - `scripts/adb_multi_model_benchmark_matrix.sh` now supports repeat counts and median rollups
   - this is documented in `qnn_feasibility_plan.md` and `docs/qnn_sdk_setup.md`

2. OpenCL layer sweeps were done in the exact spirit of the supplementary plan.
   - multiple `n_gpu_layers` configurations were measured
   - result: `n_gpu_layers=4` was the best observed OpenCL point, and more offload got worse

3. The project did pivot toward QNN / NPU work.
   - this matches the supplementary plan at the strategic level
   - however, the repo did it through direct QAIRT / QNN integration, not through a prebuilt QNN-enabled llama.cpp replacement

4. Backend reporting and diagnostics improved.
   - false CPU normalization was removed
   - the app now exposes backend state much more explicitly
   - this partially matches the supplementary-plan call for stronger diagnostics

### 11.2. Supplementary-Plan Items That Were Explored But Did Not Win

1. "Fix OpenCL first and make it beat CPU" was explored seriously.
   - the repo measured it instead of assuming it
   - the result stayed below CPU for the current workload because fused-op fallback and mixed CPU/GPU scheduling dominated

2. Continuing to spend most effort on OpenCL was rejected by evidence.
   - once the repo had enough measurements, QNN became the more rational path

### 11.3. Supplementary-Plan Items That Were Not Implemented

1. No direct `Gemma 4 26B-A4B` implementation was completed in this repo.
   - the actual real-model work shifted to Qwen, GPT-2, and finally TinyLlama as practical QNN / Genie bring-up targets

2. No `chraac/llama-cpp-qnn-builder` integration was done.
   - the repo built its own QNN validation, app integration, and Genie export path instead

3. There is no confirmed implementation of supplementary-plan-specific HTP tuning knobs.
   - no validated repo evidence shows `QNN_HTP_GRAPH_FINALIZE_OPTIMIZATIONS=1` or `QNN_POWER_SAVER` became part of the working path

4. No speculative decoding, multimodal QNN chat path, LiteRT, ExecuTorch, or MLC fallback implementation was completed.
   - those remained ideas, not landed workstreams

### 11.4. Honest Summary Of The Supplementary Plan's Role

The supplementary plan mattered because it pushed the project to think beyond small OpenCL tuning wins and toward NPU/QNN as the real candidate for a large throughput jump.

But the actual repo implementation took a different route:

- OpenCL did not beat CPU
- Vulkan crashed in the vendor driver
- app subprocess execution was blocked by Android app policy
- in-process JNI QNN execution became the breakthrough
- Genie/TinyLlama became the first real-LLM success path

So the correct reading is:

- the supplementary plan influenced the direction
- some of its ideas were implemented
- many of its concrete proposed steps were replaced by a more direct QAIRT / QNN / Genie route

## 12. What Is Not Solved Yet

The current unsolved problem is no longer "Can QNN run at all?".

The current unsolved problem is:

- how to get the real LLM path to execute on HTP / NPU rather than falling back to CPU

Observed current state:

- TinyLlama generation succeeds
- logcat shows `QNN_CPU`
- the current real-LLM validation path is the CPU-realized Genie path, not a proven HTP path

Meaning:

- real LLM execution is proven
- real NPU acceleration for that LLM path is not yet proven

This is an important distinction. The project has crossed the "real model runs" milestone, but not the "real model runs on HTP and beats CPU" milestone.

## 13. Main Issues Faced Across The Whole Project

This is the condensed issue list from beginning to current state.

### 13.1. CPU / Build Issues

- native code was accidentally built with `-O0`
- low performance looked like a runtime problem but was primarily a build-configuration problem

### 13.2. OpenCL Issues

- OpenCL was slower than CPU on the current model
- fused Gated Delta Net fallback created mixed CPU/GPU execution overhead
- higher `n_gpu_layers` made performance worse
- Adreno-specific kernel flag testing introduced hangs

### 13.3. Vulkan Issues

- Vulkan enumerated and loaded
- first decode crashed inside Qualcomm's Vulkan driver

### 13.4. QNN App Packaging Issues

- app namespace loadability for HTP-related libs was incomplete
- `libcdsprpc.so` had to be declared and runtime-preloaded dependencies had to be managed carefully

### 13.5. Android App Exec Policy

- packaged subprocess execution from app-private storage failed with `Permission denied`
- this killed the executable-launch strategy inside the app

### 13.6. QNN HTP Runtime Setup Issue

- `ADSP_LIBRARY_PATH` separator formatting was wrong
- `:` instead of `;` caused the app-side HTP bring-up failure

### 13.7. Host Toolchain Issues

- QAIRT GenAI tooling required Python 3.10 ABI compatibility
- multiple missing Python packages and native shared library dependencies had to be resolved manually

### 13.8. Model Compatibility Issues

- current Qwen3.5 local model is not a simple first-class text-only transformer target for the bundled composer path
- local shard naming and internal key layout did not fit the composer's assumptions
- GPT-2 composed but did not finalize cleanly in runtime

### 13.9. Current Remaining Genie Issue

- TinyLlama runs, but current realization is still `QNN_CPU`
- HTP path for the real LLM remains the next real blocker

## 14. Key Repo Files That Matter Most

Planning and status:

- `plan.md`
- `cpu_performance_fix_plan.md`
- `suplementary-plan.md`
- `qnn_feasibility_plan.md`
- `docs/qnn_sdk_setup.md`

QNN standalone and validation:

- `scripts/check_qnn_sdk.sh`
- `scripts/build_qnn_smoke.sh`
- `scripts/run_qnn_smoke_android.sh`
- `scripts/run_qairt_sample_android.sh`
- `native/qnn/src/qnn_smoke.cpp`

App-side QNN integration:

- `app/android/app/build.gradle.kts`
- `native/CMakeLists.txt`
- `native/jni/jni_bridge.cpp`
- `native/qnn/src/qnn_runner.cpp`
- `native/engine/src/app_engine.cpp`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/NativeBridge.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/QnnControlRunner.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainActivity.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/MainViewModel.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugBenchmarkWorker.kt`
- `app/android/app/src/main/java/ai/ondev/snapdragonlab/DebugReceiver.kt`
- `app/android/app/src/main/AndroidManifest.xml`

Real LLM / Genie bring-up:

- `scripts/run_genie_cpu_real_llm_android.sh`
- `scripts/run_genie_htp_real_llm_android.sh`
- `.artifacts/qnn-genai/tinyllama_1p1b_q4.bin`
- `.artifacts/qnn-genai/gpt2_124m_q4.bin`
- `.artifacts/qnn-genai/gpt2_auto_q4.bin`

## 15. What Path We Took, In One Sentence Per Phase

1. Build the app and establish CPU as the reliable baseline.
2. Try to push Adreno OpenCL, discover that the current model/backend combination is structurally poor.
3. Try Vulkan, hit vendor-driver crashes.
4. Pivot to QNN / QAIRT as the next serious step-change candidate.
5. Prove QNN outside the app with official/sample-style harnesses.
6. Package QNN into the app and discover subprocess execution is blocked by app sandbox policy.
7. Replace subprocess execution with an in-process JNI QNN runner.
8. Fix app-side HTP bring-up by correcting `ADSP_LIBRARY_PATH` handling.
9. Integrate an experimental engine-visible `qnn` control path into the app.
10. Bring up Qualcomm Genie host tooling and move from control graphs to real LLM export.
11. Fail on Qwen3.5 direct conversion, fail on GPT-2 runtime finalize, then succeed with TinyLlama.
12. Prove real LLM generation on-device, then discover the current Genie flow is still CPU-realized rather than HTP-realized.

## 16. What We Are Doing Now

Current active objective:

- keep the newly working real LLM path
- make the realized backend become HTP / NPU instead of `QNN_CPU`
- then measure whether it actually beats the existing CPU baseline

That is the exact current state of the project.

We are no longer asking:

- can the app run a local model?
- can QNN load on the phone?
- can a real LLM be exported at all?

We are now asking:

- how do we make the real Genie LLM path actually realize on HTP `v79`
- and is it materially faster than the existing CPU baseline when measured with the same discipline?

## 17. Bottom-Line Assessment

This project has made real progress, not just exploratory churn.

The strongest completed milestones are:

- CPU baseline fixed and measured properly
- OpenCL path characterized rather than hand-waved
- Vulkan path investigated and ruled out for concrete reasons
- QNN runtime viability proven on-device
- app-side in-process QNN control execution proven
- first real LLM export and on-device generation via Genie completed

The strongest remaining blocker is:

- converting that real LLM success into real HTP / NPU execution and then into a benchmark win over CPU

That is the next decision point for the project.
