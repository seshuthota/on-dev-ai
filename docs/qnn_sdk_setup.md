# QAIRT / QNN SDK Setup

## Purpose

This repo uses a pinned local QAIRT SDK for the QNN feasibility spike. This note records the exact SDK layout, host prerequisites, and the first recommended smoke path.

This is a setup note for the spike branch, not a production integration guide.

## Pinned SDK

- Version: `2.45.0.260326`
- External local root: `/home/curious/Documents/hermes-projects/OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326`
- Release family: `QAIRT`

Recommended environment variables:

```bash
export QAIRT_SDK_ROOT=/home/curious/Documents/hermes-projects/OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326
export QNN_SDK_ROOT="${QAIRT_SDK_ROOT}"
export ANDROID_NDK_ROOT=/home/curious/Android/Sdk/ndk/27.3.13750724
export PATH="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}"
export ONDEVAI_QNN_ADB_SERIAL=192.168.29.11:41993
```

Before running any bundled SDK host tools, source the SDK environment:

```bash
source "${QAIRT_SDK_ROOT}/bin/envsetup.sh"
```

The SDK env script sets `QAIRT_SDK_ROOT`, the legacy `QNN_SDK_ROOT` alias, `PYTHONPATH`, `PATH`, and `LD_LIBRARY_PATH`.

## Repo Tooling Baseline

Current Android app settings in this repo:

- Compile SDK: `35`
- Target SDK: `35`
- Min SDK: `31`
- NDK version: `27.3.13750724`

Host tools expected on the machine:

- `adb`
- `cmake`
- `java`
- `python3`
- `clang++`
- `ndk-build`

Use the repo-local validator to confirm the full setup:

```bash
scripts/sidecar_qnn/check_qnn_sdk.sh --adb-serial 192.168.29.11:41993
```

Use the repo-local sample wrapper to reproduce the official QAIRT Android sample flow:

```bash
scripts/sidecar_qnn/run_qairt_sample_android.sh --backend cpu --adb-serial 192.168.29.11:41993
scripts/sidecar_qnn/run_qairt_sample_android.sh --backend htp-v79 --adb-serial 192.168.29.11:41993
```

For a repo-owned native Workstream C harness (outside app UI code), use:

```bash
scripts/sidecar_qnn/build_qnn_smoke.sh

scripts/sidecar_qnn/run_qnn_smoke_android.sh \
  --backend cpu \
  --adb-serial 192.168.29.11:41993

scripts/sidecar_qnn/run_qnn_smoke_android.sh \
  --backend htp-v79 \
  --adb-serial 192.168.29.11:41993
```

For Workstream D app packaging toggle:

```bash
# default path (no QNN runtime packaged)
./gradlew :app:assembleDebug

# package QNN runtime libs + HTP skels into APK
./gradlew :app:assembleDebug -Pondevai.enableQnn=true

# optional SDK override
./gradlew :app:assembleDebug \
  -Pondevai.enableQnn=true \
  -Pondevai.qairtSdkRoot=/absolute/path/to/qairt/2.45.0.260326
```

Current Workstream E behavior in a QNN-enabled APK:

- app settings expose an experimental `QNN` backend target
- runtime info reports `qnn_compiled`, `qnn_packaged_libs`, `qnn_packaged_count`, and `qnn_status`
- selecting `qnn` fails fast with an explicit "GGUF app-side QNN execution is not implemented yet" message
- there is no silent fallback to CPU when `qnn` is requested
- app-side HTP/QNN stub loadability on this device required:
  - `<uses-native-library android:name="libcdsprpc.so" android:required="false" />`
  - explicit `System.loadLibrary(...)` preload of packaged QNN libs in the app process

For app-side worker benchmarks across multiple GGUF models, use:

```bash
scripts/sidecar_qnn/adb_multi_model_benchmark_matrix.sh \
  192.168.29.11:41993 cpu smoke -1 \
  "primary-model.gguf,control-llama3.2-3b.gguf"

# repeat each model 3x and include median lines
scripts/sidecar_qnn/adb_multi_model_benchmark_matrix.sh \
  192.168.29.11:41993 cpu micro -1 \
  "primary-model.gguf,control-llama3.2-3b.gguf" \
  240 /tmp/ondevai_matrix_cpu_micro_repeat3.txt 3
```

To install extra GGUF files into app internal storage for those benchmarks:

```bash
scripts/android/adb_install_model_internal.sh \
  192.168.29.11:41993 /absolute/path/to/control-llama3.2-3b.gguf \
  control-llama3.2-3b.gguf

scripts/android/adb_list_internal_models.sh 192.168.29.11:41993
```

Model argument behavior in worker benchmarks:

- if model argument contains `/`, it is treated as an absolute `model_path`
- otherwise it is treated as a filename under app internal `files/models/`
- default remains `primary-model.gguf`
- worker completion detection is history-based (`files/benchmarks/history.jsonl`), not dependent on `OnDevAI` logcat lines
- optional: `ONDEVAI_BENCH_FORCE_STOP=1` re-enables force-stop before each run (default is `0`)
- matrix script emits per-repeat lines (`model_repeat_*`) and median rollups (`model_median*`)

## Required SDK Layout

The first spike path depends on these SDK files being present.

Host-side tools:

- `bin/envsetup.sh`
- `bin/x86_64-linux-clang/qnn-model-lib-generator`
- `bin/x86_64-linux-clang/qnn-context-binary-generator`

Android-side tools:

- `bin/aarch64-android/qnn-net-run`
- `bin/aarch64-android/qnn-context-binary-generator`

Headers:

- `include/QNN/QnnInterface.h`
- `include/QNN/QnnBackend.h`
- `include/Genie/GenieDialog.h`

Android runtime libraries:

- `lib/aarch64-android/libQnnCpu.so`
- `lib/aarch64-android/libQnnGpu.so`
- `lib/aarch64-android/libQnnSystem.so`
- `lib/aarch64-android/libQnnHtp.so`
- `lib/aarch64-android/libQnnHtpPrepare.so`
- `lib/aarch64-android/libQnnHtpV79Stub.so`
- `lib/aarch64-android/libQnnHtpV81Stub.so`
- `lib/aarch64-android/libGenie.so`
- `lib/aarch64-android/libQnnGenAiTransformer.so`

Hexagon artifacts for HTP packaging:

- `lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so`
- `lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so`

Example assets for the first smoke:

- `examples/QNN/converter/models/qnn_model_float.cpp`
- `examples/QNN/converter/models/qnn_model_float.bin`
- `examples/QNN/converter/models/input_list_float.txt`

## Device Notes

Current adb-connected phone:

- Serial: `192.168.29.11:41993`
- Model: `I2401`
- Board platform: `sun`
- SoC: `SM8750`
- Fingerprint prefix: `iQOO/I2401i/I2401:16/...`

Working assumption for this device class:

- first HTP backend guess: `v79`
- fallback HTP backend guess: `v81`

That assumption was based on the device SoC class and the SDK's own `hexagon-v79` and `hexagon-v81` runtime packaging paths. It has now been validated on-device with the official QAIRT `qnn-net-run` sample path for `htp-v79`.

## First Smoke Path

Do not start with the LPAI sample app. The SDK docs show that path requires signed Hexagon artifacts and root execution, which is the wrong first milestone for this repo.

Start with the lower-risk Android path:

1. Use `qnn-model-lib-generator` on the bundled example model.
2. Push `qnn-net-run`, `libc++_shared.so`, the generated model library, and the required backend runtime libraries.
3. Validate CPU first.
4. Validate HTP with `libQnnHtp.so`, `libQnnHtpPrepare.so`, the target stub, the matching Hexagon skel, and `ADSP_LIBRARY_PATH`.

Why this path first:

- it does not require app integration
- it uses official SDK example assets
- it exercises the same Android runtime packaging constraints we will hit later
- it gives a clean answer on whether the device accepts QNN HTP runtime loading

Validated on `2026-04-09`:

- QNN sample CPU path completed end-to-end on device
- QNN sample HTP `v79` path completed end-to-end on device
- Android runtime packaging is viable on this `SM8750` / Android `16` phone
- repo-native `qnn_smoke` harness runs on-device and reports end-to-end latency for `cpu` and `htp-v79`
- app build/install works with QNN packaging disabled and enabled via `-Pondevai.enableQnn=true`
- app runtime now exposes `qnn` as an experimental backend target and reports explicit not-yet-implemented load failures
- app runtime inventory now reaches `qnn_status=runtime_packaged` after the `libcdsprpc.so` manifest declaration and QNN preload path
- a packaged control-model subprocess path was tested and is blocked by Android app exec restrictions (`error=13 Permission denied` when launching extracted `qnn_smoke`)
- that means the correct next implementation path is an in-process native/JNI QNN runner, not `ProcessBuilder` over extracted binaries
- the in-process native/JNI control runner is now validated on-device for both `cpu` and `htp-v79`
- the app-side HTP `14001` failure was caused by an `ADSP_LIBRARY_PATH` formatting bug: QAIRT FastRPC on Android expects `;` separators here, not `:`
- after extracting `libQnnHtpV79Skel.so` into app files storage and switching `ADSP_LIBRARY_PATH` to `;`, app-side `deviceCreate` and `contextCreate` succeed on `htp-v79`
- the app engine now has a real experimental `qnn` load path backed by the in-process control-model runner, with validated `LoadModel(..., "qnn")` and benchmark execution on-device

Reference command shape:

```bash
scripts/sidecar_qnn/run_qairt_sample_android.sh --backend htp-v79 --adb-serial 192.168.29.11:41993
```

## Known SDK Quirk

In this local SDK drop, the bundled `qnn-platform-validator` Python entrypoint does not currently resolve its helper imports correctly in a default shell. Treat it as optional.

For this repo, use `scripts/sidecar_qnn/check_qnn_sdk.sh` as the authoritative setup check and keep `qnn-platform-validator` as a secondary probe only if its Python packaging is corrected later.

There is also a host-side path footgun: `qnn-model-lib-generator` expects `clang++` to be discoverable in `PATH`. On this machine, that requirement is satisfied by prepending the Android NDK LLVM toolchain bin directory shown above.

## Host Context-Binary Workflow

For explicit HTP-targeted `ctx-bins` generation on host, use:

```bash
scripts/sidecar_qnn/build_qnn_sample_ctx_bin.sh --backend htp-v79
```

This runs a reproducible sample pipeline:

1. builds a host model library from SDK sample model sources (`qnn_model_8bit_quantized`)
2. invokes `qnn-context-binary-generator` against `libQnnHtp.so`
3. writes context artifacts under:
   - `.artifacts/qnn-context/sample_qnn_model_8bit_quantized/ctx/`

Generated by the current validated run:

- `.artifacts/qnn-context/sample_qnn_model_8bit_quantized/ctx/qnn_model_8bit_quantized_htp-v79.bin`
- `.artifacts/qnn-context/sample_qnn_model_8bit_quantized/ctx/qnn_model_8bit_quantized_htp-v79_backend.bin`

For custom model libraries, use the generic wrapper:

```bash
scripts/sidecar_qnn/generate_qnn_context_bins.sh \
  --model-so /abs/path/libyour_model.so \
  --backend htp-v79 \
  --binary-file your_model_htp_v79 \
  --backend-binary your_model_htp_v79_backend
```

Note:

- these scripts set up the host runtime and include a fallback for `libc++.so.1` from `/home/curious/Documents/hermes-projects/OnDevAI_external/.conda-qairt310/lib` when needed
- this host context generation path is the bridge required by the new `genie_htp_real_llm` execution track

## Genie LLM Bring-Up (TinyLlama)

As of `2026-04-10`, this repo can run a real LLM through QAIRT Genie on-device using a composed `.bin` model:

- model source: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- composer output: `.artifacts/qnn-genai/tinyllama_1p1b_q4.bin`
- current CPU-realized runner script: `scripts/sidecar_qnn/run_genie_cpu_real_llm_android.sh`

Example CPU run:

```bash
scripts/sidecar_qnn/run_genie_cpu_real_llm_android.sh \
  --adb-serial 192.168.29.11:41993 \
  --model-bin /home/curious/Documents/hermes-projects/OnDevAI/.artifacts/qnn-genai/tinyllama_1p1b_q4.bin \
  --tokenizer-json /home/curious/models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json \
  --n-vocab 32000 \
  --bos-token 1 \
  --eos-token 2 \
  --context-size 1024 \
  --prompt "Hello from OnDevAI"
```

Observed behavior on `192.168.29.11:41993`:

- prompt->generation path succeeds end-to-end and returns text
- runtime logs currently show `QNN_CPU` execution for this flow
- this path should be treated as `genie_cpu_real_llm`, not as HTP proof
- `scripts/sidecar_qnn/run_genie_t2t_android.sh` is now only a compatibility wrapper
- HTP acceleration for this Genie LLM path is not yet confirmed and needs dedicated backend config work through `scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh`

## QAIRT Python Builder Path (HTP Real-LLM)

For the new `genie_htp_real_llm` path, use a Python 3.10 environment with QAIRT API dependencies.

Default project environment:

- Python: `/home/curious/Documents/hermes-projects/OnDevAI_external/.conda-qairt310/bin/python`

Bootstrap dependencies:

```bash
scripts/sidecar_qnn/setup_qairt_python_env.sh
```

Build container + runtime manifest from model input (exports/ONNX/GGUF):

```bash
scripts/sidecar_qnn/build_genie_htp_real_llm_assets.sh \
  --model-input /abs/path/model_or_exports \
  --chipset SM8750
```

Output:

- container directory: `.artifacts/genie-htp-real-llm/container/`
- manifest: `.artifacts/genie-htp-real-llm/container/htp_runtime_manifest.json`

To build and immediately run the generated `ctx-bins` on Android:

```bash
scripts/sidecar_qnn/build_genie_htp_real_llm_assets.sh \
  --model-input /abs/path/model_or_exports \
  --run-android \
  --adb-serial 192.168.29.11:41993
```

Notes:

- the wrapper auto-reads `ctx_bins`, tokenizer path, and token ids from the manifest
- if ids are missing, pass runtime overrides:
  - `--tokenizer-json`, `--n-vocab`, `--bos-token`, `--eos-token`
- for GGUF-derived ONNX pipelines, `--embedding-lut off` may be required when activation encodings are empty
- wrapper now always emits an intake report before build:
  - `.artifacts/genie-htp-real-llm/container/model_intake_report.json`
- known blockers are now fail-fast by default:
  - `SimplifiedLayerNormalization` + high opset risk
  - very large ONNX external-data protobuf serialization risk
  - GGUF architecture unsupported by current QAIRT GGUF builder (example observed: `qwen35`)
- override fail-fast only when intentionally testing:
  - `--ignore-intake-blockers`
- ONNX preflight is integrated in wrapper:
  - `--onnx-preflight auto|on|off` (default: `auto`)
  - shape overrides: `--onnx-preflight-batch-size`, `--onnx-preflight-prompt-len`, `--onnx-preflight-ctx-len`, `--onnx-preflight-past-len`
  - includes automatic lowering: `SimplifiedLayerNormalization -> LayerNormalization` for default-domain ONNX ops
- wrapper now defaults `QAIRT_TMP_DIR` to a short path (`/tmp/ondevai_qairt_tmp`) to avoid multiprocessing AF_UNIX path length failures

Quick intake-only check for any model path:

```bash
/home/curious/Documents/hermes-projects/OnDevAI_external/.conda-qairt310/bin/python scripts/sidecar_qnn/model_intake_report.py \
  --model-input /abs/path/model_or_exports \
  --output-json /tmp/model_intake.json
```

## Manual ONNX Patch Helpers (Optional)

The wrapper already applies preflight patching when enabled. Manual helpers are kept for targeted debugging:

```bash
/home/curious/Documents/hermes-projects/OnDevAI_external/.conda-qairt310/bin/python scripts/sidecar_qnn/fix_onnx_reduce_axes.py \
  --input /abs/path/model_Q4_K_M.onnx

/home/curious/Documents/hermes-projects/OnDevAI_external/.conda-qairt310/bin/python scripts/sidecar_qnn/fix_onnx_constant_valueinfo_shapes.py \
  --input /abs/path/model_Q4_K_M.onnx
```

These scripts currently address:

- `ReduceMean` opset-18+ mismatch (`axes` attr vs tensor input)
- constant/value_info scalar-vs-vector shape metadata mismatches
- large ONNX save path (`>2GB`) by writing with external data mode
