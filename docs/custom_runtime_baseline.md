# Custom Runtime Baseline

Date: 2026-04-11

Purpose: record the current repo state before starting the custom runtime reset.

## Repository State

This workspace does not currently have `.git` metadata:

```text
git rev-parse --is-inside-work-tree
fatal: not a git repository (or any of the parent directories): .git
```

Because of that, branch/tag actions from `RESET_PLAN_V2.md` are not available in this checkout. Treat this file as the baseline marker unless the project is later placed inside a git repository.

## Local Build Baseline

Command:

```bash
./gradlew :app:assembleDebug
```

Result:

```text
BUILD SUCCESSFUL in 9s
41 actionable tasks: 13 executed, 28 up-to-date
```

Interpretation: the existing Android app and native CMake integration are build-green before custom runtime work begins.

## Device Baseline

Command:

```bash
adb devices
```

Result:

```text
List of devices attached
```

Interpretation: no adb device was connected for this baseline capture. CPU benchmark and QNN smoke results were not rerun on-device in this pass.

## Model Baseline

Chosen model:

```text
/home/curious/models/TinyLlama-1.1B-Chat-v1.0
```

Directory size:

```text
2.1G
```

Key files:

```text
config.json 608 bytes
generation_config.json 124 bytes
model.safetensors 2200119864 bytes
tokenizer.json 1842767 bytes
tokenizer.model 499723 bytes
tokenizer_config.json 1289 bytes
```

Weight checksum:

```text
sha256(model.safetensors)=6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933
```

## Sidecar Status

QAIRT/QNN remains preserved as sidecar work. The current repo docs report prior validation of QNN runtime packaging, in-process QNN execution, and HTP `v79` control-path reachability. Those claims were not revalidated in this baseline because no adb device was connected.

Use these commands for the next device-connected baseline:

```bash
./scripts/sidecar_qnn/check_qnn_sdk.sh --adb-serial <serial>
./scripts/sidecar_qnn/run_qnn_smoke_android.sh --backend cpu --adb-serial <serial>
./scripts/sidecar_qnn/run_qnn_smoke_android.sh --backend htp-v79 --adb-serial <serial>
./scripts/sidecar_qnn/adb_backend_benchmark_worker.sh <serial> cpu smoke -1 240 primary-model.gguf
```

## Baseline Rule

Custom runtime work may start from this point. If a device becomes available before Milestone 1 implementation begins, append on-device CPU and QNN smoke results here rather than changing the reset plan.
