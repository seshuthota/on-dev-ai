# Cleanup Manifest

Date: 2026-04-11

Purpose: record the reset cleanup so future work starts from the custom-runtime docs instead of old plan clutter.

## Current Root Docs

Keep these visible at repo root:

- `AGENTS.md`
- `CLAUDE.md`
- `RESET_PLAN_V2.md`
- `MODEL_DECISION.md`

Historical planning docs were moved to:

```text
docs/archive/pre_reset/
```

## Script Layout

Current mainline/helper layout:

```text
scripts/android/
scripts/custom_runtime/
scripts/sidecar_qnn/
```

Root wrappers retained:

```text
scripts/install_debug.sh
scripts/adb_push_model.sh
```

Third-party dependency layout:

```text
native/third_party/llama.cpp        # Git submodule, pinned upstream dependency
native/third_party/Vulkan-Headers   # Git submodule
patches/llama.cpp/ondevai-local.patch
```

The patch file preserves local `llama.cpp` edits that existed before git initialization. It is not applied automatically.

## Externalized Local State

Large local-only state was moved outside the repo:

```text
../OnDevAI_external/v2.45.0.260326/
../OnDevAI_external/.conda-qairt310/
../OnDevAI_external/artifacts_pre_cleanup_2026-04-11/
../OnDevAI_external/generated_cache_pre_cleanup_2026-04-11/
../OnDevAI_external/generated_cache_post_verify_2026-04-11/
```

The repo itself should not depend on these paths for the custom-runtime mainline. QNN/QAIRT sidecar scripts may use them through `QAIRT_SDK_ROOT`, `QNN_SDK_ROOT`, or `ONDEVAI_QAIRT_PYTHON`.

## Verification

Post-cleanup checks completed:

```text
./gradlew :app:assembleDebug
BUILD SUCCESSFUL in 44s
```

After cache recreation, the default build was rerun:

```text
./gradlew :app:assembleDebug
BUILD SUCCESSFUL in 834ms
```

QNN-enabled packaging was also checked against the external SDK path:

```text
./gradlew :app:assembleDebug -Pondevai.enableQnn=true
BUILD SUCCESSFUL in 48s
```

```text
scripts/sidecar_qnn/check_qnn_sdk.sh --sdk-root ../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326
setup check passed with warnings
```

Warnings were limited to missing `clang++`/`ndk-build` in `PATH`, a known optional `qnn-platform-validator` packaging issue, and no adb device auto-detected.
