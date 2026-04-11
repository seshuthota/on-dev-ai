# Repository Guidelines

## Project Structure & Module Organization
- `app/android/app/`: Android app module (Kotlin + Compose UI, ViewModel, JNI bridge, manifest/resources).
- `native/`: current C++ engine/JNI/QNN code; `native/custom/` is reserved for the reset runtime.
- `native/third_party/llama.cpp` and `native/third_party/Vulkan-Headers` are Git submodules; initialize them after clone.
- `scripts/android/`: Android install and model-management helpers. Root wrappers exist for common commands.
- `scripts/sidecar_qnn/`: QAIRT/QNN/Genie tooling retained as sidecar, not mainline reset work.
- `docs/`: current custom-runtime docs plus archived pre-reset history under `docs/archive/pre_reset/`.
- Generated/vendor-local outputs such as `.artifacts/`, `.conda-qairt310/`, and `v2.45.0.260326/` should stay outside the repo.

## Build, Test, and Development Commands
- `./gradlew :app:assembleDebug`: builds debug APK (includes native CMake build).
- `git submodule update --init --recursive`: fetches third-party native dependencies after clone.
- `./scripts/install_debug.sh`: installs `app-debug.apk` on a connected device.
- `./gradlew :app:assembleDebug -Pondevai.enableQnn=true`: build with QNN runtime packaging enabled.
- `./gradlew :app:testDebugUnitTest`: run JVM unit tests.
- `./gradlew :app:connectedDebugAndroidTest`: run instrumentation tests on device/emulator.
- `./gradlew :app:lint`: run Android lint checks.
- `./scripts/sidecar_qnn/check_qnn_sdk.sh --adb-serial <serial>`: validate QAIRT SDK + host/device prerequisites.

## Coding Style & Naming Conventions
- Kotlin: 4-space indentation, `PascalCase` types, `camelCase` members, package under `ai.ondev.snapdragonlab`.
- C++: follow existing `native/` style (`k`-prefixed constants, clear helper functions, minimal header surface).
- Scripts: keep `bash` scripts strict (`set -euo pipefail`) and Python helpers small/composable.
- No repo-wide formatter config is checked in; use IDE/clang-format defaults conservatively and avoid broad reformat-only diffs.

## Testing Guidelines
- Add unit tests under `app/android/app/src/test/` and device tests under `app/android/app/src/androidTest/`.
- Prefer names like `FeatureNameTest` and test methods that describe behavior.
- For native/QNN sidecar changes, include at least one reproducible validation, for example `scripts/sidecar_qnn/run_qnn_smoke_android.sh --backend cpu`.
- No explicit coverage gate is defined; new features should include targeted tests or documented manual verification.

## Commit & Pull Request Guidelines
- This workspace snapshot does not include `.git` history, so no local commit convention can be inferred.
- Use clear, scoped commit subjects (example: `native: fix QNN backend preload order`).
- PRs should include: purpose, files/areas changed, exact verification commands run, and device/backend details (CPU/OpenCL/QNN).
- For UI changes, include screenshots; for runtime/backend changes, include relevant log snippets (for example `adb logcat -s OnDevAI`).

## Security & Configuration Tips
- Start from `.env.example` for local paths (`QAIRT_SDK_ROOT`, `ANDROID_NDK_ROOT`, adb serial).
- Do not commit machine-specific absolute paths, device identifiers, or model binaries.
