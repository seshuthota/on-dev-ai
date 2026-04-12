#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_CPP_DIR="${REPO_ROOT}/native/third_party/llama.cpp"

BUILD_DIR="${REPO_ROOT}/.artifacts/llama-cpp/build-android"
INSTALL_DIR="${REPO_ROOT}/.artifacts/llama-cpp/android-install"
ENABLE_PROFILER=0

usage() {
  cat <<'EOF'
Usage: scripts/llama_cpp/build_android.sh [OPTIONS]

Build llama.cpp for Android arm64-v8a using the Android NDK.

Options:
  --build-dir PATH    Build directory (default: .artifacts/llama-cpp/build-android)
  --install-dir PATH  Install directory (default: .artifacts/llama-cpp/android-install)
  --profiler          Enable GGML_PROFILER instrumentation (compile-time).
                      Runtime also requires GGML_PROFILER=1 env var.
  -h, --help          Show this help message

NDK Detection:
  Searches in order: ANDROID_NDK_ROOT, ANDROID_NDK_HOME,
  ANDROID_SDK_ROOT/ndk, ANDROID_HOME/ndk,
  /home/curious/Android/Sdk/ndk/27.3.13750724

Environment:
  ANDROID_NDK_ROOT    Override NDK path directly
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --profiler)
      ENABLE_PROFILER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_latest_ndk() {
  local base="$1"
  find "${base}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n 1
}

detect_ndk_root() {
  if [[ -n "${ANDROID_NDK_ROOT:-}" && -d "${ANDROID_NDK_ROOT}" ]]; then
    printf '%s\n' "${ANDROID_NDK_ROOT}"
    return 0
  fi
  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "${ANDROID_NDK_HOME}" ]]; then
    printf '%s\n' "${ANDROID_NDK_HOME}"
    return 0
  fi
  if [[ -n "${ANDROID_SDK_ROOT:-}" && -d "${ANDROID_SDK_ROOT}/ndk" ]]; then
    resolve_latest_ndk "${ANDROID_SDK_ROOT}/ndk"
    return 0
  fi
  if [[ -n "${ANDROID_HOME:-}" && -d "${ANDROID_HOME}/ndk" ]]; then
    resolve_latest_ndk "${ANDROID_HOME}/ndk"
    return 0
  fi
  if [[ -d "/home/curious/Android/Sdk/ndk/27.3.13750724" ]]; then
    printf '/home/curious/Android/Sdk/ndk/27.3.13750724\n'
    return 0
  fi
  return 1
}

NDK_ROOT="$(detect_ndk_root)" || true
if [[ -z "${NDK_ROOT}" ]]; then
  echo "Error: Could not detect Android NDK root." >&2
  echo "Please set ANDROID_NDK_ROOT, ANDROID_NDK_HOME, ANDROID_SDK_ROOT/ndk, or ANDROID_HOME/ndk" >&2
  exit 1
fi

TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Error: Android toolchain file not found: ${TOOLCHAIN_FILE}" >&2
  exit 1
fi

echo "[info] NDK root: ${NDK_ROOT}"
echo "[info] Build directory: ${BUILD_DIR}"
echo "[info] Install directory: ${INSTALL_DIR}"
echo "[info] Profiler compile: ${ENABLE_PROFILER}"

mkdir -p "${BUILD_DIR}"

PROFILER_C_FLAGS="-march=armv8.7a"
PROFILER_CXX_FLAGS="-march=armv8.7a"
if [[ "${ENABLE_PROFILER}" -eq 1 ]]; then
  PROFILER_C_FLAGS+=" -DGGML_PROFILER"
  PROFILER_CXX_FLAGS+=" -DGGML_PROFILER"
fi

cmake \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="${PROFILER_C_FLAGS}" \
  -DCMAKE_CXX_FLAGS="${PROFILER_CXX_FLAGS}" \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -B "${BUILD_DIR}" \
  -S "${LLAMA_CPP_DIR}"

cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

cmake --install "${BUILD_DIR}" --prefix "${INSTALL_DIR}" --config Release

if [[ ! -f "${INSTALL_DIR}/bin/llama-cli" ]]; then
  echo "Error: llama-cli not found after install: ${INSTALL_DIR}/bin/llama-cli" >&2
  exit 1
fi

if [[ ! -f "${INSTALL_DIR}/bin/llama-bench" ]]; then
  echo "Error: llama-bench not found after install: ${INSTALL_DIR}/bin/llama-bench" >&2
  exit 1
fi

if [[ ! -f "${INSTALL_DIR}/bin/llama-completion" ]]; then
  echo "Error: llama-completion not found after install: ${INSTALL_DIR}/bin/llama-completion" >&2
  exit 1
fi

echo "[info] Build complete. Install directory: ${INSTALL_DIR}"
echo "[info] llama-cli: ${INSTALL_DIR}/bin/llama-cli"
echo "[info] llama-bench: ${INSTALL_DIR}/bin/llama-bench"
echo "[info] llama-completion: ${INSTALL_DIR}/bin/llama-completion"
