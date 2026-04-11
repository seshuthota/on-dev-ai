#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/native/qnn"

BUILD_DIR="/tmp/ondevai_qnn_smoke_build"
ANDROID_ABI="arm64-v8a"
ANDROID_PLATFORM=31
BUILD_TYPE="Release"
GENERATOR="Unix Makefiles"

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/build_qnn_smoke.sh [--build-dir PATH] [--abi ABI] [--android-platform API] [--generator NAME]

Builds the standalone QNN smoke harness binary (`qnn_smoke`) for Android.
EOF
}

resolve_latest_ndk() {
  local base="$1"
  find "${base}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n 1
}

detect_ndk_root() {
  if [[ -n "${ANDROID_NDK_ROOT:-}" && -d "${ANDROID_NDK_ROOT}" ]]; then
    printf '%s\n' "${ANDROID_NDK_ROOT}"
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

  if [[ -d "${HOME}/Android/Sdk/ndk" ]]; then
    resolve_latest_ndk "${HOME}/Android/Sdk/ndk"
    return 0
  fi

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --abi)
      ANDROID_ABI="$2"
      shift 2
      ;;
    --android-platform)
      ANDROID_PLATFORM="$2"
      shift 2
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --generator)
      GENERATOR="$2"
      shift 2
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

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "QNN smoke source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

if [[ ! "${ANDROID_PLATFORM}" =~ ^[0-9]+$ ]]; then
  echo "android platform must be numeric: ${ANDROID_PLATFORM}" >&2
  exit 1
fi

NDK_ROOT="$(detect_ndk_root || true)"
if [[ -z "${NDK_ROOT}" ]]; then
  echo "Could not detect Android NDK root." >&2
  exit 1
fi

TOOLCHAIN_FILE="${NDK_ROOT}/build/cmake/android.toolchain.cmake"
if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Android CMake toolchain not found: ${TOOLCHAIN_FILE}" >&2
  exit 1
fi

cmake \
  -S "${SRC_DIR}" \
  -B "${BUILD_DIR}" \
  -G "${GENERATOR}" \
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="android-${ANDROID_PLATFORM}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"

cmake --build "${BUILD_DIR}" --target qnn_smoke -j"$(nproc)"

BIN_PATH="${BUILD_DIR}/qnn_smoke"
if [[ ! -f "${BIN_PATH}" ]]; then
  echo "Build finished but binary not found: ${BIN_PATH}" >&2
  exit 1
fi

echo "output_binary=${BIN_PATH}"
