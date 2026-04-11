#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"
BACKEND="cpu"

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/run_qairt_sample_android.sh [--backend NAME] [--adb-serial SERIAL] [--sdk-root PATH]

Wraps the official QAIRT Android qnn-net-run sample with the repo's pinned defaults.

Examples:
  scripts/sidecar_qnn/run_qairt_sample_android.sh --backend cpu
  scripts/sidecar_qnn/run_qairt_sample_android.sh --backend htp-v79 --adb-serial 192.168.29.11:41993
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --adb-serial)
      ADB_SERIAL="$2"
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT="$2"
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

detect_adb_serial() {
  local devices
  mapfile -t devices < <(adb devices | awk '$2 == "device" {print $1}')
  if [[ "${#devices[@]}" -eq 1 ]]; then
    printf '%s\n' "${devices[0]}"
    return 0
  fi
  return 1
}

if [[ ! -f "${SDK_ROOT}/examples/QNN/NetRun/android/android-qnn-net-run.sh" ]]; then
  echo "QAIRT Android sample not found under ${SDK_ROOT}" >&2
  exit 1
fi

NDK_ROOT="$(detect_ndk_root || true)"
if [[ -z "${NDK_ROOT}" ]]; then
  echo "Could not detect ANDROID_NDK_ROOT" >&2
  exit 1
fi

if [[ -z "${ADB_SERIAL}" ]]; then
  ADB_SERIAL="$(detect_adb_serial || true)"
fi

if [[ -z "${ADB_SERIAL}" ]]; then
  echo "No adb serial provided and auto-detection did not find exactly one connected device" >&2
  exit 1
fi

export ANDROID_NDK_ROOT="${NDK_ROOT}"
export PATH="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${ANDROID_NDK_ROOT}:${PATH}"
export ANDROID_SERIAL="${ADB_SERIAL}"

printf '[info] sdk root: %s\n' "${SDK_ROOT}"
printf '[info] ndk root: %s\n' "${ANDROID_NDK_ROOT}"
printf '[info] adb serial: %s\n' "${ANDROID_SERIAL}"
printf '[info] backend: %s\n' "${BACKEND}"

bash "${SDK_ROOT}/examples/QNN/NetRun/android/android-qnn-net-run.sh" -b "${BACKEND}"
