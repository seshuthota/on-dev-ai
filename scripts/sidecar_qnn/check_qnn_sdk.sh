#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/check_qnn_sdk.sh [--sdk-root PATH] [--adb-serial SERIAL]

Checks the pinned QAIRT/QNN SDK tree, basic host prerequisites, and optional adb device info.

Options:
  --sdk-root PATH     Override QAIRT/QNN SDK root
  --adb-serial VALUE  adb serial to inspect
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk-root)
      SDK_ROOT="$2"
      shift 2
      ;;
    --adb-serial)
      ADB_SERIAL="$2"
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

if [[ -d "${SDK_ROOT}" ]]; then
  SDK_ROOT="$(cd "${SDK_ROOT}" && pwd)"
fi

failures=0
warnings=0

info() {
  printf '[info] %s\n' "$*"
}

pass() {
  printf '[ok] %s\n' "$*"
}

warn() {
  warnings=$((warnings + 1))
  printf '[warn] %s\n' "$*"
}

fail() {
  failures=$((failures + 1))
  printf '[fail] %s\n' "$*"
}

check_command() {
  local cmd="$1"
  if command -v "${cmd}" >/dev/null 2>&1; then
    pass "host tool found: ${cmd} -> $(command -v "${cmd}")"
  else
    fail "host tool missing: ${cmd}"
  fi
}

check_file() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    pass "file present: ${path}"
  else
    fail "missing file: ${path}"
  fi
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

detect_adb_serial() {
  local devices
  mapfile -t devices < <(adb devices | awk '$2 == "device" {print $1}')
  if [[ "${#devices[@]}" -eq 1 ]]; then
    printf '%s\n' "${devices[0]}"
    return 0
  fi
  return 1
}

strip_cr() {
  tr -d '\r'
}

HOST_CLANGXX=""
HOST_NDK_BUILD=""

info "repo root: ${REPO_ROOT}"
info "sdk root: ${SDK_ROOT}"

check_command adb
check_command cmake
check_command java
check_command python3

NDK_ROOT=""
if NDK_ROOT="$(detect_ndk_root)"; then
  pass "android ndk found: ${NDK_ROOT}"
else
  warn "android ndk not detected from ANDROID_NDK_ROOT, ANDROID_SDK_ROOT, ANDROID_HOME, or ~/Android/Sdk/ndk"
fi

if command -v clang++ >/dev/null 2>&1; then
  HOST_CLANGXX="$(command -v clang++)"
  pass "host tool found: clang++ -> ${HOST_CLANGXX}"
elif [[ -n "${NDK_ROOT}" && -x "${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++" ]]; then
  HOST_CLANGXX="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
  warn "clang++ is not in PATH; prepend ${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin before running qnn-model-lib-generator"
else
  fail "host tool missing: clang++"
fi

if command -v ndk-build >/dev/null 2>&1; then
  HOST_NDK_BUILD="$(command -v ndk-build)"
  pass "host tool found: ndk-build -> ${HOST_NDK_BUILD}"
elif [[ -n "${NDK_ROOT}" && -x "${NDK_ROOT}/ndk-build" ]]; then
  HOST_NDK_BUILD="${NDK_ROOT}/ndk-build"
  warn "ndk-build is not in PATH; prepend ${NDK_ROOT} before running qnn-model-lib-generator"
else
  fail "host tool missing: ndk-build"
fi

check_file "${SDK_ROOT}/bin/envsetup.sh"
check_file "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator"
check_file "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator"
check_file "${SDK_ROOT}/bin/aarch64-android/qnn-net-run"
check_file "${SDK_ROOT}/include/QNN/QnnInterface.h"
check_file "${SDK_ROOT}/include/Genie/GenieDialog.h"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnCpu.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnGpu.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnSystem.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnHtp.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnHtpV79Stub.so"
check_file "${SDK_ROOT}/lib/aarch64-android/libQnnHtpV81Stub.so"
check_file "${SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
check_file "${SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
check_file "${SDK_ROOT}/examples/QNN/converter/models/qnn_model_float.cpp"
check_file "${SDK_ROOT}/examples/QNN/converter/models/qnn_model_float.bin"
check_file "${SDK_ROOT}/examples/QNN/converter/models/input_list_float.txt"

if bash -lc "source '${SDK_ROOT}/bin/envsetup.sh' >/dev/null 2>&1 && [[ \"\${QAIRT_SDK_ROOT:-}\" == '${SDK_ROOT}' ]]" >/dev/null 2>&1; then
  pass "envsetup.sh initializes QAIRT_SDK_ROOT correctly"
else
  fail "envsetup.sh did not initialize QAIRT_SDK_ROOT as expected"
fi

if grep -q 'from common_utils.adb import Adb' "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-platform-validator" 2>/dev/null && \
   [[ ! -f "${SDK_ROOT}/lib/python/common_utils/adb.py" ]] && \
   [[ -f "${SDK_ROOT}/lib/python/common_utils/protocol/adb.py" ]]; then
  warn "bundled qnn-platform-validator appears to reference a missing Python module path; prefer this repo validator for setup checks"
fi

if [[ -z "${ADB_SERIAL}" ]]; then
  if ADB_SERIAL="$(detect_adb_serial)"; then
    info "auto-selected adb serial: ${ADB_SERIAL}"
  fi
fi

if [[ -n "${ADB_SERIAL}" ]]; then
  if adb -s "${ADB_SERIAL}" get-state >/dev/null 2>&1; then
    pass "adb device available: ${ADB_SERIAL}"
    model="$(adb -s "${ADB_SERIAL}" shell getprop ro.product.model | strip_cr)"
    board="$(adb -s "${ADB_SERIAL}" shell getprop ro.board.platform | strip_cr)"
    soc="$(adb -s "${ADB_SERIAL}" shell getprop ro.soc.model | strip_cr)"
    android_release="$(adb -s "${ADB_SERIAL}" shell getprop ro.build.version.release | strip_cr)"
    fingerprint="$(adb -s "${ADB_SERIAL}" shell getprop ro.build.fingerprint | strip_cr)"
    info "device model: ${model:-unknown}"
    info "device board: ${board:-unknown}"
    info "device soc: ${soc:-unknown}"
    info "android release: ${android_release:-unknown}"
    info "fingerprint: ${fingerprint:-unknown}"

    for dsp_path in /vendor/dsp/cdsp /vendor/lib/rfsa/adsp /system/lib/rfsa/adsp /dsp; do
      if adb -s "${ADB_SERIAL}" shell "[ -d '${dsp_path}' ]" >/dev/null 2>&1; then
        pass "device path present: ${dsp_path}"
      else
        warn "device path missing or inaccessible: ${dsp_path}"
      fi
    done

    case "${soc}" in
      SM8750)
        info "backend hint: start with HTP v79, then try v81 if v79 packaging loads but execution fails"
        ;;
      *)
        warn "no backend hint encoded for soc '${soc:-unknown}'; inspect the QAIRT docs for the correct HTP stub/skel pair"
        ;;
    esac
  else
    fail "adb device not available: ${ADB_SERIAL}"
  fi
else
  warn "no adb serial provided and auto-detection did not find exactly one connected device"
fi

if [[ "${failures}" -gt 0 ]]; then
  printf '\n[summary] %d failure(s), %d warning(s)\n' "${failures}" "${warnings}"
  exit 1
fi

if [[ -n "${HOST_CLANGXX}" ]]; then
  info "effective clang++: ${HOST_CLANGXX}"
fi
if [[ -n "${HOST_NDK_BUILD}" ]]; then
  info "effective ndk-build: ${HOST_NDK_BUILD}"
fi

printf '\n[summary] setup check passed with %d warning(s)\n' "${warnings}"
