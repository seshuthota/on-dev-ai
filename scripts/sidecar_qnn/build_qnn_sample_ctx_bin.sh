#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
MODEL="qnn_model_8bit_quantized"
OUTPUT_DIR="${REPO_ROOT}/.artifacts/qnn-context/sample_${MODEL}"
BACKEND="htp-v79"
BUILD_ANDROID_LIB=0

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/build_qnn_sample_ctx_bin.sh [options]

Builds a sample QNN model library from SDK converter examples and generates
a host-side context binary for backend validation.

Options:
  --backend cpu|gpu|htp-v79   Backend to target (default: htp-v79)
  --sdk-root PATH             QAIRT SDK root
  --output-dir PATH           Output directory (default: .artifacts/qnn-context/sample_<model>)
  --with-android-lib          Also build aarch64-android model library (requires ndk-build)
  -h, --help                  Show help
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
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --with-android-lib)
      BUILD_ANDROID_LIB=1
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

if [[ ! -d "${SDK_ROOT}" ]]; then
  echo "SDK root not found: ${SDK_ROOT}" >&2
  exit 1
fi

MODEL_CPP="${SDK_ROOT}/examples/QNN/converter/models/${MODEL}.cpp"
MODEL_BIN="${SDK_ROOT}/examples/QNN/converter/models/${MODEL}.bin"
if [[ ! -f "${MODEL_CPP}" || ! -f "${MODEL_BIN}" ]]; then
  echo "Sample model files not found under SDK examples: ${MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
MODEL_LIBS_DIR="${OUTPUT_DIR}/model_libs"
CTX_DIR="${OUTPUT_DIR}/ctx"
mkdir -p "${MODEL_LIBS_DIR}" "${CTX_DIR}"

set +u
source "${SDK_ROOT}/bin/envsetup.sh"
set -u

need_toolchain_fix=0
if ! command -v clang++ >/dev/null 2>&1; then
  need_toolchain_fix=1
fi
if [[ "${BUILD_ANDROID_LIB}" == "1" ]] && ! command -v ndk-build >/dev/null 2>&1; then
  need_toolchain_fix=1
fi
if [[ "${need_toolchain_fix}" == "1" ]]; then
  NDK_ROOT="$(detect_ndk_root || true)"
  if [[ -n "${NDK_ROOT}" ]]; then
    export ANDROID_NDK_ROOT="${NDK_ROOT}"
    export PATH="${ANDROID_NDK_ROOT}:${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}"
  fi
fi

if ! command -v clang++ >/dev/null 2>&1; then
  echo "clang++ not found. Install clang++ or set ANDROID_NDK_ROOT so NDK clang++ can be used." >&2
  exit 1
fi
if [[ "${BUILD_ANDROID_LIB}" == "1" ]] && ! command -v ndk-build >/dev/null 2>&1; then
  echo "ndk-build not found. Either install Android NDK tools in PATH or omit --with-android-lib." >&2
  exit 1
fi

printf '[info] sdk root: %s\n' "${SDK_ROOT}"
printf '[info] output dir: %s\n' "${OUTPUT_DIR}"
printf '[info] backend: %s\n' "${BACKEND}"
printf '[info] building model libraries\n'

lib_targets=(x86_64-linux-clang)
if [[ "${BUILD_ANDROID_LIB}" == "1" ]]; then
  lib_targets+=(aarch64-android)
fi

"${SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
  -c "${MODEL_CPP}" \
  -b "${MODEL_BIN}" \
  -t "${lib_targets[@]}" \
  -o "${MODEL_LIBS_DIR}"

MODEL_SO="${MODEL_LIBS_DIR}/x86_64-linux-clang/lib${MODEL}.so"
if [[ ! -f "${MODEL_SO}" ]]; then
  echo "Generated host model library missing: ${MODEL_SO}" >&2
  exit 1
fi

printf '[info] generating context binaries\n'
"${SCRIPT_DIR}/generate_qnn_context_bins.sh" \
  --sdk-root "${SDK_ROOT}" \
  --backend "${BACKEND}" \
  --model-so "${MODEL_SO}" \
  --output-dir "${CTX_DIR}" \
  --binary-file "${MODEL}_${BACKEND}" \
  --backend-binary "${MODEL}_${BACKEND}_backend"

echo "[info] sample context pipeline complete"
echo "[info] host model so: ${MODEL_SO}"
echo "[info] context dir: ${CTX_DIR}"
