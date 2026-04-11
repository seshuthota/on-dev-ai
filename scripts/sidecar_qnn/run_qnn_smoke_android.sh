#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"
BACKEND="cpu"
BUILD_DIR="/tmp/ondevai_qnn_smoke_build"
HOST_OUTPUT_DIR=""
KEEP_TMP=0

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/run_qnn_smoke_android.sh [--backend cpu|gpu|htp-v79|htp-v81] [--adb-serial SERIAL] [--sdk-root PATH] [--build-dir PATH] [--output-dir PATH]

Builds and runs the standalone QNN smoke harness on an adb-connected Android device.
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

detect_adb_serial() {
  local devices
  mapfile -t devices < <(adb devices | awk '$2 == "device" {print $1}')
  if [[ "${#devices[@]}" -eq 1 ]]; then
    printf '%s\n' "${devices[0]}"
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
    --adb-serial)
      ADB_SERIAL="$2"
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --output-dir)
      HOST_OUTPUT_DIR="$2"
      shift 2
      ;;
    --keep-tmp)
      KEEP_TMP=1
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

case "${BACKEND}" in
  cpu|gpu|htp-v79|htp-v81) ;;
  *)
    echo "Unsupported backend: ${BACKEND}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" ]]; then
  echo "qnn-model-lib-generator missing under SDK: ${SDK_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${SDK_ROOT}/bin/aarch64-android/qnn-net-run" ]]; then
  echo "qnn-net-run missing under SDK: ${SDK_ROOT}" >&2
  exit 1
fi

if [[ -z "${ADB_SERIAL}" ]]; then
  ADB_SERIAL="$(detect_adb_serial || true)"
fi
if [[ -z "${ADB_SERIAL}" ]]; then
  echo "No adb serial provided and auto-detection did not find exactly one device" >&2
  exit 1
fi

if ! adb -s "${ADB_SERIAL}" get-state >/dev/null 2>&1; then
  echo "adb device not available: ${ADB_SERIAL}" >&2
  exit 1
fi

NDK_ROOT="$(detect_ndk_root || true)"
if [[ -z "${NDK_ROOT}" ]]; then
  echo "Could not detect Android NDK root." >&2
  exit 1
fi

CXX_SHARED="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
if [[ ! -f "${CXX_SHARED}" ]]; then
  echo "libc++_shared.so not found in NDK: ${CXX_SHARED}" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  if [[ "${KEEP_TMP}" -ne 1 ]]; then
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT

MODEL_ROOT="${SDK_ROOT}/examples/QNN/converter/models"
INPUT_LIST="${MODEL_ROOT}/input_list_float.txt"
INPUT_DATA_DIR="${MODEL_ROOT}/input_data_float"

MODEL_NAME="qnn_model_float"
BACKEND_LIB="libQnnCpu.so"
PROBE_LIBS=("libQnnSystem.so" "libQnnCpu.so")
RUNTIME_LIBS=("libQnnSystem.so" "libQnnCpu.so")
HTP_STUB=""
HTP_SKEL=""

case "${BACKEND}" in
  cpu)
    MODEL_NAME="qnn_model_float"
    BACKEND_LIB="libQnnCpu.so"
    PROBE_LIBS=("libQnnSystem.so" "libQnnCpu.so")
    RUNTIME_LIBS=("libQnnSystem.so" "libQnnCpu.so")
    ;;
  gpu)
    MODEL_NAME="qnn_model_float"
    BACKEND_LIB="libQnnGpu.so"
    PROBE_LIBS=("libQnnSystem.so" "libQnnGpu.so")
    RUNTIME_LIBS=("libQnnSystem.so" "libQnnGpu.so")
    ;;
  htp-v79)
    MODEL_NAME="qnn_model_8bit_quantized"
    BACKEND_LIB="libQnnHtp.so"
    PROBE_LIBS=("libQnnSystem.so" "libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnHtpV79Stub.so")
    RUNTIME_LIBS=("libQnnSystem.so" "libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnHtpV79Stub.so")
    HTP_STUB="libQnnHtpV79Stub.so"
    HTP_SKEL="${SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"
    ;;
  htp-v81)
    MODEL_NAME="qnn_model_8bit_quantized"
    BACKEND_LIB="libQnnHtp.so"
    PROBE_LIBS=("libQnnSystem.so" "libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnHtpV81Stub.so")
    RUNTIME_LIBS=("libQnnSystem.so" "libQnnHtp.so" "libQnnHtpPrepare.so" "libQnnHtpV81Stub.so")
    HTP_STUB="libQnnHtpV81Stub.so"
    HTP_SKEL="${SDK_ROOT}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
    ;;
esac

MODEL_CPP="${MODEL_ROOT}/${MODEL_NAME}.cpp"
MODEL_BIN="${MODEL_ROOT}/${MODEL_NAME}.bin"
MODEL_LIB_OUT="${TMP_DIR}/model_libs"
mkdir -p "${MODEL_LIB_OUT}"

echo "[info] sdk root: ${SDK_ROOT}"
echo "[info] ndk root: ${NDK_ROOT}"
echo "[info] adb serial: ${ADB_SERIAL}"
echo "[info] backend: ${BACKEND}"
echo "[info] model: ${MODEL_NAME}"

(
  set +u
  # shellcheck disable=SC1091
  source "${SDK_ROOT}/bin/envsetup.sh"
  set -u
  export PATH="${NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${NDK_ROOT}:${PATH}"
  "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" \
    -c "${MODEL_CPP}" \
    -b "${MODEL_BIN}" \
    -t aarch64-android \
    -l "${MODEL_NAME}" \
    -o "${MODEL_LIB_OUT}"
)

MODEL_LIB_SO="${MODEL_LIB_OUT}/aarch64-android/lib${MODEL_NAME}.so"
if [[ ! -f "${MODEL_LIB_SO}" ]]; then
  echo "Generated model library missing: ${MODEL_LIB_SO}" >&2
  exit 1
fi

"${SCRIPT_DIR}/build_qnn_smoke.sh" --build-dir "${BUILD_DIR}" >/dev/null
QNN_SMOKE_BIN="${BUILD_DIR}/qnn_smoke"
if [[ ! -f "${QNN_SMOKE_BIN}" ]]; then
  echo "qnn_smoke binary missing after build: ${QNN_SMOKE_BIN}" >&2
  exit 1
fi

TARGET_ROOT="/data/local/tmp/ondevai_qnn_smoke/${BACKEND}"
adb -s "${ADB_SERIAL}" shell "rm -rf '${TARGET_ROOT}' && mkdir -p '${TARGET_ROOT}'" >/dev/null

adb -s "${ADB_SERIAL}" push "${QNN_SMOKE_BIN}" "${TARGET_ROOT}/qnn_smoke" >/dev/null
adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/bin/aarch64-android/qnn-net-run" "${TARGET_ROOT}/qnn-net-run" >/dev/null
adb -s "${ADB_SERIAL}" push "${CXX_SHARED}" "${TARGET_ROOT}/libc++_shared.so" >/dev/null
adb -s "${ADB_SERIAL}" push "${MODEL_LIB_SO}" "${TARGET_ROOT}/lib${MODEL_NAME}.so" >/dev/null
adb -s "${ADB_SERIAL}" push "${INPUT_LIST}" "${TARGET_ROOT}/input_list_float.txt" >/dev/null
adb -s "${ADB_SERIAL}" push "${INPUT_DATA_DIR}" "${TARGET_ROOT}/input_data_float" >/dev/null

for lib in "${RUNTIME_LIBS[@]}"; do
  adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/lib/aarch64-android/${lib}" "${TARGET_ROOT}/${lib}" >/dev/null
done

if [[ -n "${HTP_SKEL}" ]]; then
  if [[ ! -f "${HTP_SKEL}" ]]; then
    echo "HTP skel missing: ${HTP_SKEL}" >&2
    exit 1
  fi
  adb -s "${ADB_SERIAL}" push "${HTP_SKEL}" "${TARGET_ROOT}/$(basename "${HTP_SKEL}")" >/dev/null
fi

PROBE_ARGS=()
for probe in "${PROBE_LIBS[@]}"; do
  PROBE_ARGS+=(--probe-lib "./${probe}")
done

RUN_ARGS=(
  ./qnn_smoke
  "${PROBE_ARGS[@]}"
  --qnn-net-run ./qnn-net-run
  --backend "./${BACKEND_LIB}"
  --model "./lib${MODEL_NAME}.so"
  --input-list ./input_list_float.txt
  --output-dir ./output
)

run_cmd_escaped="$(printf '%q ' "${RUN_ARGS[@]}")"

DEVICE_ENV="export LD_LIBRARY_PATH='${TARGET_ROOT}':\$LD_LIBRARY_PATH"
if [[ "${BACKEND}" == htp-v79 || "${BACKEND}" == htp-v81 ]]; then
  DEVICE_ENV="export ADSP_LIBRARY_PATH='${TARGET_ROOT}:/vendor/dsp/cdsp:/vendor/lib/rfsa/adsp:/system/lib/rfsa/adsp:/dsp' && ${DEVICE_ENV}"
fi

LOG_FILE="${TMP_DIR}/qnn_smoke_${BACKEND}.log"
set -x
set +e
adb -s "${ADB_SERIAL}" shell "cd '${TARGET_ROOT}' && chmod +x ./qnn_smoke ./qnn-net-run && ${DEVICE_ENV} && ${run_cmd_escaped}" 2>&1 | tee "${LOG_FILE}"
run_rc=${PIPESTATUS[0]}
set -e
set +x

if [[ -z "${HOST_OUTPUT_DIR}" ]]; then
  HOST_OUTPUT_DIR="${TMP_DIR}/output_${BACKEND}"
fi
mkdir -p "${HOST_OUTPUT_DIR}"
adb -s "${ADB_SERIAL}" pull "${TARGET_ROOT}/output" "${HOST_OUTPUT_DIR}" >/dev/null || true

RESULT_LINE="$(rg '^qnn_smoke_result ' "${LOG_FILE}" | tail -n 1 || true)"
if [[ -n "${RESULT_LINE}" ]]; then
  echo "${RESULT_LINE}"
else
  echo "qnn_smoke_result status=unknown"
fi
echo "log_file=${LOG_FILE}"
echo "host_output_dir=${HOST_OUTPUT_DIR}"

if [[ "${run_rc}" -ne 0 ]]; then
  exit "${run_rc}"
fi
