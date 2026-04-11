#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"
DEFAULT_ENV_LIB_DIR="${REPO_ROOT}/../OnDevAI_external/.conda-qairt310/lib"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
MODEL_SO=""
BACKEND="htp-v79"
OUTPUT_DIR="${REPO_ROOT}/.artifacts/qnn-context"
BINARY_FILE=""
BACKEND_BINARY=""
CONFIG_FILE=""
MODEL_PREFIX="QnnModel"
LOG_LEVEL="info"

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/generate_qnn_context_bins.sh [options]

Generates QNN context binaries using qnn-context-binary-generator.

Required:
  --model-so PATH             Path to generated model library (.so)

Optional:
  --backend cpu|gpu|htp-v79   Backend to target (default: htp-v79)
  --sdk-root PATH             QAIRT SDK root
  --output-dir PATH           Output directory for artifacts
  --binary-file NAME.bin      Output context binary name
  --backend-binary NAME.bin   Optional backend-specific binary name
  --config-file PATH          Optional backend extension JSON
  --model-prefix NAME         Model function prefix (default: QnnModel)
  --log-level error|warn|info|verbose (default: info)
  -h, --help                  Show help

Examples:
  scripts/sidecar_qnn/generate_qnn_context_bins.sh \
    --model-so /abs/path/libqnn_model_8bit_quantized.so \
    --backend htp-v79 \
    --binary-file sample_ctx.bin \
    --config-file /abs/path/htp_backend_ext_config.json
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-so)
      MODEL_SO="$2"
      shift 2
      ;;
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
    --binary-file)
      BINARY_FILE="$2"
      shift 2
      ;;
    --backend-binary)
      BACKEND_BINARY="$2"
      shift 2
      ;;
    --config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --model-prefix)
      MODEL_PREFIX="$2"
      shift 2
      ;;
    --log-level)
      LOG_LEVEL="$2"
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

if [[ -z "${MODEL_SO}" ]]; then
  echo "--model-so is required" >&2
  exit 1
fi
if [[ ! -f "${MODEL_SO}" ]]; then
  echo "Model shared library not found: ${MODEL_SO}" >&2
  exit 1
fi
if [[ ! -d "${SDK_ROOT}" ]]; then
  echo "SDK root not found: ${SDK_ROOT}" >&2
  exit 1
fi

backend_lib=""
case "${BACKEND}" in
  cpu)
    backend_lib="${SDK_ROOT}/lib/x86_64-linux-clang/libQnnCpu.so"
    ;;
  gpu)
    backend_lib="${SDK_ROOT}/lib/x86_64-linux-clang/libQnnGpu.so"
    ;;
  htp-v79)
    backend_lib="${SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"
    ;;
  *)
    echo "Unsupported backend: ${BACKEND} (expected cpu|gpu|htp-v79)" >&2
    exit 1
    ;;
esac

if [[ ! -f "${backend_lib}" ]]; then
  echo "Backend library not found: ${backend_lib}" >&2
  exit 1
fi

if [[ -n "${CONFIG_FILE}" && ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

binary_arg=""
backend_binary_arg=""
if [[ -n "${BINARY_FILE}" ]]; then
  binary_arg="${BINARY_FILE%.bin}"
  # Cleanup stale artifacts from older invocations that passed .bin and ended up with .bin.bin.
  rm -f "${OUTPUT_DIR}/${binary_arg}.bin.bin" || true
fi
if [[ -n "${BACKEND_BINARY}" ]]; then
  backend_binary_arg="${BACKEND_BINARY%.bin}"
  rm -f "${OUTPUT_DIR}/${backend_binary_arg}.bin.bin" || true
fi

cmd=(
  "${SDK_ROOT}/bin/x86_64-linux-clang/qnn-context-binary-generator"
  --model "${MODEL_SO}"
  --backend "${backend_lib}"
  --output_dir "${OUTPUT_DIR}"
  --model_prefix "${MODEL_PREFIX}"
  --log_level "${LOG_LEVEL}"
)

if [[ -n "${binary_arg}" ]]; then
  cmd+=(--binary_file "${binary_arg}")
fi
if [[ -n "${backend_binary_arg}" ]]; then
  cmd+=(--backend_binary "${backend_binary_arg}")
fi
if [[ -n "${CONFIG_FILE}" ]]; then
  cmd+=(--config_file "${CONFIG_FILE}")
fi

# QAIRT host tools may require libc++.so.1 from a local env.
extra_ld_path=""
if [[ -d "${DEFAULT_ENV_LIB_DIR}" ]]; then
  extra_ld_path="${DEFAULT_ENV_LIB_DIR}"
fi

set +u
source "${SDK_ROOT}/bin/envsetup.sh"
set -u

if [[ -n "${extra_ld_path}" ]]; then
  export LD_LIBRARY_PATH="${extra_ld_path}:${LD_LIBRARY_PATH:-}"
fi

printf '[info] sdk root: %s\n' "${SDK_ROOT}"
printf '[info] backend: %s\n' "${BACKEND}"
printf '[info] model so: %s\n' "${MODEL_SO}"
printf '[info] output dir: %s\n' "${OUTPUT_DIR}"
if [[ -n "${CONFIG_FILE}" ]]; then
  printf '[info] config: %s\n' "${CONFIG_FILE}"
fi

"${cmd[@]}"
if [[ -n "${binary_arg}" && -f "${OUTPUT_DIR}/${binary_arg}.bin" && -f "${OUTPUT_DIR}/${binary_arg}.bin.bin" ]]; then
  rm -f "${OUTPUT_DIR}/${binary_arg}.bin.bin" || true
fi
if [[ -n "${binary_arg}" ]]; then
  printf '[info] context binary: %s\n' "${OUTPUT_DIR}/${binary_arg}.bin"
fi
if [[ -n "${backend_binary_arg}" ]]; then
  printf '[info] backend binary: %s\n' "${OUTPUT_DIR}/${backend_binary_arg}.bin"
fi
echo "[info] done"
