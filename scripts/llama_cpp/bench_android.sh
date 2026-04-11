#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_CPP_RUNS_DIR="${REPO_ROOT}/.artifacts/llama-cpp/runs"

SERIAL=""
MODEL=""
MODEL_ID=""
QUANT="Q4_K_M"
THREADS=6
CPU_MASK="auto-big"
RUNS=3
CTX=512
MAX_NEW_TOKENS=64
INSTALL_DIR="${REPO_ROOT}/.artifacts/llama-cpp/android-install"
OUTPUT_JSONL="${REPO_ROOT}/files/benchmarks/llama_cpp_android.jsonl"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/bench_android.py"

usage() {
  cat <<'EOF'
Usage: scripts/llama_cpp/bench_android.sh [OPTIONS]

Run llama-bench on an Android device via adb for low-noise kernel-level profiling.

Required:
  --serial SERIAL       ADB device serial number
  --model PATH          Path to GGUF model file on host
  --model-id ID         Model identifier string

Options:
  --quant FORMAT        Quantization format (default: Q4_K_M)
  --threads N           Number of threads (default: 6)
  --cpu-mask HEX|auto-big  CPU affinity mask (default: auto-big)
  --runs N              Number of measured runs (default: 3)
  --ctx N               Context length (default: 512)
  --max-new-tokens N    Max new tokens to generate (default: 64)
  --install-dir PATH    Install directory (default: .artifacts/llama-cpp/android-install)
  --output-jsonl PATH   Output JSONL file (default: files/benchmarks/llama_cpp_android.jsonl)
  -h, --help           Show this help message

Examples:
  scripts/llama_cpp/bench_android.sh \
    --serial 1234567890 \
    --model /path/to/model.Q4_K_M.gguf \
    --model-id Qwen3.5-4B-Q4_K_M \
    --threads 6 \
    --cpu-mask auto-big
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)
      SERIAL="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --cpu-mask)
      CPU_MASK="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --ctx)
      CTX="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --install-dir)
      INSTALL_DIR="$2"
      shift 2
      ;;
    --output-jsonl)
      OUTPUT_JSONL="$2"
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

if [[ -z "${SERIAL}" ]]; then
  echo "Error: --serial is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  echo "Error: --model is required" >&2
  usage >&2
  exit 1
fi

if [[ -z "${MODEL_ID}" ]]; then
  echo "Error: --model-id is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "Error: Model file not found: ${MODEL}" >&2
  exit 1
fi

if [[ ! -f "${INSTALL_DIR}/bin/llama-bench" ]]; then
  echo "Error: llama-bench not found: ${INSTALL_DIR}/bin/llama-bench" >&2
  echo "Did you run build_android.sh?" >&2
  exit 1
fi

DEVICE_MODEL="$(adb -s "${SERIAL}" shell getprop ro.product.model 2>/dev/null | tr -d '\r' | tr -d ' ' || echo "unknown")"

TARGET_DIR="/data/local/tmp/ondevai-llama-cpp"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LLAMA_CPP_RUNS_DIR}/${RUN_TIMESTAMP}"
mkdir -p "${RUN_DIR}"

MODEL_CHECKSUM="$(sha256sum "${MODEL}" | awk '{print $1}')"

echo "[info] Device: ${DEVICE_MODEL} (${SERIAL})"
echo "[info] Model: ${MODEL}"
echo "[info] Model ID: ${MODEL_ID}"
echo "[info] Quant: ${QUANT}"
echo "[info] Threads: ${THREADS}"
echo "[info] CPU mask: ${CPU_MASK}"
echo "[info] Runs: ${RUNS}"
echo "[info] Ctx: ${CTX}, max-new-tokens: ${MAX_NEW_TOKENS}"
echo "[info] Run dir: ${RUN_DIR}"

# Push model if not already there with matching checksum
DEVICE_CHECKSUM_FILE="${TARGET_DIR}/model.sha256"
DEVICE_MODEL_FILE="${TARGET_DIR}/model.gguf"

if [[ -f "${DEVICE_CHECKSUM_FILE}" ]]; then
  DEVICE_CHECKSUM="$(adb -s "${SERIAL}" shell "cat '${DEVICE_CHECKSUM_FILE}'" 2>/dev/null | tr -d '\r' | tr -d ' \n')"
else
  DEVICE_CHECKSUM=""
fi

if [[ "${DEVICE_CHECKSUM}" != "${MODEL_CHECKSUM}" || ! -f "${DEVICE_MODEL_FILE}" ]]; then
  echo "[info] Pushing model to device..."
  adb -s "${SERIAL}" push "${MODEL}" "${DEVICE_MODEL_FILE}" >/dev/null
  echo -n "${MODEL_CHECKSUM}" | adb -s "${SERIAL}" shell "cat > '${DEVICE_CHECKSUM_FILE}'"
  echo "[info] Model checksum: ${MODEL_CHECKSUM}"
else
  echo "[info] Device model matches (checksum verified), skipping push."
fi

resolve_cpu_mask() {
  local mask="$1"
  local cpu_mask=""
  local cpu_strict="1"

  if [[ "${mask}" == "auto-big" ]]; then
    echo "[info] Detecting big cores via max frequency..." >&2
    local max_freq_files
    mapfile -t max_freq_files < <(adb -s "${SERIAL}" shell 'find /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq -type f 2>/dev/null' | tr -d '\r' || true)

    if [[ ${#max_freq_files[@]} -eq 0 ]]; then
      echo "[warn] Could not detect CPU frequencies, using 0x0 mask" >&2
      echo "0x0"
      return
    fi

    declare -A freq_map
    for f in "${max_freq_files[@]}"; do
      local freq
      freq="$(adb -s "${SERIAL}" shell "cat '${f}' 2>/dev/null" | tr -d '\r' || echo "0")"
      if [[ -n "${freq}" && "${freq}" =~ ^[0-9]+$ ]]; then
        freq_map["${freq}"]=1
      fi
    done

    if [[ ${#freq_map[@]} -eq 0 ]]; then
      echo "[warn] Could not parse CPU frequencies, using 0x0 mask" >&2
      echo "0x0"
      return
    fi

    local max_freqs=($(for k in "${!freq_map[@]}"; do echo "$k"; done | sort -n | tail -n 5))
    local threshold=$((max_freqs[-1] * 80 / 100))

    local mask_int=0
    for i in $(seq 0 7); do
      local freq_file="/sys/devices/system/cpu/cpu${i}/cpufreq/cpuinfo_max_freq"
      local freq
      freq="$(adb -s "${SERIAL}" shell "cat '${freq_file}' 2>/dev/null" | tr -d '\r' || echo "0")"
      if [[ -n "${freq}" && "${freq}" =~ ^[0-9]+$ && ${freq} -ge ${threshold} ]]; then
        mask_int=$((mask_int | (1 << i)))
      fi
    done

    cpu_mask=$(printf '0x%x' ${mask_int})
    echo "[info] Detected big cores mask: ${cpu_mask}" >&2
    echo "${cpu_mask}"
  else
    cpu_mask="${mask}"
    echo "${cpu_mask}"
  fi
}

CPU_MASK_RESOLVED="$(resolve_cpu_mask "${CPU_MASK}")"

LLAMA_BENCH="./bin/llama-bench"
DEVICE_MODEL_PATH="${TARGET_DIR}/model.gguf"

echo "[info] Running llama-bench..."

adb -s "${SERIAL}" shell \
  "cd '${TARGET_DIR}/install' && \
   LD_LIBRARY_PATH=lib ${LLAMA_BENCH} \
     -m '${DEVICE_MODEL_PATH}' \
     -o json \
     -r ${RUNS} \
     -t ${THREADS} \
     -C ${CPU_MASK_RESOLVED} \
     --no-warmup \
     -p ${CTX} \
     -ngl 0 \
     -pg 0,${MAX_NEW_TOKENS}" \
  > "${RUN_DIR}/llama_bench_output.json" 2>&1

LLAMA_CPP_COMMIT="$(git -C "${REPO_ROOT}/native/third_party/llama.cpp" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

echo "[info] Parsing results..."

python3 "${BENCHMARK_SCRIPT}" \
  --run-dir "${RUN_DIR}" \
  --model-id "${MODEL_ID}" \
  --model-checksum "${MODEL_CHECKSUM}" \
  --quant-format "${QUANT}" \
  --device-serial "${SERIAL}" \
  --device-model "${DEVICE_MODEL}" \
  --threads "${THREADS}" \
  --cpu-mask "${CPU_MASK_RESOLVED}" \
  --run-count "${RUNS}" \
  --ctx "${CTX}" \
  --llama-cpp-commit "${LLAMA_CPP_COMMIT}" \
  --model-path "${DEVICE_MODEL_PATH}" \
  --output-jsonl "${OUTPUT_JSONL}"

echo "[info] Benchmark complete. Results in: ${OUTPUT_JSONL}"
echo "[info] Run artifacts: ${RUN_DIR}"
