#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROMPTS_FILE="${REPO_ROOT}/docs/custom_runtime_benchmark_prompts.jsonl"
LLAMA_CPP_RUNS_DIR="${REPO_ROOT}/.artifacts/llama-cpp/runs"

SERIAL=""
MODEL=""
MODEL_ID=""
QUANT="Q4_K_M"
THREADS=6
CPU_MASK="auto-big"
RUNS=5
WARMUP=1
CTX=512
MAX_NEW_TOKENS=64
INSTALL_DIR="${REPO_ROOT}/.artifacts/llama-cpp/android-install"
OUTPUT_JSONL="${REPO_ROOT}/files/benchmarks/llama_cpp_android.jsonl"

usage() {
  cat <<'EOF'
Usage: scripts/llama_cpp/benchmark_android.sh [OPTIONS]

Run llama.cpp benchmarks on an Android device via adb.

Required:
  --serial SERIAL       ADB device serial number
  --model PATH          Path to GGUF model file on host
  --model-id ID         Model identifier string

Options:
  --quant FORMAT        Quantization format (default: Q4_K_M)
  --threads N           Number of threads (default: 6)
  --cpu-mask HEX|auto-big  CPU affinity mask (default: auto-big)
  --runs N              Number of measured runs (default: 5)
  --warmup N            Number of warmup runs (default: 1)
  --ctx N               Context length (default: 512)
  --max-new-tokens N    Max new tokens to generate (default: 64)
  --install-dir PATH    Install directory (default: .artifacts/llama-cpp/android-install)
  --output-jsonl PATH   Output JSONL file (default: files/benchmarks/llama_cpp_android.jsonl)
  -h, --help            Show this help message

CPU Mask:
  auto-big: Detect max-frequency cores and create hex mask
  HEX:      Direct hex mask value (e.g., 0xf)
  If auto-big detection fails, defaults to 0x0 with cpu_strict=0

Examples:
  scripts/llama_cpp/benchmark_android.sh \
    --serial 1234567890 \
    --model /path/to/model.Q4_K_M.gguf \
    --model-id Qwen3.5-4B-Q4_K_M \
    --threads 6
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
    --warmup)
      WARMUP="$2"
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

if [[ ! -f "${INSTALL_DIR}/bin/llama-cli" ]]; then
  echo "Error: llama-cli not found in install directory: ${INSTALL_DIR}/bin/llama-cli" >&2
  echo "Did you run build_android.sh first?" >&2
  exit 1
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "Error: Prompts file not found: ${PROMPTS_FILE}" >&2
  exit 1
fi

if ! adb -s "${SERIAL}" get-state >/dev/null 2>&1; then
  echo "Error: adb device not available: ${SERIAL}" >&2
  exit 1
fi

DEVICE_MODEL="$(adb -s "${SERIAL}" shell getprop ro.product.model 2>/dev/null | tr -d '\r' | tr -d ' ' || echo "unknown")"
DEVICE_SERIAL="${SERIAL}"

TARGET_DIR="/data/local/tmp/ondevai-llama-cpp"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LLAMA_CPP_RUNS_DIR}/${RUN_TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "[info] Device: ${DEVICE_MODEL} (${DEVICE_SERIAL})"
echo "[info] Model: ${MODEL}"
echo "[info] Model ID: ${MODEL_ID}"
echo "[info] Quant: ${QUANT}"
echo "[info] Threads: ${THREADS}"
echo "[info] CPU mask: ${CPU_MASK}"
echo "[info] Runs: ${RUNS} warmup: ${WARMUP}"
echo "[info] Ctx: ${CTX}, max-new-tokens: ${MAX_NEW_TOKENS}"
echo "[info] Run dir: ${RUN_DIR}"
echo "[info] Output: ${OUTPUT_JSONL}"

adb -s "${SERIAL}" shell "rm -rf '${TARGET_DIR}' && mkdir -p '${TARGET_DIR}'" >/dev/null

echo "[info] Pushing install directory to device..."
adb -s "${SERIAL}" push "${INSTALL_DIR}" "${TARGET_DIR}/install" >/dev/null

echo "[info] Pushing model to device..."
adb -s "${SERIAL}" push "${MODEL}" "${TARGET_DIR}/model.gguf" >/dev/null

MODEL_CHECKSUM="$(sha256sum "${MODEL}" | awk '{print $1}')"

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

if [[ "${CPU_MASK_RESOLVED}" == "0x0" ]]; then
  cpu_strict="0"
else
  cpu_strict="1"
fi

LLAMA_CLI="./bin/llama-cli"
MODEL_PATH="${TARGET_DIR}/model.gguf"

NORMALIZE_SCRIPT="${SCRIPT_DIR}/normalize_llama_cli.py"

echo "[info] Running benchmarks..."

DEVICE_PROMPT_DIR="${TARGET_DIR}/prompts"
adb -s "${SERIAL}" shell "mkdir -p '${DEVICE_PROMPT_DIR}'" >/dev/null

LLAMA_CPP_COMMIT="$(git -C "${REPO_ROOT}/native/third_party/llama.cpp" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

DEVICE_MODEL_PATH="${TARGET_DIR}/model.gguf"

while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line}" ]] && continue

  prompt_id="$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('prompt_id',''))" 2>/dev/null)"
  prompt_text="$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('prompt',''))" 2>/dev/null)"
  prompt_max_new="$(echo "$line" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('max_new_tokens',${MAX_NEW_TOKENS}))" 2>/dev/null)"

  if [[ -z "${prompt_id}" || -z "${prompt_text}" ]]; then
    echo "[warn] Skipping invalid prompt line" >&2
    continue
  fi

  host_prompt_file="${RUN_DIR}/${prompt_id}_prompt.txt"
  printf '%s' "${prompt_text}" > "${host_prompt_file}"
  device_prompt_file="${DEVICE_PROMPT_DIR}/${prompt_id}_prompt.txt"
  adb -s "${SERIAL}" push "${host_prompt_file}" "${device_prompt_file}" >/dev/null 2>&1

  echo "[info] Benchmarking prompt_id=${prompt_id}"

  for run_idx in $(seq 1 "${WARMUP}"); do
    log_file="${RUN_DIR}/${prompt_id}_warmup_${run_idx}.log"

    set +e
    if [[ "${CPU_MASK_RESOLVED}" != "0x0" ]]; then
      adb -s "${SERIAL}" shell \
        "cd '${TARGET_DIR}/install' && \
         LD_LIBRARY_PATH=lib ${LLAMA_CLI} \
           -m '${MODEL_PATH}' \
           -c '${CTX}' \
           -n '${prompt_max_new}' \
           -t '${THREADS}' \
           -f '${device_prompt_file}' \
           --seed 0 \
           --temp 0 \
           -C '${CPU_MASK_RESOLVED}' \
           --cpu-strict 1" \
        < /dev/null > "${log_file}" 2>&1
    else
      adb -s "${SERIAL}" shell \
        "cd '${TARGET_DIR}/install' && \
         LD_LIBRARY_PATH=lib ${LLAMA_CLI} \
           -m '${MODEL_PATH}' \
           -c '${CTX}' \
           -n '${prompt_max_new}' \
           -t '${THREADS}' \
           -f '${device_prompt_file}' \
           --seed 0 \
           --temp 0" \
        < /dev/null > "${log_file}" 2>&1
    fi
    run_rc=$?
    set -e

    if [[ ${run_rc} -ne 0 ]]; then
      echo "[warn] Warmup run failed with rc=${run_rc}: ${log_file}" >&2
    fi
  done

  for run_idx in $(seq 1 "${RUNS}"); do
    log_file="${RUN_DIR}/${prompt_id}_measured_${run_idx}.log"

    set +e
    if [[ "${CPU_MASK_RESOLVED}" != "0x0" ]]; then
      adb -s "${SERIAL}" shell \
        "cd '${TARGET_DIR}/install' && \
         LD_LIBRARY_PATH=lib ${LLAMA_CLI} \
           -m '${MODEL_PATH}' \
           -c '${CTX}' \
           -n '${prompt_max_new}' \
           -t '${THREADS}' \
           -f '${device_prompt_file}' \
           --seed 0 \
           --temp 0 \
           -C '${CPU_MASK_RESOLVED}' \
           --cpu-strict 1" \
        < /dev/null > "${log_file}" 2>&1
    else
      adb -s "${SERIAL}" shell \
        "cd '${TARGET_DIR}/install' && \
         LD_LIBRARY_PATH=lib ${LLAMA_CLI} \
           -m '${MODEL_PATH}' \
           -c '${CTX}' \
           -n '${prompt_max_new}' \
           -t '${THREADS}' \
           -f '${device_prompt_file}' \
           --seed 0 \
           --temp 0" \
        < /dev/null > "${log_file}" 2>&1
    fi
    run_rc=$?
    set -e

    if [[ ${run_rc} -ne 0 ]]; then
      echo "[warn] Measured run failed with rc=${run_rc}: ${log_file}" >&2
    fi
  done

  echo "${prompt_id}" >> "${RUN_DIR}/completed_prompts.txt"

done < "${PROMPTS_FILE}"

echo "[info] Generating normalized output..."

python3 "${NORMALIZE_SCRIPT}" \
  --run-dir "${RUN_DIR}" \
  --model-id "${MODEL_ID}" \
  --model-checksum "${MODEL_CHECKSUM}" \
  --quant-format "${QUANT}" \
  --device-serial "${DEVICE_SERIAL}" \
  --device-model "${DEVICE_MODEL}" \
  --threads "${THREADS}" \
  --cpu-mask "${CPU_MASK_RESOLVED}" \
  --warmup-count "${WARMUP}" \
  --run-count "${RUNS}" \
  --ctx "${CTX}" \
  --install-dir "${INSTALL_DIR}" \
  --llama-cpp-commit "${LLAMA_CPP_COMMIT}" \
  --model-path "${DEVICE_MODEL_PATH}" \
  --output-jsonl "${OUTPUT_JSONL}"

echo "[info] Benchmark complete. Results in: ${OUTPUT_JSONL}"
echo "[info] Run artifacts: ${RUN_DIR}"
