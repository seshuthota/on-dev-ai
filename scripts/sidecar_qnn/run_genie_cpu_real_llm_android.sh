#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"
DEFAULT_MODEL_BIN="${REPO_ROOT}/.artifacts/qnn-genai/gpt2_124m_q4.bin"
DEFAULT_TOKENIZER_JSON="/home/curious/models/gpt2-local/tokenizer.json"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"
PROMPT="Write one short sentence about on-device AI."
MODEL_BIN="${DEFAULT_MODEL_BIN}"
TOKENIZER_JSON="${DEFAULT_TOKENIZER_JSON}"
OUT_DIR="/data/local/tmp/ondevai_genie_cpu_real_llm"
CONTEXT_SIZE="1024"
N_VOCAB="50257"
BOS_TOKEN="50256"
EOS_TOKEN="50256"

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/run_genie_cpu_real_llm_android.sh [options]

Runs the current proven Genie real-LLM path on Android.

Important:
  This is the current CPU-realized reference path.
  It uses QnnGenAiTransformer with a library-style model.bin flow.
  It does NOT prove HTP/NPU realization.

Options:
  --model-bin PATH            Host path to composer output .bin
  --tokenizer-json PATH       Host path to tokenizer.json
  --prompt TEXT               Prompt text
  --adb-serial SERIAL         adb serial (e.g. 192.168.29.11:41993)
  --sdk-root PATH             QAIRT SDK root
  --out-dir PATH              Device work directory
  --context-size INT          Context size (default: 1024)
  --n-vocab INT               Vocabulary size (default: 50257)
  --bos-token INT             BOS token id (default: 50256)
  --eos-token INT             EOS token id (default: 50256)
  -h, --help                  Show help

Example:
  scripts/sidecar_qnn/run_genie_cpu_real_llm_android.sh \
    --adb-serial 192.168.29.11:41993 \
    --model-bin /abs/path/model.bin \
    --tokenizer-json /abs/path/tokenizer.json
EOF
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
    --model-bin)
      MODEL_BIN="$2"
      shift 2
      ;;
    --tokenizer-json)
      TOKENIZER_JSON="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
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
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --context-size)
      CONTEXT_SIZE="$2"
      shift 2
      ;;
    --n-vocab)
      N_VOCAB="$2"
      shift 2
      ;;
    --bos-token)
      BOS_TOKEN="$2"
      shift 2
      ;;
    --eos-token)
      EOS_TOKEN="$2"
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

if [[ -z "${ADB_SERIAL}" ]]; then
  ADB_SERIAL="$(detect_adb_serial || true)"
fi
if [[ -z "${ADB_SERIAL}" ]]; then
  echo "No adb serial provided and auto-detection did not find exactly one connected device" >&2
  exit 1
fi
if ! adb -s "${ADB_SERIAL}" get-state >/dev/null 2>&1; then
  echo "adb device not available: ${ADB_SERIAL}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_BIN}" ]]; then
  echo "Model bin not found: ${MODEL_BIN}" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER_JSON}" ]]; then
  echo "Tokenizer JSON not found: ${TOKENIZER_JSON}" >&2
  exit 1
fi
if [[ ! -x "${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" ]]; then
  echo "Missing binary: ${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/ondevai_genie_cpu_real_llm.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

CONFIG_JSON="${TMP_DIR}/dialog.json"
cat >"${CONFIG_JSON}" <<EOF
{
  "dialog": {
    "version": 1,
    "type": "basic",
    "max-num-tokens": 32,
    "context": {
      "version": 1,
      "size": ${CONTEXT_SIZE},
      "n-vocab": ${N_VOCAB},
      "bos-token": ${BOS_TOKEN},
      "eos-token": ${EOS_TOKEN}
    },
    "sampler": {
      "version": 1,
      "seed": 7,
      "temp": 0.8,
      "top-k": 40,
      "top-p": 0.95,
      "greedy": false
    },
    "tokenizer": {
      "version": 1,
      "path": "tokenizer.json"
    },
    "engine": {
      "version": 1,
      "n-threads": 4,
      "backend": {
        "version": 1,
        "type": "QnnGenAiTransformer",
        "QnnGenAiTransformer": {
          "version": 1,
          "kv-quantization": false,
          "shared-engine": false
        }
      },
      "model": {
        "version": 1,
        "type": "library",
        "library": {
          "version": 1,
          "model-bin": "model.bin"
        }
      }
    }
  }
}
EOF

printf '[info] path_kind: genie_cpu_real_llm\n'
printf '[info] sdk root: %s\n' "${SDK_ROOT}"
printf '[info] adb serial: %s\n' "${ADB_SERIAL}"
printf '[info] target: %s\n' "${OUT_DIR}"
printf '[info] note: this path is expected to realize as QNN_CPU unless proven otherwise in logs.\n'

adb -s "${ADB_SERIAL}" shell "rm -rf '${OUT_DIR}' && mkdir -p '${OUT_DIR}'" >/dev/null
adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" "${OUT_DIR}/genie-t2t-run" >/dev/null
adb -s "${ADB_SERIAL}" push "${MODEL_BIN}" "${OUT_DIR}/model.bin" >/dev/null
adb -s "${ADB_SERIAL}" push "${TOKENIZER_JSON}" "${OUT_DIR}/tokenizer.json" >/dev/null
adb -s "${ADB_SERIAL}" push "${CONFIG_JSON}" "${OUT_DIR}/dialog.json" >/dev/null

for lib in \
  libGenie.so \
  libQnnSystem.so \
  libQnnGenAiTransformer.so \
  libQnnGenAiTransformerModel.so \
  libQnnHtp.so \
  libQnnHtpPrepare.so \
  libQnnHtpV79Stub.so
do
  adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/lib/aarch64-android/${lib}" "${OUT_DIR}/${lib}" >/dev/null
done

adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${OUT_DIR}/libQnnHtpV79Skel.so" >/dev/null

run_cmd="cd '${OUT_DIR}' && chmod +x ./genie-t2t-run && export LD_LIBRARY_PATH='${OUT_DIR}' && export ADSP_LIBRARY_PATH='${OUT_DIR};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./genie-t2t-run --config dialog.json --prompt \"${PROMPT}\" --log info"
adb -s "${ADB_SERIAL}" shell "${run_cmd}"
