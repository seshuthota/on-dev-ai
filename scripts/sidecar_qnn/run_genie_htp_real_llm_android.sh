#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"
PROMPT="Write one short sentence about on-device AI."
TOKENIZER_JSON=""
OUT_DIR="/data/local/tmp/ondevai_genie_htp_real_llm"
CONTEXT_SIZE="1024"
N_VOCAB=""
BOS_TOKEN=""
EOS_TOKEN=""
Htp_EXT_SOC_ID="57"
declare -a CTX_BINS=()

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh [options]

Runs the HTP-targeted Genie real-LLM path on Android.

Important:
  This script expects already-generated HTP context binaries.
  It is the correct target shape for proving real HTP realization:
  QnnHtp backend + model.type=binary + ctx-bins.

Required options:
  --tokenizer-json PATH       Host path to tokenizer.json
  --ctx-bin PATH              Host path to one context binary; repeat for multiple bins
  --n-vocab INT               Vocabulary size
  --bos-token INT             BOS token id
  --eos-token INT             EOS token id

Optional:
  --prompt TEXT               Prompt text
  --adb-serial SERIAL         adb serial (e.g. 192.168.29.11:41993)
  --sdk-root PATH             QAIRT SDK root
  --out-dir PATH              Device work directory
  --context-size INT          Context size (default: 1024)
  -h, --help                  Show help

Example:
  scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh \
    --adb-serial 192.168.29.11:41993 \
    --tokenizer-json /abs/path/tokenizer.json \
    --ctx-bin /abs/path/file_1_of_4.bin \
    --ctx-bin /abs/path/file_2_of_4.bin \
    --ctx-bin /abs/path/file_3_of_4.bin \
    --ctx-bin /abs/path/file_4_of_4.bin \
    --n-vocab 32000 --bos-token 1 --eos-token 2
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
    --tokenizer-json)
      TOKENIZER_JSON="$2"
      shift 2
      ;;
    --ctx-bin)
      CTX_BINS+=("$2")
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

if [[ -z "${TOKENIZER_JSON}" ]]; then
  echo "--tokenizer-json is required" >&2
  exit 1
fi
if [[ "${#CTX_BINS[@]}" -eq 0 ]]; then
  echo "At least one --ctx-bin is required" >&2
  exit 1
fi
if [[ -z "${N_VOCAB}" || -z "${BOS_TOKEN}" || -z "${EOS_TOKEN}" ]]; then
  echo "--n-vocab, --bos-token, and --eos-token are required" >&2
  exit 1
fi

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
if [[ ! -f "${TOKENIZER_JSON}" ]]; then
  echo "Tokenizer JSON not found: ${TOKENIZER_JSON}" >&2
  exit 1
fi
if [[ ! -x "${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" ]]; then
  echo "Missing binary: ${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" >&2
  exit 1
fi
for ctx_bin in "${CTX_BINS[@]}"; do
  if [[ ! -f "${ctx_bin}" ]]; then
    echo "Context binary not found: ${ctx_bin}" >&2
    exit 1
  fi
done

TMP_DIR="$(mktemp -d /tmp/ondevai_genie_htp_real_llm.XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

HTP_EXT_JSON="${TMP_DIR}/htp_backend_ext_config.json"
cat >"${HTP_EXT_JSON}" <<EOF
{
  "devices": [
    {
      "soc_id": ${Htp_EXT_SOC_ID},
      "dsp_arch": "v79",
      "cores": [
        {
          "core_id": 0,
          "perf_profile": "burst"
        }
      ]
    }
  ]
}
EOF

ctx_bins_json=""
for ctx_bin in "${CTX_BINS[@]}"; do
  ctx_name="$(basename "${ctx_bin}")"
  if [[ -n "${ctx_bins_json}" ]]; then
    ctx_bins_json+=", "
  fi
  ctx_bins_json+="\"${ctx_name}\""
done

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
      "n-threads": 3,
      "backend": {
        "version": 1,
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "spill-fill-bufsize": 320000000,
          "use-mmap": true,
          "mmap-budget": 0,
          "poll": true,
          "allow-async-init": false,
          "enable-graph-switching": false
        },
        "extensions": "htp_backend_ext_config.json"
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": [ ${ctx_bins_json} ]
        }
      }
    }
  }
}
EOF

printf '[info] path_kind: genie_htp_real_llm\n'
printf '[info] sdk root: %s\n' "${SDK_ROOT}"
printf '[info] adb serial: %s\n' "${ADB_SERIAL}"
printf '[info] target: %s\n' "${OUT_DIR}"
printf '[info] ctx_bin_count: %s\n' "${#CTX_BINS[@]}"
printf '[info] note: this is the HTP-targeted path. Confirm realization in logs; do not assume success from launch alone.\n'

adb -s "${ADB_SERIAL}" shell "rm -rf '${OUT_DIR}' && mkdir -p '${OUT_DIR}'" >/dev/null
adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/bin/aarch64-android/genie-t2t-run" "${OUT_DIR}/genie-t2t-run" >/dev/null
adb -s "${ADB_SERIAL}" push "${TOKENIZER_JSON}" "${OUT_DIR}/tokenizer.json" >/dev/null
adb -s "${ADB_SERIAL}" push "${CONFIG_JSON}" "${OUT_DIR}/dialog.json" >/dev/null
adb -s "${ADB_SERIAL}" push "${HTP_EXT_JSON}" "${OUT_DIR}/htp_backend_ext_config.json" >/dev/null

for ctx_bin in "${CTX_BINS[@]}"; do
  adb -s "${ADB_SERIAL}" push "${ctx_bin}" "${OUT_DIR}/$(basename "${ctx_bin}")" >/dev/null
done

for lib in \
  libGenie.so \
  libQnnSystem.so \
  libQnnHtp.so \
  libQnnHtpPrepare.so \
  libQnnHtpV79Stub.so
do
  adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/lib/aarch64-android/${lib}" "${OUT_DIR}/${lib}" >/dev/null
done

adb -s "${ADB_SERIAL}" push "${SDK_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${OUT_DIR}/libQnnHtpV79Skel.so" >/dev/null

run_cmd="cd '${OUT_DIR}' && chmod +x ./genie-t2t-run && export LD_LIBRARY_PATH='${OUT_DIR}' && export ADSP_LIBRARY_PATH='${OUT_DIR};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp' && ./genie-t2t-run --config dialog.json --prompt \"${PROMPT}\" --log info"
adb -s "${ADB_SERIAL}" shell "${run_cmd}"
