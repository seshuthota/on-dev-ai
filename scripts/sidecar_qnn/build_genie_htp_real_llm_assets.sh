#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_SDK_ROOT="${REPO_ROOT}/../OnDevAI_external/v2.45.0.260326/qairt/2.45.0.260326"

SDK_ROOT="${QAIRT_SDK_ROOT:-${QNN_SDK_ROOT:-${DEFAULT_SDK_ROOT}}}"
PYTHON_BIN="${ONDEVAI_QAIRT_PYTHON:-${REPO_ROOT}/../OnDevAI_external/.conda-qairt310/bin/python}"
MODEL_INPUT=""
CONTAINER_DIR="${REPO_ROOT}/.artifacts/genie-htp-real-llm/container"
CACHE_ROOT="${REPO_ROOT}/.artifacts/genie-htp-real-llm/cache"
QAIRT_TMP_DIR="/tmp/ondevai_qairt_tmp"
CHIPSET="SM8750"
EMBEDDING_LUT_MODE="auto"
TOKENIZER_PATH=""
CONFIG_PATH=""
CALIBRATION_INPUT_LIST=""
INTAKE_REPORT_PATH=""
ONNX_PREFLIGHT_MODE="auto"
ONNX_PREFLIGHT_BATCH_SIZE="1"
ONNX_PREFLIGHT_PROMPT_LENGTH="1"
ONNX_PREFLIGHT_CONTEXT_LENGTH="2048"
ONNX_PREFLIGHT_PAST_SEQ_LEN="-1"
IGNORE_INTAKE_BLOCKERS="0"
CLEAN="0"
RUN_ANDROID="0"
ADB_SERIAL="${ONDEVAI_QNN_ADB_SERIAL:-}"
PROMPT="Write one short sentence about on-device AI."
OUT_DIR="/data/local/tmp/ondevai_genie_htp_real_llm"
OVERRIDE_TOKENIZER_JSON=""
OVERRIDE_N_VOCAB=""
OVERRIDE_BOS_TOKEN=""
OVERRIDE_EOS_TOKEN=""

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/build_genie_htp_real_llm_assets.sh [options]

Builds a QAIRT GenAI HTP container + runtime manifest from model artifacts.
Optionally runs the generated ctx-bins on Android via run_genie_htp_real_llm_android.sh.

Required:
  --model-input PATH          Model input path (exports dir, ONNX file, or GGUF file)

Build options:
  --container-dir PATH        Output container dir
  --cache-root PATH           Cache dir for resumable builds
  --qairt-tmp-dir PATH        QAIRT_TMP_DIR path
  --chipset NAME              Target chipset (default: SM8750)
  --embedding-lut MODE        Embedding LUT mode: auto|on|off (default: auto)
  --sdk-root PATH             QAIRT SDK root
  --python PATH               Python interpreter (default: .conda-qairt310/bin/python)
  --tokenizer-path PATH       Optional explicit tokenizer path for builder
  --config-path PATH          Optional explicit config path for builder
  --calibration-input-list    Optional calibration input-list path for builder
  --intake-report-path PATH   Optional path for model intake JSON report
  --onnx-preflight MODE       ONNX preflight mode: auto|on|off (default: auto)
  --onnx-preflight-batch-size Static batch size for preflight (default: 1)
  --onnx-preflight-prompt-len Static prompt length for preflight (default: 1)
  --onnx-preflight-ctx-len    Static context length for preflight (default: 2048)
  --onnx-preflight-past-len   Static past-seq length for preflight (default: -1 => context_len-1)
  --ignore-intake-blockers    Continue build even if intake finds known blockers
  --clean                     Delete container/cache/tmp dirs before build

Android run options:
  --run-android               After build, run on Android using generated manifest
  --adb-serial SERIAL         Device serial
  --prompt TEXT               Prompt for genie-t2t-run
  --out-dir PATH              Target dir on device
  --tokenizer-json PATH       Runtime tokenizer override (if manifest lacks it)
  --n-vocab INT               Runtime n-vocab override
  --bos-token INT             Runtime bos-token override
  --eos-token INT             Runtime eos-token override

Example:
  scripts/sidecar_qnn/build_genie_htp_real_llm_assets.sh \
    --model-input /home/curious/models/Qwen3.5-4B-Q4_K_M.gguf \
    --run-android \
    --adb-serial 192.168.29.11:41993
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-input)
      MODEL_INPUT="$2"
      shift 2
      ;;
    --container-dir)
      CONTAINER_DIR="$2"
      shift 2
      ;;
    --cache-root)
      CACHE_ROOT="$2"
      shift 2
      ;;
    --qairt-tmp-dir)
      QAIRT_TMP_DIR="$2"
      shift 2
      ;;
    --chipset)
      CHIPSET="$2"
      shift 2
      ;;
    --embedding-lut)
      EMBEDDING_LUT_MODE="$2"
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --config-path)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --calibration-input-list)
      CALIBRATION_INPUT_LIST="$2"
      shift 2
      ;;
    --intake-report-path)
      INTAKE_REPORT_PATH="$2"
      shift 2
      ;;
    --onnx-preflight)
      ONNX_PREFLIGHT_MODE="$2"
      shift 2
      ;;
    --onnx-preflight-batch-size)
      ONNX_PREFLIGHT_BATCH_SIZE="$2"
      shift 2
      ;;
    --onnx-preflight-prompt-len)
      ONNX_PREFLIGHT_PROMPT_LENGTH="$2"
      shift 2
      ;;
    --onnx-preflight-ctx-len)
      ONNX_PREFLIGHT_CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --onnx-preflight-past-len)
      ONNX_PREFLIGHT_PAST_SEQ_LEN="$2"
      shift 2
      ;;
    --ignore-intake-blockers)
      IGNORE_INTAKE_BLOCKERS="1"
      shift
      ;;
    --clean)
      CLEAN="1"
      shift
      ;;
    --run-android)
      RUN_ANDROID="1"
      shift
      ;;
    --adb-serial)
      ADB_SERIAL="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --tokenizer-json)
      OVERRIDE_TOKENIZER_JSON="$2"
      shift 2
      ;;
    --n-vocab)
      OVERRIDE_N_VOCAB="$2"
      shift 2
      ;;
    --bos-token)
      OVERRIDE_BOS_TOKEN="$2"
      shift 2
      ;;
    --eos-token)
      OVERRIDE_EOS_TOKEN="$2"
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

if [[ -z "${MODEL_INPUT}" ]]; then
  echo "--model-input is required" >&2
  exit 1
fi
if [[ ! -e "${MODEL_INPUT}" ]]; then
  echo "Model input not found: ${MODEL_INPUT}" >&2
  exit 1
fi
if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${SDK_ROOT}/bin/envsetup.sh" ]]; then
  echo "Missing SDK envsetup: ${SDK_ROOT}/bin/envsetup.sh" >&2
  exit 1
fi

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf "${CONTAINER_DIR}" "${CACHE_ROOT}" "${QAIRT_TMP_DIR}"
fi
mkdir -p "${CONTAINER_DIR}" "${CACHE_ROOT}" "${QAIRT_TMP_DIR}"

if [[ "${ONNX_PREFLIGHT_MODE}" != "auto" && "${ONNX_PREFLIGHT_MODE}" != "on" && "${ONNX_PREFLIGHT_MODE}" != "off" ]]; then
  echo "Invalid --onnx-preflight mode: ${ONNX_PREFLIGHT_MODE} (expected auto|on|off)" >&2
  exit 1
fi

EFFECTIVE_MODEL_INPUT="${MODEL_INPUT}"
INTAKE_REPORT_EFFECTIVE="${INTAKE_REPORT_PATH:-${CONTAINER_DIR}/model_intake_report.json}"

intake_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/model_intake_report.py"
  --model-input "${MODEL_INPUT}"
  --output-json "${INTAKE_REPORT_EFFECTIVE}"
)
if [[ -n "${CALIBRATION_INPUT_LIST}" ]]; then
  intake_cmd+=(--calibration-input-list "${CALIBRATION_INPUT_LIST}")
fi
"${intake_cmd[@]}"

mapfile -t _intake_fields < <(
  "${PYTHON_BIN}" - "${INTAKE_REPORT_EFFECTIVE}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    r = json.load(f)
print(r.get("model_kind", "unknown"))
print(r.get("onnx_path", ""))
print("1" if r.get("qairt_ar_cl_context_heuristic_likely_fail") else "0")
print("1" if r.get("qairt_simplified_layernorm_opset_risk") else "0")
print("1" if r.get("requires_calibration_or_encodings") else "0")
print("1" if r.get("qairt_protobuf_serialization_risk") else "0")
print("1" if r.get("qairt_gguf_architecture_unsupported") else "0")
print(r.get("gguf_architecture", "") or "")
PY
)
INTAKE_MODEL_KIND="${_intake_fields[0]:-unknown}"
INTAKE_ONNX_PATH="${_intake_fields[1]:-}"
INTAKE_AR_CL_RISK="${_intake_fields[2]:-0}"
INTAKE_SLN_RISK="${_intake_fields[3]:-0}"
INTAKE_NEEDS_CALIBRATION="${_intake_fields[4]:-0}"
INTAKE_PROTOBUF_RISK="${_intake_fields[5]:-0}"
INTAKE_GGUF_ARCH_UNSUPPORTED="${_intake_fields[6]:-0}"
INTAKE_GGUF_ARCH="${_intake_fields[7]:-}"

printf '[info] intake_report=%s\n' "${INTAKE_REPORT_EFFECTIVE}"
printf '[info] intake.model_kind=%s\n' "${INTAKE_MODEL_KIND}"
if [[ -n "${INTAKE_ONNX_PATH}" ]]; then
  printf '[info] intake.onnx_path=%s\n' "${INTAKE_ONNX_PATH}"
fi
printf '[info] intake.risk_ar_cl=%s\n' "${INTAKE_AR_CL_RISK}"
printf '[info] intake.risk_simplified_layernorm=%s\n' "${INTAKE_SLN_RISK}"
printf '[info] intake.risk_protobuf_serialize=%s\n' "${INTAKE_PROTOBUF_RISK}"
printf '[info] intake.needs_calibration_or_encodings=%s\n' "${INTAKE_NEEDS_CALIBRATION}"
if [[ -n "${INTAKE_GGUF_ARCH}" ]]; then
  printf '[info] intake.gguf_architecture=%s\n' "${INTAKE_GGUF_ARCH}"
fi
printf '[info] intake.risk_gguf_arch_unsupported=%s\n' "${INTAKE_GGUF_ARCH_UNSUPPORTED}"

if [[ "${INTAKE_PROTOBUF_RISK}" == "1" && "${IGNORE_INTAKE_BLOCKERS}" != "1" ]]; then
  echo "Known blocker: ONNX external data size indicates high risk of protobuf serialization failure during AR/CL conversion." >&2
  echo "Intake blocker active. Re-run with --ignore-intake-blockers to force the build attempt." >&2
  exit 2
fi
if [[ "${INTAKE_GGUF_ARCH_UNSUPPORTED}" == "1" && "${IGNORE_INTAKE_BLOCKERS}" != "1" ]]; then
  echo "Known blocker: GGUF architecture is unsupported by the current QAIRT GGUF builder." >&2
  echo "Detected architecture: ${INTAKE_GGUF_ARCH:-unknown}" >&2
  echo "Intake blocker active. Re-run with --ignore-intake-blockers to force the build attempt." >&2
  exit 2
fi

if [[ "${INTAKE_MODEL_KIND}" == "onnx" && -n "${INTAKE_ONNX_PATH}" ]]; then
  if [[ -d "${MODEL_INPUT}" ]]; then
    EFFECTIVE_MODEL_INPUT="${INTAKE_ONNX_PATH}"
    if [[ -z "${CONFIG_PATH}" ]]; then
      if [[ -f "${MODEL_INPUT}/config.json" ]]; then
        CONFIG_PATH="${MODEL_INPUT}/config.json"
      fi
    fi
    if [[ -z "${TOKENIZER_PATH}" ]]; then
      if [[ -f "${MODEL_INPUT}/tokenizer.json" ]]; then
        TOKENIZER_PATH="${MODEL_INPUT}/tokenizer.json"
      elif [[ -f "${MODEL_INPUT}/tokenizer.model" ]]; then
        TOKENIZER_PATH="${MODEL_INPUT}/tokenizer.model"
      fi
    fi
    printf '[info] intake.selected_onnx=%s\n' "${EFFECTIVE_MODEL_INPUT}"
  fi

  SHOULD_PREFLIGHT="0"
  if [[ "${ONNX_PREFLIGHT_MODE}" == "on" ]]; then
    SHOULD_PREFLIGHT="1"
  elif [[ "${ONNX_PREFLIGHT_MODE}" == "auto" && ( "${INTAKE_AR_CL_RISK}" == "1" || "${INTAKE_SLN_RISK}" == "1" ) ]]; then
    SHOULD_PREFLIGHT="1"
  fi

  if [[ "${SHOULD_PREFLIGHT}" == "1" ]]; then
    PREFLIGHT_DIR="${CACHE_ROOT}/onnx_preflight"
    mkdir -p "${PREFLIGHT_DIR}"
    _onnx_name="$(basename "${INTAKE_ONNX_PATH}")"
    _onnx_stem="${_onnx_name%.onnx}"
    PREFLIGHT_ONNX_PATH="${PREFLIGHT_DIR}/${_onnx_stem}.preflight.onnx"
    PREFLIGHT_PAST_SEQ_LEN="${ONNX_PREFLIGHT_PAST_SEQ_LEN}"
    if [[ "${PREFLIGHT_PAST_SEQ_LEN}" -lt 0 ]]; then
      PREFLIGHT_PAST_SEQ_LEN="$((ONNX_PREFLIGHT_CONTEXT_LENGTH - 1))"
    fi

    "${PYTHON_BIN}" "${SCRIPT_DIR}/onnx_preflight_fix.py" \
      --input-onnx "${INTAKE_ONNX_PATH}" \
      --output-onnx "${PREFLIGHT_ONNX_PATH}" \
      --batch-size "${ONNX_PREFLIGHT_BATCH_SIZE}" \
      --prompt-length "${ONNX_PREFLIGHT_PROMPT_LENGTH}" \
      --context-length "${ONNX_PREFLIGHT_CONTEXT_LENGTH}" \
      --past-seq-len "${PREFLIGHT_PAST_SEQ_LEN}"

    EFFECTIVE_MODEL_INPUT="${PREFLIGHT_ONNX_PATH}"
    if [[ -z "${CONFIG_PATH}" ]]; then
      if [[ -f "${MODEL_INPUT}/config.json" ]]; then
        CONFIG_PATH="${MODEL_INPUT}/config.json"
      elif [[ -f "$(dirname "${MODEL_INPUT}")/config.json" ]]; then
        CONFIG_PATH="$(dirname "${MODEL_INPUT}")/config.json"
      fi
    fi
    if [[ -z "${TOKENIZER_PATH}" ]]; then
      if [[ -f "${MODEL_INPUT}/tokenizer.json" ]]; then
        TOKENIZER_PATH="${MODEL_INPUT}/tokenizer.json"
      elif [[ -f "$(dirname "${MODEL_INPUT}")/tokenizer.json" ]]; then
        TOKENIZER_PATH="$(dirname "${MODEL_INPUT}")/tokenizer.json"
      elif [[ -f "${MODEL_INPUT}/tokenizer.model" ]]; then
        TOKENIZER_PATH="${MODEL_INPUT}/tokenizer.model"
      elif [[ -f "$(dirname "${MODEL_INPUT}")/tokenizer.model" ]]; then
        TOKENIZER_PATH="$(dirname "${MODEL_INPUT}")/tokenizer.model"
      fi
    fi
    printf '[info] preflight.onnx=%s\n' "${PREFLIGHT_ONNX_PATH}"
    if [[ -n "${CONFIG_PATH}" ]]; then
      printf '[info] preflight.config_path=%s\n' "${CONFIG_PATH}"
    fi
    if [[ -n "${TOKENIZER_PATH}" ]]; then
      printf '[info] preflight.tokenizer_path=%s\n' "${TOKENIZER_PATH}"
    fi
  fi
fi

if [[ "${INTAKE_SLN_RISK}" == "1" && "${IGNORE_INTAKE_BLOCKERS}" != "1" && "${ONNX_PREFLIGHT_MODE}" == "off" ]]; then
  echo "Known blocker: SimplifiedLayerNormalization with high opset is likely unsupported in current QAIRT path." >&2
  echo "Enable preflight (default auto) or re-run with --ignore-intake-blockers to force the build attempt." >&2
  exit 2
fi

if [[ -n "${CONFIG_PATH}" && -f "${CONFIG_PATH}" ]]; then
  NEEDS_FLATTENED_CONFIG="$(
    "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = json.load(f)
has_vocab = "vocab_size" in c
has_text_vocab = isinstance(c.get("text_config"), dict) and ("vocab_size" in c["text_config"])
print("1" if (not has_vocab and has_text_vocab) else "0")
PY
  )"
  if [[ "${NEEDS_FLATTENED_CONFIG}" == "1" ]]; then
    FLAT_CONFIG_PATH="${CACHE_ROOT}/onnx_preflight/$(basename "${CONFIG_PATH%.json}").flattened.json"
    mkdir -p "$(dirname "${FLAT_CONFIG_PATH}")"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/flatten_multimodal_text_config.py" \
      --input-config "${CONFIG_PATH}" \
      --output-config "${FLAT_CONFIG_PATH}"
    CONFIG_PATH="${FLAT_CONFIG_PATH}"
    printf '[info] preflight.flattened_config=%s\n' "${CONFIG_PATH}"
  fi
fi

printf '[info] model_input=%s\n' "${MODEL_INPUT}"
printf '[info] effective_model_input=%s\n' "${EFFECTIVE_MODEL_INPUT}"
printf '[info] python=%s\n' "${PYTHON_BIN}"
printf '[info] sdk_root=%s\n' "${SDK_ROOT}"
printf '[info] container_dir=%s\n' "${CONTAINER_DIR}"
printf '[info] cache_root=%s\n' "${CACHE_ROOT}"
printf '[info] chipset=%s\n' "${CHIPSET}"
printf '[info] embedding_lut_mode=%s\n' "${EMBEDDING_LUT_MODE}"

set +u
source "${SDK_ROOT}/bin/envsetup.sh"
set -u
export LD_LIBRARY_PATH="${REPO_ROOT}/../OnDevAI_external/.conda-qairt310/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${QAIRT_TMP_DIR}"

build_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_genai_llm_container_htp.py"
  --model-input "${EFFECTIVE_MODEL_INPUT}"
  --container-dir "${CONTAINER_DIR}"
  --cache-root "${CACHE_ROOT}"
  --chipset "${CHIPSET}"
  --embedding-lut "${EMBEDDING_LUT_MODE}"
  --qairt-tmp-dir "${QAIRT_TMP_DIR}"
)
if [[ -n "${TOKENIZER_PATH}" ]]; then
  build_cmd+=(--tokenizer-path "${TOKENIZER_PATH}")
fi
if [[ -n "${CONFIG_PATH}" ]]; then
  build_cmd+=(--config-path "${CONFIG_PATH}")
fi
if [[ -n "${CALIBRATION_INPUT_LIST}" ]]; then
  build_cmd+=(--calibration-input-list "${CALIBRATION_INPUT_LIST}")
fi

"${build_cmd[@]}"

MANIFEST_PATH="${CONTAINER_DIR}/htp_runtime_manifest.json"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

printf '[info] manifest=%s\n' "${MANIFEST_PATH}"
printf '[info] next: run scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh with the ctx-bins from this manifest\n'

if [[ "${RUN_ANDROID}" != "1" ]]; then
  exit 0
fi

if [[ -z "${ADB_SERIAL}" ]]; then
  mapfile -t _adb_devices < <(adb devices | awk '$2 == "device" {print $1}')
  if [[ "${#_adb_devices[@]}" -eq 1 ]]; then
    ADB_SERIAL="${_adb_devices[0]}"
  fi
fi
if [[ -z "${ADB_SERIAL}" ]]; then
  echo "No adb serial provided and auto-detection did not find exactly one connected device" >&2
  exit 1
fi

readarray -t MANIFEST_CTX_BINS < <(
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
for p in d.get("ctx_bins", []):
    print(p)
PY
)

if [[ "${#MANIFEST_CTX_BINS[@]}" -eq 0 ]]; then
  echo "Manifest has no ctx bins: ${MANIFEST_PATH}" >&2
  exit 1
fi

MANIFEST_TOKENIZER_JSON="$("${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(d.get("tokenizer_json") or "")
PY
)"
MANIFEST_N_VOCAB="$("${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print("" if d.get("n_vocab") is None else d["n_vocab"])
PY
)"
MANIFEST_BOS="$("${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print("" if d.get("bos_token") is None else d["bos_token"])
PY
)"
MANIFEST_EOS="$("${PYTHON_BIN}" - "${MANIFEST_PATH}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print("" if d.get("eos_token") is None else d["eos_token"])
PY
)"

RUNTIME_TOKENIZER_JSON="${OVERRIDE_TOKENIZER_JSON:-${MANIFEST_TOKENIZER_JSON}}"
RUNTIME_N_VOCAB="${OVERRIDE_N_VOCAB:-${MANIFEST_N_VOCAB}}"
RUNTIME_BOS_TOKEN="${OVERRIDE_BOS_TOKEN:-${MANIFEST_BOS}}"
RUNTIME_EOS_TOKEN="${OVERRIDE_EOS_TOKEN:-${MANIFEST_EOS}}"

if [[ -z "${RUNTIME_TOKENIZER_JSON}" || -z "${RUNTIME_N_VOCAB}" || -z "${RUNTIME_BOS_TOKEN}" || -z "${RUNTIME_EOS_TOKEN}" ]]; then
  echo "Missing runtime tokenizer/id values. Pass overrides: --tokenizer-json --n-vocab --bos-token --eos-token" >&2
  exit 1
fi

run_cmd=(
  "${SCRIPT_DIR}/run_genie_htp_real_llm_android.sh"
  --sdk-root "${SDK_ROOT}"
  --adb-serial "${ADB_SERIAL}"
  --tokenizer-json "${RUNTIME_TOKENIZER_JSON}"
  --n-vocab "${RUNTIME_N_VOCAB}"
  --bos-token "${RUNTIME_BOS_TOKEN}"
  --eos-token "${RUNTIME_EOS_TOKEN}"
  --prompt "${PROMPT}"
  --out-dir "${OUT_DIR}"
)
for ctx_bin in "${MANIFEST_CTX_BINS[@]}"; do
  run_cmd+=(--ctx-bin "${ctx_bin}")
done

"${run_cmd[@]}"
