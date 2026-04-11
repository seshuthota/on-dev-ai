#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 6 ]]; then
  echo "Usage: $0 <serial> <backend: cpu|opencl|vulkan> <mode: micro|smoke|standard> <gpu_layers> [timeout_sec] [model_name_or_path]"
  exit 1
fi

SER="$1"
BACKEND="$2"
MODE="$3"
GPU_LAYERS="$4"
TIMEOUT_SEC="${5:-240}"
MODEL_SPEC="${6:-primary-model.gguf}"
APP_BENCH_HISTORY="files/benchmarks/history.jsonl"
APP_MODEL_DIR="/data/user/0/ai.ondev.snapdragonlab/files/models"
if [[ "$MODEL_SPEC" == */* ]]; then
  EXPECTED_MODEL_PATH="$MODEL_SPEC"
else
  EXPECTED_MODEL_PATH="$APP_MODEL_DIR/$MODEL_SPEC"
fi

COMP="ai.ondev.snapdragonlab/.DebugReceiver"
ACTION="ai.ondev.snapdragonlab.DEBUG_OPENCL_SMOKE"
MODEL_TOKEN="$(printf '%s' "$MODEL_SPEC" | sed 's#[^A-Za-z0-9._-]#_#g' | cut -c1-48)"
LOG_FILE="/tmp/ondevai_${BACKEND}_${MODE}_${MODEL_TOKEN}_worker_$(date +%s).log"

echo "serial=$SER backend=$BACKEND mode=$MODE gpu_layers=$GPU_LAYERS timeout_sec=$TIMEOUT_SEC model=$MODEL_SPEC"
echo "log_file=$LOG_FILE"
echo "force_stop=${ONDEVAI_BENCH_FORCE_STOP:-0}"

if [[ "${ONDEVAI_BENCH_FORCE_STOP:-0}" == "1" ]]; then
  adb -s "$SER" shell am force-stop ai.ondev.snapdragonlab >/dev/null || true
fi
adb -s "$SER" logcat -c

history_count_before="$(adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'wc -l $APP_BENCH_HISTORY 2>/dev/null | awk \"{print \\\$1}\"'" | tr -d '\r' | tr -d ' ' || true)"
if [[ -z "$history_count_before" ]]; then
  history_count_before=0
fi

broadcast_args=(
  shell am broadcast -n "$COMP" -a "$ACTION"
  --es backend "$BACKEND"
  --ei gpu_layers "$GPU_LAYERS"
  --es benchmark_mode "$MODE"
)

if [[ "$MODEL_SPEC" == */* ]]; then
  broadcast_args+=(--es model_path "$MODEL_SPEC")
else
  broadcast_args+=(--es model_name "$MODEL_SPEC")
fi

adb -s "$SER" "${broadcast_args[@]}" >/tmp/ondevai_bcast_${BACKEND}_${MODE}.txt

attempts=$((TIMEOUT_SEC / 5))
if [[ $attempts -lt 1 ]]; then
  attempts=1
fi

result_json=""
for _ in $(seq 1 "$attempts"); do
  sleep 5
  adb -s "$SER" logcat -d >"$LOG_FILE"

  history_count_now="$(adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'wc -l $APP_BENCH_HISTORY 2>/dev/null | awk \"{print \\\$1}\"'" | tr -d '\r' | tr -d ' ' || true)"
  if [[ -z "$history_count_now" ]]; then
    history_count_now=0
  fi

  if [[ "$history_count_now" =~ ^[0-9]+$ ]] && [[ "$history_count_before" =~ ^[0-9]+$ ]] && (( history_count_now > history_count_before )); then
    tail_count=$((history_count_now - history_count_before + 20))
    if [[ $tail_count -lt 40 ]]; then
      tail_count=40
    fi
    recent_history="$(adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'tail -n $tail_count $APP_BENCH_HISTORY 2>/dev/null'" | tr -d '\r' || true)"
    result_json="$(
      RECENT_HISTORY="$recent_history" python3 - "$MODE" "$BACKEND" "$EXPECTED_MODEL_PATH" <<'PY'
import json
import os
import sys

mode, backend, expected_model_path = sys.argv[1:4]

lines = [line.strip() for line in os.environ.get("RECENT_HISTORY", "").splitlines() if line.strip()]
best = None
for line in lines:
    try:
        obj = json.loads(line)
    except Exception:
        continue
    if obj.get("mode") != mode:
        continue
    if obj.get("backend_target") != backend:
        continue
    model_path = obj.get("model_path", "")
    if model_path != expected_model_path:
        continue
    best = obj

if best is not None:
    print(json.dumps(best, ensure_ascii=False))
PY
    )"
    if [[ -n "$result_json" ]]; then
      break
    fi
  fi
done

load_line="$(rg "LoadModel: complete total=.* backend=${BACKEND}" "$LOG_FILE" | tail -n 1 || true)"

if [[ -z "$result_json" ]]; then
  echo "status=missing_result"
  echo "recent_worker_lines:"
  rg -n "DebugReceiver: queued|DebugReceiver: starting|DebugReceiver: model|DebugReceiver: backend init|DebugReceiver: backend load|DebugReceiver: failure|WM-WorkerWrapper|WM-Processor" "$LOG_FILE" | tail -n 80 || true
  echo "recent_history_tail:"
  adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'tail -n 5 $APP_BENCH_HISTORY 2>/dev/null'" | tr -d '\r' || true
  exit 0
fi

python3 - "$MODE" "$load_line" "$result_json" "$LOG_FILE" <<'PY'
import json
import re
import sys
from pathlib import Path

mode, load_line, raw_json, log_file = sys.argv[1:5]
m = re.search(r"total=(\d+) ms", load_line)
load_ms = int(m.group(1)) if m else None

obj = json.loads(raw_json)

print(
    "summary "
    f"mode={obj.get('mode')} backend_target={obj.get('backend_target')} "
    f"backend_effective={obj.get('backend_effective')} load_ms={load_ms} "
    f"model_label={obj.get('model_label')} model_path={obj.get('model_path')} "
    f"threads={obj.get('threads')} ctx={obj.get('context_size')}"
)
for entry in obj.get("entries", []):
    print(
        "entry "
        f"name={entry.get('name')} decoded={entry.get('decoded_tokens')} "
        f"elapsed_ms={entry.get('elapsed_ms')} ttft_ms={entry.get('ttft_ms')} "
        f"tok_per_sec={entry.get('tok_per_sec')} success={entry.get('success')}"
    )

text = Path(log_file).read_text(errors="ignore")
gdn_fallbacks = len(re.findall(r"fused Gated Delta Net .* not supported", text))
print(f"signals fused_gated_delta_net_not_supported={gdn_fallbacks}")
PY
