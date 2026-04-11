#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "Usage: $0 <serial> <backend> <mode> <gpu_layers> <models_csv> [timeout_sec] [report_file|repeat_count] [repeat_count]"
  echo "Examples:"
  echo "  $0 192.168.29.11:41993 cpu smoke -1 'primary-model.gguf,model2.gguf'"
  echo "  $0 192.168.29.11:41993 cpu smoke -1 'primary-model.gguf,model2.gguf' 240 3"
  echo "  $0 192.168.29.11:41993 cpu smoke -1 'primary-model.gguf,model2.gguf' 240 /tmp/report.txt 3"
  exit 1
fi

SER="$1"
BACKEND="$2"
MODE="$3"
GPU_LAYERS="$4"
MODELS_CSV="$5"
TIMEOUT_SEC="${6:-240}"
DEFAULT_REPORT_FILE="/tmp/ondevai_matrix_${BACKEND}_${MODE}_$(date +%s).txt"
REPORT_FILE="$DEFAULT_REPORT_FILE"
REPEAT_COUNT=1

if [[ $# -ge 7 ]]; then
  if [[ $# -eq 7 && "$7" =~ ^[0-9]+$ ]]; then
    REPEAT_COUNT="$7"
  else
    REPORT_FILE="$7"
  fi
fi

if [[ $# -eq 8 ]]; then
  REPEAT_COUNT="$8"
fi

if [[ ! "$TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$TIMEOUT_SEC" -lt 1 ]]; then
  echo "timeout_sec must be a positive integer: $TIMEOUT_SEC" >&2
  exit 1
fi

if [[ ! "$REPEAT_COUNT" =~ ^[0-9]+$ ]] || [[ "$REPEAT_COUNT" -lt 1 ]]; then
  echo "repeat_count must be a positive integer: $REPEAT_COUNT" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/adb_backend_benchmark_worker.sh"

if [[ ! -x "$WORKER_SCRIPT" ]]; then
  chmod +x "$WORKER_SCRIPT"
fi

IFS=',' read -r -a MODELS <<<"$MODELS_CSV"

if [[ "${#MODELS[@]}" -eq 0 ]]; then
  echo "No models were provided in models_csv" >&2
  exit 1
fi

echo "matrix_start backend=$BACKEND mode=$MODE gpu_layers=$GPU_LAYERS timeout_sec=$TIMEOUT_SEC repeat_count=$REPEAT_COUNT models=${#MODELS[@]}" | tee "$REPORT_FILE"

for raw_model in "${MODELS[@]}"; do
  model="$(printf '%s' "$raw_model" | sed 's/^ *//; s/ *$//')"
  if [[ -z "$model" ]]; then
    continue
  fi

  echo "" | tee -a "$REPORT_FILE"
  echo "model_start model=$model" | tee -a "$REPORT_FILE"

  model_tmp_dir="$(mktemp -d)"
  for repeat_idx in $(seq 1 "$REPEAT_COUNT"); do
    echo "model_repeat_start model=$model repeat=${repeat_idx}/${REPEAT_COUNT}" | tee -a "$REPORT_FILE"

    tmp_out="$model_tmp_dir/repeat_${repeat_idx}.out"
    if "$WORKER_SCRIPT" "$SER" "$BACKEND" "$MODE" "$GPU_LAYERS" "$TIMEOUT_SEC" "$model" | tee "$tmp_out"; then
      :
    else
      echo "model_repeat_status model=$model repeat=$repeat_idx status=script_error" | tee -a "$REPORT_FILE"
      continue
    fi

    summary_line="$(rg '^summary ' "$tmp_out" | tail -n 1 || true)"
    entry_lines="$(rg '^entry ' "$tmp_out" || true)"
    signal_line="$(rg '^signals ' "$tmp_out" | tail -n 1 || true)"
    missing_line="$(rg '^status=missing_result' "$tmp_out" | tail -n 1 || true)"

    if [[ -n "$summary_line" ]]; then
      echo "model_repeat_summary model=$model repeat=$repeat_idx $summary_line" | tee -a "$REPORT_FILE"
      if [[ -n "$entry_lines" ]]; then
        while IFS= read -r line; do
          [[ -n "$line" ]] && echo "model_repeat_entry model=$model repeat=$repeat_idx $line" | tee -a "$REPORT_FILE"
        done <<<"$entry_lines"
      fi
      if [[ -n "$signal_line" ]]; then
        echo "model_repeat_signals model=$model repeat=$repeat_idx $signal_line" | tee -a "$REPORT_FILE"
      fi
    elif [[ -n "$missing_line" ]]; then
      echo "model_repeat_status model=$model repeat=$repeat_idx status=missing_result" | tee -a "$REPORT_FILE"
    else
      echo "model_repeat_status model=$model repeat=$repeat_idx status=unknown" | tee -a "$REPORT_FILE"
    fi
  done

  python3 - "$model" "$model_tmp_dir" "$REPEAT_COUNT" <<'PY' | tee -a "$REPORT_FILE"
import re
import statistics
import sys
from pathlib import Path

model = sys.argv[1]
model_tmp_dir = Path(sys.argv[2])
repeats_total = int(sys.argv[3])

def parse_key(line: str, key: str):
    m = re.search(rf"{re.escape(key)}=([^ ]+)", line)
    return m.group(1) if m else None

def parse_int(line: str, key: str):
    v = parse_key(line, key)
    if v is None or v == "None":
        return None
    try:
        return int(float(v))
    except ValueError:
        return None

def parse_float(line: str, key: str):
    v = parse_key(line, key)
    if v is None or v == "None":
        return None
    try:
        return float(v)
    except ValueError:
        return None

def fmt_number(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

repeat_files = sorted(
    model_tmp_dir.glob("repeat_*.out"),
    key=lambda p: int(re.search(r"repeat_(\d+)\.out", p.name).group(1)),
)

records = []
for repeat_file in repeat_files:
    m = re.search(r"repeat_(\d+)\.out", repeat_file.name)
    repeat_idx = int(m.group(1)) if m else -1
    lines = repeat_file.read_text(errors="ignore").splitlines()
    summary = next((line for line in lines if line.startswith("summary ")), None)
    if not summary:
        continue

    entry_lines = [line for line in lines if line.startswith("entry ")]
    signal = next((line for line in lines if line.startswith("signals ")), None)
    signal_count = parse_int(signal or "", "fused_gated_delta_net_not_supported")
    entries = []
    for entry_line in entry_lines:
        entries.append(
            {
                "name": parse_key(entry_line, "name"),
                "decoded_tokens": parse_int(entry_line, "decoded"),
                "elapsed_ms": parse_int(entry_line, "elapsed_ms"),
                "ttft_ms": parse_int(entry_line, "ttft_ms"),
                "tok_per_sec": parse_float(entry_line, "tok_per_sec"),
                "success": parse_key(entry_line, "success"),
            }
        )

    records.append(
        {
            "repeat_idx": repeat_idx,
            "mode": parse_key(summary, "mode"),
            "backend_target": parse_key(summary, "backend_target"),
            "backend_effective": parse_key(summary, "backend_effective"),
            "load_ms": parse_int(summary, "load_ms"),
            "model_path": parse_key(summary, "model_path"),
            "entries": entries,
            "signal_count": signal_count,
        }
    )

if not records:
    print(
        f"model_median_status model={model} status=no_successful_repeats repeats_ok=0 repeats_total={repeats_total}"
    )
    sys.exit(0)

def median(values):
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None

first = records[0]
load_median = median([r["load_ms"] for r in records])
signal_median = median([r["signal_count"] for r in records])

print(
    "model_median "
    f"model={model} repeats_ok={len(records)} repeats_total={repeats_total} "
    f"mode={first['mode']} backend_target={first['backend_target']} "
    f"backend_effective={first['backend_effective']} "
    f"model_path={first['model_path']} load_ms_median={fmt_number(load_median)}"
)

entry_names = []
for rec in records:
    for entry in rec["entries"]:
        if entry["name"] and entry["name"] not in entry_names:
            entry_names.append(entry["name"])

for name in entry_names:
    selected = []
    for rec in records:
        for entry in rec["entries"]:
            if entry["name"] == name:
                selected.append(entry)
                break

    decoded_median = median([e["decoded_tokens"] for e in selected])
    elapsed_median = median([e["elapsed_ms"] for e in selected])
    ttft_median = median([e["ttft_ms"] for e in selected])
    tok_median = median([e["tok_per_sec"] for e in selected])
    success_count = sum(1 for e in selected if str(e["success"]).lower() == "true")
    success_rate = success_count / len(selected) if selected else 0.0

    print(
        "model_median_entry "
        f"model={model} name={name} "
        f"decoded_tokens_median={fmt_number(decoded_median)} "
        f"elapsed_ms_median={fmt_number(elapsed_median)} "
        f"ttft_ms_median={fmt_number(ttft_median)} "
        f"tok_per_sec_median={fmt_number(tok_median)} "
        f"success_rate={fmt_number(success_rate)}"
    )

if signal_median is not None:
    print(
        "model_median_signals "
        f"model={model} fused_gated_delta_net_not_supported_median={fmt_number(signal_median)}"
    )
PY

  rm -rf "$model_tmp_dir"
done

echo "" | tee -a "$REPORT_FILE"
echo "matrix_done report_file=$REPORT_FILE" | tee -a "$REPORT_FILE"
