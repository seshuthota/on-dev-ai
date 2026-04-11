#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <serial> <host_model_path> [device_model_name] [--set-primary]"
  echo "Example: $0 192.168.29.11:41993 /models/llama.gguf control-llama3.2-3b.gguf"
  exit 1
fi

SER="$1"
HOST_MODEL_PATH="$2"
DEVICE_MODEL_NAME="${3:-}"
SET_PRIMARY="false"

if [[ "${3:-}" == "--set-primary" ]]; then
  DEVICE_MODEL_NAME=""
  SET_PRIMARY="true"
fi
if [[ "${4:-}" == "--set-primary" ]]; then
  SET_PRIMARY="true"
fi

if [[ ! -f "$HOST_MODEL_PATH" ]]; then
  echo "Model file not found: $HOST_MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$DEVICE_MODEL_NAME" ]]; then
  DEVICE_MODEL_NAME="$(basename "$HOST_MODEL_PATH")"
fi

TMP_DIR="/data/local/tmp/ondevai_models"
TMP_PATH="$TMP_DIR/$DEVICE_MODEL_NAME"
APP_MODEL_DIR="files/models"
APP_MODEL_PATH="$APP_MODEL_DIR/$DEVICE_MODEL_NAME"
PRIMARY_PATH="$APP_MODEL_DIR/primary-model.gguf"

echo "serial=$SER"
echo "host_model=$HOST_MODEL_PATH"
echo "device_model_name=$DEVICE_MODEL_NAME"
echo "tmp_path=$TMP_PATH"

adb -s "$SER" shell "mkdir -p '$TMP_DIR'"
adb -s "$SER" push "$HOST_MODEL_PATH" "$TMP_PATH"

adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'mkdir -p \"$APP_MODEL_DIR\" && cp \"$TMP_PATH\" \"$APP_MODEL_PATH\"'"

if [[ "$SET_PRIMARY" == "true" && "$DEVICE_MODEL_NAME" != "primary-model.gguf" ]]; then
  adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'cp \"$APP_MODEL_PATH\" \"$PRIMARY_PATH\"'"
fi

adb -s "$SER" shell "run-as ai.ondev.snapdragonlab stat -c '%n %s bytes' \"$APP_MODEL_PATH\""
if [[ "$SET_PRIMARY" == "true" ]]; then
  adb -s "$SER" shell "run-as ai.ondev.snapdragonlab stat -c '%n %s bytes' \"$PRIMARY_PATH\""
fi

adb -s "$SER" shell "rm -f '$TMP_PATH'" >/dev/null || true

echo "Installed to app internal path: $APP_MODEL_PATH"
