#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/model.gguf"
  exit 1
fi

HOST_MODEL_PATH="$1"
DEVICE_MODEL_DIR="/sdcard/Android/data/ai.ondev.snapdragonlab/files/models"
DEVICE_MODEL_PATH="$DEVICE_MODEL_DIR/primary-model.gguf"

if [[ ! -f "$HOST_MODEL_PATH" ]]; then
  echo "Model file not found: $HOST_MODEL_PATH"
  exit 1
fi

adb shell "mkdir -p '$DEVICE_MODEL_DIR'"
adb push "$HOST_MODEL_PATH" "$DEVICE_MODEL_PATH"

echo "Model pushed to $DEVICE_MODEL_PATH"
