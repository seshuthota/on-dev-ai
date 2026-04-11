#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <serial>"
  exit 1
fi

SER="$1"

adb -s "$SER" shell "run-as ai.ondev.snapdragonlab sh -c 'ls -lh files/models 2>/dev/null || true'"
