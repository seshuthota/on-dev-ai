#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APK_PATH="$ROOT_DIR/app/android/app/build/outputs/apk/debug/app-debug.apk"

if [[ ! -f "$APK_PATH" ]]; then
  echo "APK not found at $APK_PATH"
  echo "Run ./gradlew :app:assembleDebug first."
  exit 1
fi

adb install -r "$APK_PATH"
