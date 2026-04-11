#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${ONDEVAI_QAIRT_PYTHON:-${REPO_ROOT}/../OnDevAI_external/.conda-qairt310/bin/python}"
PIP_BIN="${PYTHON_BIN%/python}/pip"

if [[ ! -f "${PYTHON_BIN}" ]]; then
  echo "Python not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PIP_BIN}" ]]; then
  echo "pip not found for python: ${PIP_BIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" -V

"${PIP_BIN}" install --disable-pip-version-check \
  numpy \
  jsonschema \
  onnx \
  onnx-ir \
  onnxscript \
  scipy \
  islpy \
  onnxruntime \
  onnx-graphsurgeon \
  torch \
  gguf \
  onnxruntime-genai

echo "[info] QAIRT Python dependencies installed in: ${PYTHON_BIN%/python}"
