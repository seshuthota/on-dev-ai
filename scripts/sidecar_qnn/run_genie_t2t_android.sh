#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="cpu"
ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/sidecar_qnn/run_genie_t2t_android.sh [options]

Compatibility wrapper.

This old mixed script name is deprecated because it encouraged confusion between:

- the current proven CPU-realized Genie path
- the target HTP-realized Genie path

Use one of these instead:

- scripts/sidecar_qnn/run_genie_cpu_real_llm_android.sh
- scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh

Compatibility behavior:

- `--backend cpu` forwards to `run_genie_cpu_real_llm_android.sh`
- `--backend htp-v79` is rejected intentionally; use the explicit HTP script
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "${BACKEND}" in
  cpu)
    exec "${SCRIPT_DIR}/run_genie_cpu_real_llm_android.sh" "${ARGS[@]}"
    ;;
  htp-v79)
    echo "scripts/sidecar_qnn/run_genie_t2t_android.sh no longer accepts --backend htp-v79." >&2
    echo "Use scripts/sidecar_qnn/run_genie_htp_real_llm_android.sh with real ctx-bin inputs." >&2
    exit 2
    ;;
  *)
    echo "Unsupported backend: ${BACKEND}" >&2
    usage >&2
    exit 1
    ;;
esac
