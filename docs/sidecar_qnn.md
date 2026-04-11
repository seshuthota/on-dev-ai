# QNN / QAIRT Sidecar

## Status

QNN/QAIRT is preserved as a sidecar path. It is not the custom-runtime mainline for Milestones 0-3 in `RESET_PLAN_V2.md`.

Use it only to:

- validate known-good QNN runtime setup
- preserve prior HTP control-path evidence
- compare against the owned custom runtime after benchmark records exist

Do not use it to restart ONNX/GGUF builder debugging as mainline work.

## SDK Location

The QAIRT SDK is no longer treated as a repo-local source directory. Set it explicitly:

```bash
export QAIRT_SDK_ROOT=/absolute/path/to/qairt/2.45.0.260326
export QNN_SDK_ROOT="${QAIRT_SDK_ROOT}"
export ONDEVAI_QAIRT_PYTHON=/absolute/path/to/qairt-python/bin/python
```

See `docs/qnn_sdk_setup.md` for detailed setup notes.

## Sidecar Commands

```bash
scripts/sidecar_qnn/check_qnn_sdk.sh --sdk-root "$QAIRT_SDK_ROOT"
scripts/sidecar_qnn/run_qnn_smoke_android.sh --backend cpu --adb-serial <serial> --sdk-root "$QAIRT_SDK_ROOT"
scripts/sidecar_qnn/run_qnn_smoke_android.sh --backend htp-v79 --adb-serial <serial> --sdk-root "$QAIRT_SDK_ROOT"
```

## Rule

If a task requires QAIRT builder acceptance, ONNX graph repair, Genie config compatibility, or architecture support in vendor tooling, it is sidecar work and must be timeboxed.
