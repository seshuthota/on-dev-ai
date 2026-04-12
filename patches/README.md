# Patches

This directory stores local patches for external dependencies that are not applied automatically.

## Current Patches

### ondevai-local.patch

```text
patches/llama.cpp/ondevai-local.patch
```

Captures local changes from the pre-git `/home/curious/llama.cpp` checkout at upstream commit `d00685831`. The repo uses `native/third_party/llama.cpp` as a submodule, so apply this patch manually only when reproducing the old pre-reset `llama.cpp` behavior:

```bash
git -C native/third_party/llama.cpp apply ../../../patches/llama.cpp/ondevai-local.patch
```

To remove/reverse:
```bash
git -C native/third_party/llama.cpp apply -R ../../../patches/llama.cpp/ondevai-local.patch
```

### ondevai-profiler.patch

```text
patches/llama.cpp/ondevai-profiler.patch
```

Optional instrumentation patch providing per-op GGML profiler with type-splitting for MUL_MAT operations. Compile with `-DGGML_PROFILER` and set `GGML_PROFILER=1` env var at runtime to enable.

**Note:** Profiler builds are for hotspot direction only, not production perf numbers. Profiler overhead significantly impacts throughput (see profiler A/B findings in `docs/benchmarks/llama_cpp_android_baseline.md`).

Apply:
```bash
git -C native/third_party/llama.cpp apply ../../../patches/llama.cpp/ondevai-profiler.patch
```

Remove/reverse:
```bash
git -C native/third_party/llama.cpp apply -R ../../../patches/llama.cpp/ondevai-profiler.patch
```
