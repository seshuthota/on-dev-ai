# Patches

This directory stores local patches for external dependencies that are not applied automatically.

Current patch:

```text
patches/llama.cpp/ondevai-local.patch
```

It captures local changes from the pre-git `/home/curious/llama.cpp` checkout at upstream commit `d00685831`. The repo uses `native/third_party/llama.cpp` as a submodule, so apply this patch manually only when reproducing the old pre-reset `llama.cpp` behavior:

```bash
git -C native/third_party/llama.cpp apply ../../../patches/llama.cpp/ondevai-local.patch
```
