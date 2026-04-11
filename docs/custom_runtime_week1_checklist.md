# Custom Runtime Week 1 Checklist

This checklist is the first execution slice from `RESET_PLAN_V2.md`.

## Day 1: Freeze And Pin

- [x] Confirm `.git` metadata status.
- [x] Record local app build result.
- [x] Record adb/device availability.
- [x] Confirm TinyLlama local files exist.
- [x] Record `model.safetensors` SHA-256.
- [x] Create `MODEL_DECISION.md`.
- [x] Create `docs/custom_runtime_baseline.md`.
- [ ] Append on-device CPU benchmark if an adb device becomes available.
- [ ] Append QNN control-smoke result if an adb device becomes available.

## Day 2: Skeleton Target

- [x] Create `native/custom/include/`.
- [x] Create `native/custom/src/`.
- [x] Define `Tensor`.
- [x] Define `Model`.
- [x] Define `Runtime`.
- [x] Define `KvCache`.
- [x] Define `BenchmarkResult`.
- [x] Add a custom native target that compiles without replacing the existing app engine.

## Day 3: Packer And Metadata

- [x] Create `scripts/custom_runtime/pack_tinyllama.py`.
- [x] Read `config.json`.
- [x] Read `model.safetensors`.
- [x] Convert BF16 source weights to FP16.
- [x] Write `manifest.json`.
- [x] Write `model.bin` header.
- [x] Write tensor directory.
- [x] Pack embeddings, `lm_head`, final norm, and layer 0 tensors.
- [x] Copy tokenizer artifacts.

## Day 4: Desktop Reference Loader

- [x] Add `run_reference`.
- [x] Load `model.bin`.
- [x] Validate header fields.
- [x] Validate tensor directory offsets.
- [x] Look up tensors by name.
- [x] Print metadata summary.

## Day 5: First Kernels

- [x] Implement RMSNorm reference kernel.
- [x] Implement RoPE reference kernel.
- [x] Implement FP16 matvec reference kernel.
- [x] Add deterministic tests.
- [ ] Compare at least one layer operation against a known reference.

## Week 1 Gate

Week 1 is complete only when:

- [x] packed file loads
- [ ] desktop reference runs selected layer operations
- [ ] one layer operation matches expected output

## Hard Stops

- [ ] Do not start Vulkan work.
- [ ] Do not start QNN/HTP work.
- [ ] Do not add a second model family.
- [ ] Do not add GGUF runtime loading.
