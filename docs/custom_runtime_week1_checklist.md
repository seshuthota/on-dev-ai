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

- [ ] Create `native/custom/include/`.
- [ ] Create `native/custom/src/`.
- [ ] Define `Tensor`.
- [ ] Define `Model`.
- [ ] Define `Runtime`.
- [ ] Define `KvCache`.
- [ ] Define `BenchmarkResult`.
- [ ] Add a custom native target that compiles without replacing the existing app engine.

## Day 3: Packer And Metadata

- [ ] Create `scripts/custom_runtime/pack_tinyllama.py`.
- [ ] Read `config.json`.
- [ ] Read `model.safetensors`.
- [ ] Convert BF16 source weights to FP16.
- [ ] Write `manifest.json`.
- [ ] Write `model.bin` header.
- [ ] Write tensor directory.
- [ ] Pack embeddings, `lm_head`, final norm, and layer 0 tensors.
- [ ] Copy tokenizer artifacts.

## Day 4: Desktop Reference Loader

- [ ] Add `run_reference`.
- [ ] Load `model.bin`.
- [ ] Validate header fields.
- [ ] Validate tensor directory offsets.
- [ ] Look up tensors by name.
- [ ] Print metadata summary.

## Day 5: First Kernels

- [ ] Implement RMSNorm reference kernel.
- [ ] Implement RoPE reference kernel.
- [ ] Implement FP16 matvec reference kernel.
- [ ] Add deterministic tests.
- [ ] Compare at least one layer operation against a known reference.

## Week 1 Gate

Week 1 is complete only when:

- [ ] packed file loads
- [ ] desktop reference runs selected layer operations
- [ ] one layer operation matches expected output

## Hard Stops

- [ ] Do not start Vulkan work.
- [ ] Do not start QNN/HTP work.
- [ ] Do not add a second model family.
- [ ] Do not add GGUF runtime loading.
