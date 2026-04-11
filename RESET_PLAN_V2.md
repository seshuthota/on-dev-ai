# Strategic Reset v2: Custom Runtime Mainline

## Purpose

This document resets the project back to its original objective:

**Build phone-specific LLM inference for iQOO 13 / Snapdragon 8 Elite, including custom kernels where useful, and measure real sustained performance.**

The QAIRT/QNN path has already proven useful things:

- QNN runtime packaging works.
- App-side in-process QNN execution works.
- HTP `v79` is reachable for a control path.
- Real LLM generation via Genie has been demonstrated.

Those results are valuable, but they do not define the mainline anymore.

The reset is:

**Make the custom runtime the mainline. Freeze QAIRT/QNN as a sidecar baseline and optional future backend.**

The goal is no longer to make every model fit Qualcomm tooling, keep fighting ONNX/GGUF builder edge cases, or expand supported formats before the engine exists.

The goal is now to own one model family, one runtime, one benchmark harness, and the backend decisions.

## Core Rule

**Stop trying to make every model fit the phone through vendor tooling. Make one model fit the engine, then make the engine fit the phone.**

## 1. Frozen Decisions

These decisions are fixed for Milestones 0-3.

### First Model

`TinyLlama-1.1B-Chat-v1.0`

Reason:

- dense decoder-only Llama-style architecture
- easier first owned-runtime target than Qwen3.5 or multimodal models
- enough scale to matter, but small enough for first bring-up
- tokenizer/config/weights are easier to inspect than vendor-prepared artifacts

### Source Weights Format

Hugging Face `safetensors`

Reason:

- avoids early GGUF parser complexity
- avoids ONNX, builder, and architecture-tag drift
- gives a clean input for an owned packer

### Runtime Format v1

Owned packed **FP16** format only.

Reason:

- fastest path to correctness
- no Q4/Q8 until the engine works
- no direct GGUF runtime loading in v1

### Fixed Runtime Constraints

- context length: `512`
- decode mode: greedy only
- benchmark prompts: three prompts checked into the repo
- mainline backend through Milestone 3: Android CPU only

## 2. Project Split

### Track A: Custom Runtime Mainline

Purpose:

- build an owned inference engine
- own the model format
- own the decode loop
- own the benchmark harness
- measure CPU first, Vulkan second

Scope:

- Android NDK C++
- one model family
- one tokenizer path
- one packed format
- CPU correctness first
- Vulkan only after CPU correctness and profiling

### Track B: QAIRT/QNN Sidecar

Purpose:

- preserve current Qualcomm work as a baseline and comparison target
- keep control-model validation alive
- revisit HTP only after owned runtime benchmarking exists

Allowed work:

- keep SDK validation scripts
- keep control-model smoke tests
- keep comparison benchmark records
- keep docs describing current QNN evidence

Disallowed as mainline work:

- ONNX surgery
- architecture-tag patching
- random model conversion experiments
- embedding LUT workaround loops
- builder debugging without a direct benchmark question

## 3. Non-Goals For This Reset

Out of scope until Milestone 4 or later:

- multiple model families
- GGUF runtime loading
- ONNX runtime loading
- QAIRT/HTP as a mainline backend
- speculative decoding
- paged KV cache
- multimodal support
- tool calling
- framework-style abstraction for many architectures
- aggressive quantization before correctness
- trying to match or beat vendor tooling before the engine is stable

## 4. Success Definition

Short term:

**A small dense decoder runs end-to-end in the owned Android CPU runtime and produces benchmark records.**

Medium term:

**The owned CPU runtime is correct, profiled, and benchmarked well enough to guide optimization decisions.**

Long term:

**The engine owns model format, correctness oracle, decode loop, CPU kernels, benchmark harness, and backend decisions. QAIRT/QNN becomes an optional competitor, not the boss of the roadmap.**

## 5. Repository Layout Target

```text
native/custom/
  include/
    runtime.h
    model.h
    tensor.h
    tokenizer.h
    sampler.h
    kv_cache.h
    benchmark.h
    reference_dump.h
  src/
    runtime.cpp
    model_loader.cpp
    tokenizer.cpp
    sampler.cpp
    layers.cpp
    kv_cache.cpp
    benchmark.cpp
  tools/
    run_reference.cpp
    dump_reference_tensors.cpp
  tests/
    test_pack.cpp
    test_rmsnorm.cpp
    test_rope.cpp
    test_matvec.cpp
    test_tokenizer.cpp
  vulkan/
    vk_context.cpp
    kernels/
      fp16_matvec.comp
      q8_matvec.comp
scripts/
  pack_tinyllama.py
```

Runtime shape:

```text
Android UI -> ViewModel -> NativeBridge -> CustomRuntime
                                      -> CPU backend first
                                      -> Vulkan backend later
                                      -> QNN sidecar later
```

Important rule:

- `layers.cpp` may be explicit and ugly in v1.
- Do not over-abstract model execution early.
- Use a Python packer first; parsing `safetensors` in C++ is not the project. The runtime reads only the owned packed format.

## 6. Mandatory Oracle Path

Before Android becomes the main debug surface, there must be a desktop reference path.

Required tools:

- `scripts/custom_runtime/pack_tinyllama.py`: reads HF `safetensors` + config/tokenizer metadata and writes the owned packed format.
- `run_reference`: loads the packed model and runs a forward pass on desktop CPU.
- `dump_reference_tensors`: for one fixed prompt, dumps selected intermediate tensors.

Required tensor dump points:

- embedding output
- layer 0 RMSNorm output
- layer 0 Q projection output
- layer 0 attention output
- layer 0 MLP output
- final logits

Dump format can be binary or `.npy`. Simplicity wins.

Hard rule:

**No Android-side optimization work starts until desktop and Android can compare at least one layer boundary numerically.**

## 7. Packed Format v1

The first packed format should be intentionally boring.

Artifacts:

- `model.bin`
- `tokenizer/`
- `manifest.json`

`model.bin` contains:

- fixed header
- model hyperparameters
- tensor directory
- raw FP16 tensor blobs
- alignment padding

`tokenizer/` contains:

- tokenizer model or tokenizer JSON
- tokenizer config
- special token metadata

Header fields:

- magic
- version
- model family = `tinyllama_v1`
- hidden size
- layer count
- attention head count
- KV head count
- vocab size
- RoPE theta
- max sequence length used
- tensor count
- checksum

Tensor directory fields:

- tensor name
- dtype
- rank
- dimensions
- offset
- size
- checksum

Explicit non-goals for format v1:

- Q4
- Q8
- GGUF direct loading
- multiple model families
- direct mmap optimization
- paged KV cache
- GPU-specific packing

Format v1 is for correctness, not maximum speed.

## 8. Milestones

### Milestone 0: Freeze The Old World

Goal:

Create a stable checkpoint before rewriting.

Deliverables:

- record current CPU benchmark result
- record current QNN control-smoke result
- keep app build green
- mark QAIRT/QNN work as sidecar in docs
- write a baseline note if `.git` metadata is unavailable

Gate:

- `./gradlew :app:assembleDebug` succeeds
- at least one CPU benchmark command is documented
- at least one QNN control-smoke command or result is documented

Repository note:

This workspace currently may not have `.git` metadata. If git metadata is absent, do not block the reset on branch/tag commands. Record the baseline in `docs/custom_runtime_baseline.md` instead. If git metadata is present, create `custom-runtime-v1` and tag the old state as `pre-custom-runtime-reset`.

### Milestone 1: Correct Forward Pass On Owned Runtime

Goal:

Run `TinyLlama-1.1B-Chat-v1.0` end-to-end in owned C++ on Android CPU.

Scope:

- packed model loading
- tokenizer path
- embeddings
- RMSNorm
- RoPE
- Q/K/V/O projections
- attention
- MLP
- logits
- greedy next-token selection
- KV cache for decode

Gates:

- same prompt gives deterministic first token
- Android matches desktop oracle within tolerance at selected layer boundaries
- one short prompt produces readable text
- no dependency on QAIRT, ONNX, QNN, or GGUF in the runtime path

Explicit non-goals:

- speed
- multithreading tricks
- quantization
- Vulkan
- fancy sampling

### Milestone 2: Benchmark Harness Becomes Non-Negotiable

Goal:

Benchmark the owned runtime well enough that backend decisions are factual.

Fixed benchmark protocol:

- 3 prompts
- greedy decode
- max new tokens = `64`
- context length cap = `512`
- warmup runs = `1`
- measured runs = `5`
- report median

Every JSONL row must include:

- git commit when available, otherwise baseline id
- backend
- model checksum
- prompt id
- seed
- context length
- prompt length
- generated length
- load time ms
- TTFT ms
- prefill tok/s
- decode tok/s
- peak RSS MB
- thermal status samples

Gate:

- every benchmark writes one JSONL record
- records are comparable across future CPU/Vulkan/QNN runs
- adb script can trigger a run without manual UI steps

### Milestone 3: CPU Optimization Pass

Goal:

Make CPU strong enough to serve as a real baseline.

Work order:

1. FP16 correctness path
2. Q8 packed linear layers
3. FP16 vs Q8 comparison
4. thread count tuning
5. big-core affinity experiments
6. buffer reuse and allocation cleanup

Rules:

- no Q4 before Q8 is stable
- no GPU work before top CPU bottlenecks are measured

Gate:

- owned CPU runtime has stable decode numbers
- benchmark records are reproducible
- bottleneck report identifies the top hot kernels

### Milestone 4: Vulkan Spike

Goal:

Determine whether custom Adreno compute is worth deeper investment.

Minimum spike:

- Vulkan device/context setup in native code
- one fused decode-time `matvec` shader
- compare CPU vs Vulkan for one hot projection shape
- measure dispatch overhead separately from kernel time
- verify output against CPU within tolerance

Success criteria:

- Vulkan beats CPU for that exact shape, or
- Vulkan loses but the overhead reason is clearly measured, or
- Vulkan instability is reproducible enough to pause it decisively

Hard stop:

Two focused weeks maximum. If the spike does not produce a clear signal, return to CPU/packing work.

### Milestone 5: Mixed CPU/GPU Decode

Goal:

Accelerate decode without porting the whole model.

Port order:

1. output projection
2. MLP projections
3. Q/K/V projections
4. RMSNorm/RoPE only if profiling says they matter
5. attention inner path last

Gate:

- mixed CPU/Vulkan decode works end-to-end
- benchmark shows measurable improvement
- sustained thermal run shows the win survives beyond short bursts

### Milestone 6: Revisit QAIRT/HTP

Goal:

Turn HTP back into a benchmark question.

Questions:

- Can QAIRT/HTP beat owned CPU on the same model class?
- Can QAIRT/HTP beat owned Vulkan on the same benchmark protocol?
- Is QNN useful as a selected-op backend or only as a vendor full-graph path?

Rules:

- no HTP task resumes unless an owned baseline exists
- every HTP task must retire a specific benchmark unknown
- no HTP task is allowed unless it compares against the same prompt protocol

## 9. Revised First Two Weeks

### Week 1: Freeze And Skeleton

Day 1:

- record current build status
- record current CPU benchmark result
- record current QNN sidecar status
- create `custom-runtime-v1` branch and `pre-custom-runtime-reset` tag if `.git` exists
- if `.git` does not exist, write `docs/custom_runtime_baseline.md`
- pin exact TinyLlama source weights
- write `MODEL_DECISION.md`

Day 2:

- create `native/custom/`
- define `Tensor`, `Model`, `Runtime`, `KvCache`, and `BenchmarkResult`
- compile an empty custom target into app or as a host tool without replacing the current engine

Day 3:

- write `scripts/custom_runtime/pack_tinyllama.py`
- pack embeddings, `lm_head`, and layer 0 tensors first
- write packed metadata reader

Day 4:

- write `run_reference`
- load packed file on desktop
- verify metadata and tensor lookup

Day 5:

- implement and test RMSNorm, RoPE, and FP16 matvec
- add deterministic unit tests

Week 1 gate:

- packed file loads
- desktop reference runs selected layer ops
- one layer op matches expected output

### Week 2: First Real Decode Slice

Day 6:

- tokenizer path chosen and wired
- one prompt tokenizes identically on desktop and Android

Day 7:

- implement one full transformer layer on desktop
- dump intermediate tensors

Day 8:

- port the same full-layer path to Android CPU
- compare Android vs desktop tensor dumps

Day 9:

- connect multi-layer forward pass
- compute logits
- emit first next token

Day 10:

- connect decode loop
- generate 16-32 tokens
- write first JSONL benchmark record

Week 2 gate:

One short prompt must decode through the owned Android CPU runtime. First-token-only or partial-layer smoke does not satisfy this gate.

## 10. Decision Rules

Work is **mainline** only if it advances one of these:

- owned model loading
- owned inference correctness
- owned packed format
- owned benchmark harness
- measured CPU optimization
- measured Vulkan viability

Work is **sidecar** if it depends on:

- QAIRT builder acceptance
- ONNX graph repair
- QNN converter quirks
- Genie config compatibility
- architecture support in vendor tooling

Sidecar work is allowed only when:

- it preserves a known-good baseline
- it answers a bounded comparison question
- it is timeboxed

## 11. Kill Rules

Pause immediately if:

- a second model family enters discussion before Milestone 3
- GGUF runtime loading appears before the owned packer is stable
- Vulkan work starts before desktop-vs-Android tensor comparison exists
- QAIRT/ONNX/HTP consumes more than one day during Milestones 1-3

Continue immediately if:

- a task reduces uncertainty in model loading
- a task improves tensor-level correctness confidence
- a task improves reproducible benchmark quality

## 12. What To Preserve From The Existing Repo

Keep:

- Android app shell
- install/build flow
- JNI bridge pattern
- benchmark history JSONL idea
- adb benchmark worker scripts
- QAIRT SDK validation scripts
- docs capturing QNN evidence

Rewrite or isolate:

- engine code tightly coupled to `llama.cpp`
- backend selection logic tied to generic `ggml` assumptions
- UI controls that imply unsupported backend maturity
- model-prep paths that require vendor conversion before basic inference

The app can remain the harness. The engine under it can change aggressively.

## 13. Immediate Next Actions

1. Record baseline build and benchmark status.
2. Create `custom-runtime-v1` and tag `pre-custom-runtime-reset` if git metadata exists.
3. If git metadata is unavailable, write `docs/custom_runtime_baseline.md`.
4. Pin `TinyLlama-1.1B-Chat-v1.0`.
5. Add `MODEL_DECISION.md`.
6. Create `native/custom/`.
7. Write `scripts/custom_runtime/pack_tinyllama.py`.
8. Write `run_reference`.
9. Write `dump_reference_tensors`.
10. Add Week 1 unit tests.
11. Do not touch QAIRT/Vulkan until Week 1 gate passes.

## 14. Final Rule

**The engine owns the roadmap now. Vendor tooling is a comparison target, not the critical path.**
