# Strategic Reset v3: llama.cpp Mainline, Device-Specific Optimization

## Purpose

This document resets the project back to its original objective:

**Build phone-specific LLM inference for iQOO 13 / Snapdragon 8 Elite, including custom kernels where useful, and measure real sustained performance.**

The QAIRT/QNN path has already proven useful things:

- QNN runtime packaging works.
- App-side in-process QNN execution works.
- HTP `v79` is reachable for a control path.
- Real LLM generation via Genie has been demonstrated.

Those results are valuable, but they do not define the mainline anymore.

The reset is now:

**Make `llama.cpp` the production inference base. Use the owned runtime as a correctness oracle, benchmark harness, and kernel lab. Freeze QAIRT/QNN as a sidecar baseline and optional future backend.**

The goal is no longer to recreate a worse `llama.cpp` from scratch, make every model fit Qualcomm tooling, or keep fighting ONNX builder edge cases before performance evidence exists.

The goal is now to run real models through `llama.cpp`, benchmark them rigorously on Snapdragon 8 Elite, then push beyond stock `llama.cpp` with device-specific tuning and kernels where profiling proves the win.

## Core Rule

**Start from the strongest working engine, then make that engine fit the phone. Every optimization must beat the same-protocol Android baseline.**

## 1. Frozen Decisions

These decisions are fixed for Milestones 0-3.

### First Model

`TinyLlama-1.1B-Chat-v1.0`

Reason:

- dense decoder-only Llama-style architecture
- easier first production/lab target than Qwen3.5 or multimodal models
- enough scale to matter, but small enough for first bring-up
- tokenizer/config/weights are easier to inspect than vendor-prepared artifacts

### Production Runtime Format

GGUF for `llama.cpp` production runs.

Reason:

- aligns with the production inference base
- gives immediate access to mature quantization formats
- avoids blocking performance work on owned loader completeness

Production artifacts:

- `.gguf` model files for `llama.cpp`
- recorded quantization format (`Q4_K_M` or best available Q4 as primary; `Q5_K`/`Q6_K` fallback tiers if Q4 quality is unacceptable; `Q8_0` and `F16` as quality/perf comparison reference only)
- checksum and conversion command in benchmark metadata

### Experimental Runtime Format

Owned packed **FP16** format remains useful for oracle, kernel experiments, and reference correctness only.

Reason:

- current owned runtime already provides tensor-level confidence
- useful for isolated kernel tests without full `llama.cpp` complexity
- not the main performance target; Q4 GGUF is the primary mobile production format
- not the production inference path unless it beats `llama.cpp` Q4

### Fixed Runtime Constraints

- context length: `512`
- decode mode: greedy only
- benchmark prompts: three prompts checked into the repo
- mainline backend through Milestone 3: `llama.cpp` Android CPU only

## 2. Project Split

### Track A: llama.cpp Production Mainline

Purpose:

- run real LLM inference through `llama.cpp`
- establish the strongest CPU baseline on Snapdragon 8 Elite
- tune model format, quantization, thread count, affinity, and thermal behavior
- patch or extend kernels only when profiling proves a device-specific win

Scope:

- Android NDK C++
- pinned `llama.cpp` revision
- GGUF model artifacts
- same JSONL benchmark protocol as the owned runtime
- CPU first; Vulkan/QNN only after CPU baseline is understood

### Track B: Owned Runtime Kernel Lab

Purpose:

- preserve the working custom runtime as a correctness oracle
- test isolated kernels and packing layouts without disturbing production inference
- compare simplified kernels against `llama.cpp` kernels
- provide tensor-level debugging when `llama.cpp` output or performance is unclear

Allowed work:

- keep tokenizer/model packer/runtime tests green
- keep Android tensor and decode smoke paths available
- add microbenchmarks for hot projection shapes
- prototype Q8/NEON/Vulkan kernels in isolation

Disallowed as mainline work:

- trying to reimplement full `llama.cpp` before evidence says it will win
- expanding owned runtime architecture support
- making owned packed format the default product path without beating `llama.cpp`

### Track C: QAIRT/QNN Sidecar

Purpose:

- preserve current Qualcomm work as a baseline and comparison target
- keep control-model validation alive
- revisit HTP only after the `llama.cpp` Android baseline exists

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

Out of scope until the `llama.cpp` Android baseline is measured:

- multiple model families
- ONNX runtime loading
- QAIRT/HTP as a mainline backend
- speculative decoding
- paged KV cache
- multimodal support
- tool calling
- framework-style abstraction for many architectures
- owned-runtime feature expansion
- custom kernels without a profile proving the target hotspot
- laptop-driven performance decisions

## 4. Success Definition

Short term:

**`llama.cpp` runs the target model on Android and produces same-protocol benchmark records.**

Medium term:

**Stock `llama.cpp` is profiled and tuned on Snapdragon 8 Elite well enough to identify exact bottlenecks and optimization candidates.**

Long term:

**The project beats stock `llama.cpp` on the fixed target phone through measured device-specific changes, while retaining a clean benchmark trail and fallback production path.**

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
  benchmark_llama_cpp_android.sh
  compare_benchmarks.py
native/third_party/
  llama.cpp/   # pinned production inference base
```

Runtime shape:

```text
Android UI -> ViewModel -> NativeBridge -> llama.cpp runtime
                                      -> tuned CPU path first
                                      -> custom kernel hooks later
                                      -> Vulkan/QNN sidecars later

Owned custom runtime -> oracle, tensor dumps, microbenchmarks, kernel lab
```

Important rule:

- `layers.cpp` may be explicit and ugly in v1.
- Do not over-abstract model execution early.
- Production inference uses GGUF through `llama.cpp`.
- The owned runtime reads only the owned packed format and should stay focused on oracle/kernel-lab work.

## 6. Mandatory Oracle And Lab Path

The owned runtime remains useful as a tensor oracle and isolated kernel lab, even though production inference now uses GGUF through `llama.cpp`.

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
- GGUF direct loading inside the owned runtime
- multiple model families
- direct mmap optimization
- paged KV cache
- GPU-specific packing

Format v1 is for correctness and kernel experiments, not production inference or maximum speed.

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

### Milestone 1: Correct Forward Pass On Owned Runtime Lab

Goal:

Run `TinyLlama-1.1B-Chat-v1.0` end-to-end in owned C++ on Android CPU so it can serve as an oracle and kernel lab.

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
- no dependency on QAIRT, ONNX, QNN, or GGUF in the owned-runtime lab path

Explicit non-goals:

- speed
- multithreading tricks
- quantization
- Vulkan
- fancy sampling

### Milestone 2: Benchmark Harness Becomes Non-Negotiable

Goal:

Benchmark `llama.cpp`, the owned runtime, and future backends with one comparable protocol.

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
- records are comparable across owned CPU, `llama.cpp`, future Vulkan, and QNN runs
- adb script can trigger a run without manual UI steps
- `llama.cpp` has at least one Android benchmark record on the same prompt protocol

### Milestone 2.5: llama.cpp Production Baseline

Goal:

Establish stock `llama.cpp` as the production baseline before any custom optimization work.

Scope:

- pin the exact `llama.cpp` revision used for comparison
- document Android build flags, quantization type, thread count, and model artifact
- run the fixed benchmark protocol on Snapdragon 8 Elite
- emit JSONL records with backend = `llama_cpp`
- produce a comparison table: stock `llama.cpp`, owned FP16 CPU, and any existing QNN sidecar records

Rules:

- use the same prompt IDs, max tokens, warmup count, measured runs, and median reporting
- record model/checksum/quantization format so results are not ambiguous
- tune only against Android numbers on the target phone
- do not start hardware-specific kernel work until this baseline exists
- production inference defaults to `llama.cpp` unless an alternative backend beats it with the same protocol

Gate:

- `llama.cpp` Android decode benchmark exists
- owned runtime Android decode benchmark exists
- both records can be compared by one script without manual editing

### Milestone 3: llama.cpp CPU Optimization Pass

Goal:

Push stock `llama.cpp` closer to the Snapdragon 8 Elite limit before writing separate kernels.

Work order:

1. Build and record stock `llama.cpp` Android baseline.
2. Test quantization formats: Q4 first (`Q4_K_M` preferred when available), then Q5/Q6 fallback tiers if Q4 quality is unacceptable, then Q8/F16 as quality/perf comparison reference.
3. Sweep thread count and batch/prompt parameters.
4. Run big-core affinity and thermal stability experiments.
5. Profile the limiting kernels in stock `llama.cpp`.
6. Patch or configure `llama.cpp` only where profiling identifies a clear bottleneck.
7. Compare patched `llama.cpp` vs stock `llama.cpp` with the same JSONL protocol.
8. Use the owned runtime only to prototype isolated kernels or verify math when useful.

Rules:

- no fork patch survives unless it beats stock `llama.cpp` on Android
- no GPU work before top CPU bottlenecks are measured
- no custom kernel work before stock `llama.cpp` profiling identifies the target
- keep fork changes small enough to rebase

Gate:

- stock and patched `llama.cpp` have stable decode numbers
- benchmark records are reproducible
- bottleneck report identifies the top hot kernels
- comparison report says where patched `llama.cpp` wins, loses, and why
- any custom kernel idea has attribution and a measured reason to exist

### Milestone 4: Vulkan Spike

Goal:

Determine whether custom Adreno compute can beat tuned `llama.cpp` CPU for measured hotspots.

Minimum spike:

- Vulkan device/context setup in native code
- one fused decode-time `matvec` shader
- compare tuned `llama.cpp` CPU vs Vulkan for one hot projection shape
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

Accelerate `llama.cpp`-based decode without porting the whole model.

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

- Can QAIRT/HTP beat tuned `llama.cpp` CPU on the same model class?
- Can QAIRT/HTP beat tuned `llama.cpp` + custom kernels on the same benchmark protocol?
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

- `llama.cpp` Android production inference
- same-protocol benchmark records
- stock vs patched `llama.cpp` comparison
- measured Snapdragon 8 Elite CPU optimization
- measured kernel work that improves `llama.cpp` or a selected backend
- measured Vulkan viability against tuned `llama.cpp`

Work is **sidecar** if it depends on:

- QAIRT builder acceptance
- ONNX graph repair
- QNN converter quirks
- Genie config compatibility
- architecture support in vendor tooling
- owned-runtime expansion without a benchmark or kernel-learning purpose

Sidecar work is allowed only when:

- it preserves a known-good baseline
- it answers a bounded comparison question
- it is timeboxed

Work is **lab-track** when it uses the owned runtime to answer:

- whether an isolated kernel is mathematically correct
- whether a packing/layout idea is worth porting into `llama.cpp`
- whether a backend experiment is promising enough to benchmark end-to-end

## 11. Kill Rules

Pause immediately if:

- a second model family enters discussion before stock `llama.cpp` baseline exists
- owned-runtime work expands beyond oracle/kernel-lab tasks without beating `llama.cpp`
- Vulkan work starts before tuned `llama.cpp` CPU bottlenecks are measured
- QAIRT/ONNX/HTP consumes more than one day before the `llama.cpp` baseline exists
- code is copied across projects without checking license/attribution and measuring the win

Continue immediately if:

- a task makes `llama.cpp` faster on the target phone
- a task improves reproducible benchmark quality
- a task makes stock and patched `llama.cpp` results directly comparable on Android
- a task isolates a proven hotspot and produces a measurable device-specific result

## 12. What To Preserve From The Existing Repo

Keep:

- Android app shell
- install/build flow
- JNI bridge pattern
- benchmark history JSONL idea
- adb benchmark worker scripts
- QAIRT SDK validation scripts
- docs capturing QNN evidence
- pinned `llama.cpp` source and Android benchmark runner
- comparison scripts that read both owned runtime and `llama.cpp` JSONL records
- owned runtime artifacts that support oracle/kernel-lab work

Rewrite or isolate:

- old app paths that do not expose `llama.cpp` as the production inference engine
- backend selection logic tied to generic `ggml` assumptions
- UI controls that imply unsupported backend maturity
- model-prep paths that require vendor conversion before basic inference

Optimize deliberately:

- `llama.cpp` Android build flags
- GGUF quantization choice
- thread count and big-core affinity
- batch/context/prompt settings
- hot ARM/NEON kernels
- cache-friendly row/block packing where profiling proves benefit

Do not optimize blindly:

- owned-runtime reimplementation for its own sake
- broad architecture support
- laptop-only speedups
- changes that improve one prompt but break the fixed benchmark protocol

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

## 13.1 Current Next Actions After Week 2

1. Commit the Week 2 Android decode and Week 3 profiler fixes.
2. Pin the current `llama.cpp` revision under `native/third_party/llama.cpp` or document the existing pinned revision.
3. Build a minimal Android `llama.cpp` benchmark runner.
4. Make it emit the same JSONL fields as `run_forward`.
5. Run the fixed three-prompt protocol on Snapdragon 8 Elite with stock `llama.cpp`.
6. Sweep quantization, thread count, affinity, and thermal settings.
7. Write a comparison report: stock `llama.cpp`, tuned `llama.cpp`, owned FP16 CPU, and QNN sidecar where available.
8. Move custom-kernel work into the owned runtime only when a `llama.cpp` profile identifies the exact hotspot.

## 14. Final Rule

**The benchmark owns the roadmap now. `llama.cpp` is the production base; custom kernels, Vulkan, and QNN must earn their place by beating it on the target phone.**
