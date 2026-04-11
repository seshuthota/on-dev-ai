# Strategic Reset: Custom Runtime Mainline

## Core Diagnosis

The project has drifted.

Original goal:

**Build phone-specific LLM inference for the iQOO 13 / Snapdragon 8 Elite, including custom kernels where useful, and measure real sustained performance.**

Current drift:

**Spend most engineering time making Qualcomm's model-prep stack accept model formats, architectures, quantization layouts, and builder-specific quirks.**

Those are related, but they are not the same project. QAIRT/QNN is valuable, and the current work has proven useful things: QNN runtime packaging works, app-side in-process QNN execution works, and HTP `v79` is reachable for a control path. But the main loop has become vendor-toolchain debugging instead of engine building.

The reset is:

**Make the custom runtime the mainline. Freeze QAIRT/QNN as a sidecar baseline and optional future backend.**

Do not delete the current work. It is still useful as a comparison target, setup reference, Android packaging reference, and evidence log. But it should stop deciding the project's critical path.

## Corrections To The Previous Thinking

Do not assume Vulkan will automatically beat CPU. Vulkan is the right low-level custom GPU surface to investigate on Android, but this repo already has evidence that GPU paths can disappoint or crash. Treat Vulkan as a measured backend spike, not a guaranteed win.

Do not assume HTP is irrelevant. HTP is still the most interesting accelerator in the device. The correction is narrower: HTP should not block the custom runtime until there is a working CPU reference and a benchmark harness strong enough to compare backends.

Do not chase "any model that compiles." Pick one boring dense decoder and own it end to end.

Do not build a general-purpose inference framework. The goal is a fast phone-specific engine for one model family first.

## Track Split

### Track A: Custom Runtime Mainline

Purpose:

Build an owned native inference engine with a small, testable scope.

Initial target:

- Android NDK C++ runtime
- one dense decoder-only model family
- one tokenizer path
- one owned packed weight format
- CPU reference backend
- Vulkan compute backend only after CPU correctness is proven
- benchmark harness with TTFT, prefill, decode, memory, and thermal metrics

Candidate model class:

- Prefer Llama-style or Qwen2.5-style dense decoder.
- Avoid Qwen3.5 for the first owned runtime pass because current local QAIRT evidence already shows architecture friction.
- Avoid multimodal, MoE, speculative decoding, and toolchain-specific graph formats until the core loop works.

### Track B: QAIRT/QNN Sidecar

Purpose:

Keep the existing Qualcomm path as a reference, not as the blocker.

Allowed work:

- preserve scripts that validate SDK and device state
- keep QNN control-model smoke tests working
- keep benchmark comparison data
- use QAIRT results to understand realistic HTP performance
- revisit HTP after the custom runtime has a stable benchmark harness

Disallowed as mainline work:

- ONNX surgery
- architecture-tag patching
- random model conversion experiments
- embedding LUT workaround loops
- QAIRT builder debugging without a direct benchmark question

## What To Preserve From Current Repo

Keep:

- Android app shell and install flow
- JNI bridge pattern
- benchmark history JSONL idea
- adb benchmark worker scripts
- QAIRT SDK validation scripts
- docs capturing current QNN evidence

Rewrite or isolate:

- native engine code that is tightly coupled to `llama.cpp`
- backend selection logic if it assumes generic `ggml` backends
- UI controls that imply unsupported backend maturity
- model preparation paths that require vendor conversion before basic inference can run

The current app can remain a harness. The engine under it can change aggressively.

## Architecture Target

New native layout:

```text
native/custom/
  include/
    runtime.h
    model.h
    tokenizer.h
    benchmark.h
  src/
    runtime.cpp
    model_loader.cpp
    tokenizer.cpp
    sampler.cpp
    kernels_cpu.cpp
    benchmark.cpp
  tools/
    pack_model.cpp
  vulkan/
    vk_context.cpp
    kernels/
      q4_matvec.comp
      rmsnorm.comp
      rope.comp
```

Runtime shape:

```text
Android UI -> ViewModel -> NativeBridge -> CustomRuntime
                                      -> CPU backend first
                                      -> Vulkan backend later
                                      -> QNN sidecar later
```

Model format:

- Do not run directly from arbitrary GGUF forever.
- Use GGUF or Hugging Face files as source input only.
- Convert once into an owned packed format.
- Include a small manifest with architecture, dimensions, quantization, tokenizer metadata, and checksums.

First packed format can be simple:

- header magic/version
- model hyperparameters
- tokenizer metadata reference
- tensor table
- contiguous tensor data
- alignment chosen for CPU/Vulkan kernels

Optimize format only after correctness and profiling show where layout matters.

## Milestones And Gates

### Milestone 0: Freeze The Current State

Goal:

Create a known baseline before rewriting.

Deliverables:

- record current CPU benchmark numbers
- record current QNN control-smoke status
- mark QAIRT/QNN docs as sidecar/reference
- make the app still build

Gate:

- `./gradlew :app:assembleDebug` succeeds
- at least one CPU benchmark command is documented with result location

### Milestone 1: Minimal CPU Reference Runtime

Goal:

Run one small dense decoder model in owned C++ on Android CPU.

Scope:

- model metadata loader
- tokenizer path
- RMSNorm
- RoPE
- Q/K/V/O projections
- MLP projections
- KV cache
- greedy or simple temperature sampling

Correctness gates:

- host-side smoke test can load metadata and tensors
- Android runtime loads the packed model
- first-token output is deterministic for a fixed prompt
- decode produces non-garbage text for a short prompt
- no dependency on QAIRT, ONNX, or QNN

Stop rule:

If this milestone expands beyond one model family or one quant format, narrow scope immediately.

### Milestone 2: Benchmark Harness Becomes The Product

Goal:

Measure the owned runtime well enough that backend decisions are factual.

Metrics:

- model load time
- TTFT
- prompt prefill tokens/sec
- decode tokens/sec
- peak RSS or Android memory info
- 3-minute sustained decode
- 5-minute sustained decode
- Android thermal status over time
- backend, thread count, context length, model checksum

Gate:

- every benchmark writes one JSONL record
- benchmark records are comparable across CPU, Vulkan, and sidecar QNN later
- scripts can run benchmarks over adb without manual UI steps

### Milestone 3: CPU Optimization Pass

Goal:

Make CPU good enough to be a meaningful baseline.

Work order:

1. Use simple float or Q8 path for correctness.
2. Add Q4/Q8 packed matvec.
3. Tune thread count and big-core affinity if needed.
4. Profile decode loop before touching GPU.

Gate:

- CPU owned runtime has stable decode numbers
- bottleneck report identifies top kernels by time
- no GPU work starts until the CPU bottleneck is measured

### Milestone 4: Vulkan Spike

Goal:

Prove whether custom Adreno compute is worth deeper investment.

Minimum spike:

- Vulkan device/context setup in native code
- one fused dequant + matvec shader
- compare CPU vs Vulkan for one projection shape
- measure dispatch overhead separately from kernel time
- verify output against CPU within tolerance

Gate:

- Vulkan beats CPU for at least one hot kernel shape, or the measured overhead explains why it does not
- no full-model Vulkan port until this gate passes

Stop rule:

If Vulkan setup or driver behavior blocks progress for more than two focused weeks, return to CPU/packing work and keep Vulkan as an experiment.

### Milestone 5: Partial GPU Decode Path

Goal:

Accelerate the decode loop without porting everything.

Port order:

1. fused dequant + matvec
2. MLP projections
3. attention projections
4. RMSNorm/RoPE only if profiling says they matter
5. KV attention path only after projection kernels are stable

Gate:

- full prompt decode works with mixed CPU/Vulkan execution
- benchmark shows improvement in either prefill or decode
- sustained thermal run proves the win survives beyond a short burst

### Milestone 6: Revisit HTP/QNN

Goal:

Turn HTP back into a benchmark question.

Questions:

- Can QAIRT/HTP beat owned CPU on the same model class?
- Can QAIRT/HTP beat owned Vulkan on the same metric?
- Is QNN useful as a selected-op backend, or only as a full vendor graph path?

Gate:

- no HTP work resumes unless it is comparing against an owned baseline
- every HTP task must produce benchmark evidence or retire a specific unknown

## First Two-Week Execution Plan

### Week 1: Freeze And Skeleton

Day 1:

- capture current build status
- capture current CPU benchmark command and result
- move QAIRT/QNN docs mentally and explicitly into sidecar status
- choose the first model family and exact local model file

Day 2:

- create `native/custom/`
- define `Runtime`, `Model`, `Tensor`, and `BenchmarkResult` interfaces
- add a native build target that compiles but does not replace the current engine yet

Day 3:

- implement host-side or native metadata inspection for the chosen model source
- decide the first packed file schema
- write a tiny packer that emits header + tensor table + one or two tensors

Day 4:

- implement packed file reader in C++
- add Android JNI call to load packed metadata
- surface metadata in logcat or simple UI diagnostic

Day 5:

- implement first CPU kernels needed for a single layer smoke
- add deterministic test input
- verify CPU kernel output against a simple reference path

Week 1 gate:

- app builds
- custom runtime target loads a packed file
- one kernel has correctness validation

### Week 2: First Real Decode Slice

Day 6:

- complete tokenizer path decision
- implement tokenization for one prompt
- add sampling stub

Day 7:

- implement RMSNorm and RoPE
- validate shapes and intermediate buffers

Day 8:

- implement attention projections and KV cache for one layer
- prioritize clarity over speed

Day 9:

- implement MLP path
- run one full layer
- compare host/native outputs if possible

Day 10:

- connect full model loop for a tiny prompt
- emit benchmark JSONL for load time, TTFT, and decode speed
- document what is still incorrect or slow

Week 2 gate:

- either one prompt decodes through the owned runtime, or there is a precise blocker with the next smallest step identified

## Decision Rules

Work is mainline only if it moves one of these forward:

- owned model loading
- owned inference correctness
- owned packed format
- owned benchmark harness
- measured CPU optimization
- measured Vulkan kernel spike

Work is sidecar if it depends on:

- QAIRT model builder acceptance
- ONNX graph repair
- QNN converter quirks
- Genie config compatibility
- architecture support in vendor tooling

Sidecar work is allowed only when:

- it preserves a known-good baseline
- it answers a specific comparison question
- it takes less than a bounded timebox

## Practical Success Definition

Short term:

**A small dense decoder runs in the owned runtime on the phone and produces benchmark records.**

Medium term:

**A custom CPU path is correct and profiled, and a Vulkan matvec spike gives a factual yes/no on GPU acceleration.**

Long term:

**The engine owns its model format, hot kernels, benchmark harness, and backend decisions. QAIRT/QNN becomes an optional competitor, not the boss of the roadmap.**

## One-Sentence Rule

**Stop trying to make every model fit the phone through vendor tooling. Make one model fit the engine, then make the engine fit the phone.**
