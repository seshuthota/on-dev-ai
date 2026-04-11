# OnDevAI Current Execution Plan (2026-04-10)

## Status

This is the active plan for the repo.

Use this file for current execution decisions.

Latest implementation checkpoint (`2026-04-10`):

- path-separation scripts are in place (`genie_cpu_real_llm` vs `genie_htp_real_llm`)
- host-side context-binary tooling is in place (`generate_qnn_context_bins.sh`, `build_qnn_sample_ctx_bin.sh`)
- QAIRT Python HTP builder pipeline is now scripted:
  - `scripts/build_genai_llm_container_htp.py`
  - `scripts/build_genie_htp_real_llm_assets.sh`
  - `scripts/setup_qairt_python_env.sh`
- ONNX repair helpers are now scripted for GGUF-derived exports:
  - `scripts/fix_onnx_reduce_axes.py`
  - `scripts/fix_onnx_constant_valueinfo_shapes.py`
- next critical step is running one complete custom-model build (`GGUF/exports -> container -> ctx-bins`) and validating on-device realization logs

Current blocker status (`2026-04-10`):

- `qwen3.5` GGUF path blocked at architecture support (`qwen35` unsupported by current QAIRT GGUF builder)
- supported GGUF (`qwen2.5`, `smollm2`) reaches deeper but still blocked in QAIRT HTP build:
  - large-model path can hit protobuf serialization limits in AR/CL shape-infer
  - small-model path reaches conversion but fails on quantization/encoding interactions
- ONNX is now a gated research lane, not the primary HTP bring-up path:
  - Gemma ONNX got past operator/AR-CL fixes but failed deeper in QAIRT converter shape inference
  - Qwen2.5 GGUF-derived ONNX is rejected early due large external-data/protobuf risk
- GGUF is the primary custom-model lane again, but candidates must pass architecture support checks before expensive HTP build attempts

Older plan files remain valuable, but they are now historical/reference documents:

- `plan.md`: original master plan and baseline performance history
- `cpu_performance_fix_plan.md`: completed CPU recovery work
- `qnn_feasibility_plan.md`: QNN feasibility and bring-up record
- `suplementary-plan.md`: external idea document; not the literal implementation path
- `docs/progress_deep_dive_2026-04-10.md`: full historical narrative and implementation status

## 1. Current Objective

Get a real LLM running on Snapdragon 8 Elite HTP/NPU with a measurable win over the current CPU baseline.

That means:

1. prove a real LLM path that actually realizes on `QNN_HTP` rather than `QNN_CPU`
2. benchmark it with the same discipline already used for CPU/OpenCL
3. only then decide whether QNN/HTP is the winning product path

## 2. What Is Already Proven

These points are no longer open questions:

- CPU baseline is real and strong enough to beat
- OpenCL works but is slower than CPU on the current model/backend mix
- Vulkan is not the near-term path because of vendor driver crashes
- QNN runtime packaging works on-device
- app-side in-process QNN execution works
- HTP `v79` works on this device for the control-model path
- `ADSP_LIBRARY_PATH` / app runtime setup issues were fixed
- real LLM generation on-device is already working through Genie

This means the main blocker is no longer Android app sandboxing or basic QNN runtime access.

## 3. Current Best Technical Reading

The repo currently has two different kinds of success:

1. `qnn_control_htp`
   - proven in-process control-model execution on `htp-v79`
   - this proves the device/runtime path can reach HTP

2. `genie_cpu_real_llm`
   - proven real LLM generation path using Genie
   - current observed realization is `QNN_CPU`

The gap is between those two states.

The strongest current hypothesis is:

- the real-LLM artifact/config path is not yet the correct HTP-targeted workflow
- the current TinyLlama runner uses `QnnGenAiTransformer` library-style config
- documented Genie HTP flows use `QnnHtp` with compiled context binaries (`ctx-bins`)

So the next step is not more random tuning inside the current TinyLlama composer flow.

The next step is to build a clean, explicit `genie_htp_real_llm` path.

## 4. Paths We Are Not Actively Pursuing

These paths are not the active path right now:

- more OpenCL tuning as the main strategy
- Vulkan bring-up
- direct migration to `llama-cpp-qnn-builder`
- speculative decoding work before HTP realization is proven
- app-side QNN chat integration before host-side HTP LLM preparation is proven

These are not forbidden forever. They are simply not on the current critical path.

## 5. Canonical Execution Tracks

### Track A: Preserve Known-Good States

Objective:

- keep existing validated paths runnable and clearly named

Required repo states:

- `qnn_control_htp`: known-good HTP control-model path
- `genie_cpu_real_llm`: current TinyLlama-style Genie path
- `genie_htp_real_llm`: new target path to be created

Current script mapping:

- `scripts/run_genie_cpu_real_llm_android.sh`: current CPU-realized real-LLM Genie path
- `scripts/run_genie_htp_real_llm_android.sh`: HTP-targeted real-LLM Genie path
- `scripts/run_genie_t2t_android.sh`: compatibility wrapper only
- `scripts/generate_qnn_context_bins.sh`: generic host context-binary generator
- `scripts/build_qnn_sample_ctx_bin.sh`: reproducible sample host pipeline that generates HTP `ctx-bins`

Concrete repo actions:

1. Make naming in scripts/docs explicit so current Genie success is not mentally confused with HTP success.
2. Keep current TinyLlama CPU-realized runner intact as a comparison baseline.
3. Do not mutate the existing control-model path while HTP real-LLM bring-up is still unresolved.

Exit criteria:

- each path has a clear name, purpose, and expected backend realization

### Track B: Build A Known-Good Official Genie HTP Reference Path

Objective:

- prove one real LLM-style Genie HTP path using the documented `QnnHtp` + `ctx-bins` style

Why this matters:

- if this works on-device, the runtime/device layer is cleared again for real LLMs
- if this fails, the blocker is still below model preparation

Concrete tasks:

1. Identify one QAIRT 2.45 example config closest to our target:
   - start with `examples/Genie/configs/llama2-7b/llama2-7b-htp.json`
   - inspect any required preprocessing / context-binary generation steps

2. Build a host-side note or script that documents the exact artifact set required for:
   - `QnnHtp`
   - `binary.ctx-bins`
   - `htp_backend_ext_config.json`
   - tokenizer + runtime libs

3. Create a dedicated runner script for the reference HTP flow instead of overloading the current `run_genie_t2t_android.sh`.

4. Verify backend realization on-device using logs, not assumption.

Success criteria:

- real Genie run starts with `QnnHtp`
- config uses `model.type = "binary"` with `ctx-bins`
- logs or runtime evidence show HTP realization rather than CPU fallback

Kill criteria:

- if the official-style HTP path cannot be made to initialize on this device despite the already-proven control-model HTP path, stop assuming model prep is the only blocker and re-open runtime/config investigation

### Track C: Evaluate QAIRT GGUF Builder As The Primary Custom-Model Path

Objective:

- move closer to the project's natural GGUF workflow rather than staying trapped in the composer-only route

Why this is promising:

- the pinned SDK already contains `qti.aisw.converters.gguf_builder`
- its code explicitly contains HTP-oriented graph transformations

Concrete tasks:

1. Inventory the local GGUF builder entrypoints and expected host workflow.
2. Pick one small compatible GGUF model first.
3. Generate artifacts in a separate workspace under `.artifacts/gguf-builder/`.
4. Determine whether the builder output feeds:
   - direct QNN compile to context binaries
   - Genie HTP config
   - or an intermediate ONNX/encoding path that still needs compile steps

Candidate matrix (`2026-04-11`):

| Candidate | Suggested GGUF file | Why try it | Risk label | First action |
| --- | --- | --- | --- | --- |
| `unsloth/Qwen3.5-0.8B-GGUF` | `Qwen3.5-0.8B-Q4_0.gguf` or `Qwen3.5-0.8B-Q4_K_M.gguf` | smallest Qwen3.5 candidate, fast iteration, close to target family | high: likely `qwen35` / hybrid architecture unsupported by current QAIRT GenAI HTP builder | download one small quant and run GGUF intake/build smoke |
| `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-Q4_0.gguf` or `Qwen3.5-2B-Q4_K_M.gguf` | closer to current CPU baseline and intended use case | high: same Qwen3.5 architecture risk; larger compile/test cost | try only after 0.8B tells us whether Qwen3.5 architecture is accepted |
| existing `/home/curious/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | local file | architecture should align with `Qwen2ForCausalLM`, already available | medium: previous path hit protobuf/AR-CL issues after GGUF export | use as supported-family control for GGUF lane |
| existing `/home/curious/models/SmolLM2-135M-Instruct-Q4_K_M.gguf` | local file | tiny compile-time probe | medium/high: previous conversion/encoding issues | keep as converter debugging probe |

Important local SDK constraint:

- `GenAIBuilderFactory` preconfigures `Qwen2ForCausalLM`, not `Qwen3.5`-specific architectures.
- The older composer config helper has explicit entries for `Qwen2ForCausalLM` and `Qwen3ForCausalLM`, but not a proven `Qwen3.5` path in our current execution evidence.
- Therefore Qwen3.5 GGUF should be tested as a candidate, not assumed to be the mainline until it passes the GGUF architecture/build smoke.

Latest candidate outcomes (`2026-04-11`):

- downloaded and smoke-tested:
  - `/home/curious/models/Qwen3.5-0.8B-Q4_0.gguf`
  - `/home/curious/models/Qwen3.5-2B-Q4_0.gguf`
- both are rejected by current QAIRT GGUF builder architecture gate:
  - `Architecture qwen35 not supported`
- intake/wrapper now fail fast on this class before full conversion:
  - `scripts/model_intake_report.py` parses `gguf_architecture`
  - `scripts/build_genie_htp_real_llm_assets.sh` blocks when `qairt_gguf_architecture_unsupported=1`

Immediate consequence:

- keep both Qwen3.5 GGUF entries as tracked candidates, but move them to "blocked by architecture support" status until SDK support path changes

Additional GGUF smoke results (`2026-04-11`):

- `qwen2.5-0.5b-instruct-q4_k_m.gguf`:
  - architecture intake passes (`qwen2`)
  - build still fails in AR/CL shape-infer with protobuf serialization (`EncodeError: Failed to serialize proto`)
- `SmolLM2-135M-Instruct-Q4_K_M.gguf`:
  - architecture intake passes (`llama`)
  - default path fails at embedding LUT encoding lookup (`cannot find tensor encodings /model/embed_tokens/Gather/output_0`)
  - with `--embedding-lut off`, conversion proceeds further but quantization currently fails:
    - no calibration: float-fallback/quantization-overrides conflict
    - generated calibration list: netrun input mismatch (`Graph contains 1 inputs, but only found input data for 64 inputs`)

Practical meaning right now:

- we have no fully green GGUF->HTP container path yet in this SDK state
- current critical blocker is now quantization/calibration workflow compatibility for GGUF-derived artifacts (after architecture gating)

Success criteria:

- we can explain the artifact chain end to end
- we can produce a non-ambiguous HTP-targeted output path

Kill criteria:

- if the builder path is incomplete or blocked in QAIRT 2.45 for our model class, keep it as a follow-up path and return to the official Genie HTP reference path first
- if Qwen3.5 0.8B fails at unsupported architecture before graph conversion, do not spend time on Qwen3.5 2B until we have a supported Qwen3.5-specific builder path

### Track D: Only After HTP Realization, Benchmark Ruthlessly

Objective:

- decide whether HTP is actually better than the current CPU baseline

Benchmark requirements:

- TTFT
- decode tok/s
- repeat count with median rollups
- sustained run behavior
- thermal drift notes

Concrete tasks:

1. Reuse the existing benchmark discipline where possible.
2. Record exact artifact, config, and backend realization for each run.
3. Compare against current CPU baseline, not marketing expectations.

Success criteria:

- HTP path is reproducible
- HTP is measurably better than CPU on the same device for the same workload

Kill criteria:

- if HTP realization works but does not beat CPU meaningfully, do not force it into the product path

## 6. Immediate Next Experiments

These are the next experiments in order.

### Experiment 1: Freeze Current Genie Path As CPU-Realized Reference

Deliverables:

- rename/document current TinyLlama runner as `genie_cpu_real_llm`
- stop implying that `--backend htp-v79` in the current script means true HTP realization

Expected output:

- honest baseline for later comparison

### Experiment 2: Create A Separate `genie_htp_real_llm` Runner

Deliverables:

- a new runner/config path based on `QnnHtp`
- `model.type = "binary"`
- `ctx-bins` support
- dedicated HTP backend extension config

Expected output:

- clean separation between current CPU-realized flow and target HTP flow

### Experiment 3: Reproduce One Official-Style HTP Example

Deliverables:

- working host preparation notes or script
- generated context binaries
- successful on-device launch

Expected output:

- yes/no proof that real Genie HTP LLM execution is achievable on this phone with the pinned SDK

### Experiment 4: Evaluate GGUF Builder Against The Same Target

Deliverables:

- documented artifact chain from GGUF input to HTP-targeted runtime artifacts
- one trial conversion for a small supported GGUF

Expected output:

- decision whether GGUF builder becomes the primary custom-model path

## 7. Decision Tree

### Case A: Official-style HTP Genie path works

Interpretation:

- runtime/device path is cleared
- current TinyLlama composer flow is the wrong or incomplete preparation path for HTP

Action:

- move custom model work onto the HTP-targeted artifact path

### Case B: Official-style HTP Genie path fails before real execution

Interpretation:

- there is still a lower-level runtime/config gap for real Genie HTP flows

Action:

- compare against control-model HTP path
- inspect backend ext config, required libs, and context-binary expectations

### Case C: HTP path works but performance is not materially better than CPU

Interpretation:

- QNN/HTP is technically viable but not product-winning for the current model class

Action:

- stop treating HTP as the default direction
- re-evaluate CPU-first or alternative optimized paths

## 8. Repo Hygiene Rules For This Phase

1. No new plan documents should be added unless they replace this file.
2. New experiments should be mapped into Track A/B/C/D above.
3. Historical write-ups should remain, but this file is the canonical active plan.
4. Script naming must distinguish:
   - control-model HTP validation
   - real-LLM CPU-realized Genie
   - real-LLM HTP-targeted Genie

## 9. Bottom Line

The project is no longer trying to answer "can QNN work on this phone?"

That question is already answered.

The active question is:

- can we produce the right real-LLM artifacts and config so Genie/QNN actually realizes on HTP `v79`
- and if yes, does that path beat the already-strong CPU baseline enough to matter
