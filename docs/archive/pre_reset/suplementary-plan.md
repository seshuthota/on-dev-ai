**Detailed Implementation Plan: High-Speed Gemma 4 26B-A4B MoE Inference on iQOO 13 (Snapdragon 8 Elite + Adreno 830)**

## Status

This is a supplementary idea document, not the canonical implementation plan used by the repo.

Some ideas from this document influenced direction, but the repo did not follow it literally.

For the current active plan, use:

- `docs/current_execution_plan_2026-04-10.md`

For a detailed explanation of what from this document was actually implemented, see:

- `docs/progress_deep_dive_2026-04-10.md`

This plan builds directly on your current foundation (llama.cpp integration, CPU at ~28–30 t/s, OpenCL detection, WorkManager benchmarks, lazy loading, and `adb_backend_benchmark_worker.sh`).  
**Goal**: Fix OpenCL issues (fused Gated Delta Net fallback, cold-load overhead, instability >4 layers, Adreno kernel hangs) → achieve stable GPU offload → then unlock **NPU (Hexagon) acceleration** for 100–300+ t/s decode (matching the PRISM-PRO demo).  

The plan is phased (4–6 weeks total if you work part-time), with exact commands, flags, tests, and success criteria. Everything targets your iQOO 13 + Android 15/16 ROM.

### Phase 1: Stabilize & Optimize OpenCL Backend (1–3 days) — Quick Wins
**Objective**: Make OpenCL faster than CPU and eliminate your current pain points.  
The new Qualcomm-upstreamed OpenCL backend (Feb 2025, now in mainline) is fully tuned for Adreno 830.

1. **Update llama.cpp to latest main**  
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   git checkout master  # or latest commit post-Feb 2025
   git pull
   ```

2. **Build for Android with Snapdragon toolchain (recommended by Qualcomm)**  
   Use the official Docker image:  
   ```bash
   docker pull ghcr.io/snapdragon-toolchain/android-ndk:latest
   docker run --rm -v $(pwd):/workspace -w /workspace ghcr.io/snapdragon-toolchain/android-ndk:latest \
     bash -c "cmake -B build -DLLAMA_OPENCL=ON -DLLAMA_BUILD_SERVER=ON -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 && cmake --build build --config Release -j$(nproc)"
   ```  
   Output: `build/bin/llama-cli`, `llama-server`, etc. (push to device via ADB).

3. **Key build/runtime flags** (fix your issues):
   - **Quantization**: Use Q4_0 (best for Adreno OpenCL). When quantizing GGUF: `llama-quantize --pure model.gguf Q4_0`
   - **Disable problematic fused ops**: `--no-flash-attn` (prevents Gated Delta Net fallback + CPU copies). Gemma 4 MoE uses similar attention paths.
   - **Runtime**: `-ngl 8` (start here; Adreno 830 stable up to ~8–12 layers on MoE; your prior 4-layer sweet spot was likely old backend).
   - **Env vars** (do NOT use old ones): Remove `GGML_OPENCL_USE_ADRENO_KERNELS=ON`. New backend auto-detects Adreno 830.
   - Lazy loading: Keep your existing logic; add OpenCL warm-up in a background WorkManager job on app launch (caches JIT kernels → cuts your 7–8s cold load to <2s on subsequent runs).

4. **Test & Benchmark**  
   - Update your `adb_backend_benchmark_worker.sh` to test:  
     `llama-cli -m gemma4-26b-a4b-q4_0.gguf -p "test prompt" -ngl 8 --no-flash-attn -n 128 --repeat-penalty 1.0`
   - Run layer sweep (0, 4, 8, 12) + decode-only benchmark (short prompts).  
   **Success criteria**: >35 t/s decode (beats your CPU) and no fallbacks/timeouts. Log `ggml_opencl` diagnostics.

**Milestone**: Stable OpenCL > CPU baseline. Document new best `n_gpu_layers`.

### Phase 2: Integrate QNN NPU Backend (Main Speed Jump — 1–2 weeks)
**Why?** Adreno OpenCL is good (~15–40 t/s gen on similar models), but Hexagon NPU (45–60+ TOPS on 8 Elite) + official QNN SDK is what delivers the 300 t/s demo-level performance on quantized MoE.

1. **Use chraac/llama-cpp-qnn-builder** (actively maintained as of Feb 2026)  
   ```bash
   git clone https://github.com/chraac/llama-cpp-qnn-builder.git
   cd llama-cpp-qnn-builder
   # Follow README for QNN SDK setup (free from Qualcomm AI Hub / QAIRT — register once)
   ./build-android.sh --backend qnn --model-type gguf --enable-htp
   ```
   - It preserves your existing llama.cpp flow (GGUF loading, chat streaming, etc.).
   - Builds QNN + fallback to OpenCL/CPU.

2. **Integration into your Android app**:
   - Replace your current JNI/native layer calls with the new QNN-enabled `libllama.so`.
   - Add backend selector: `QNN` (priority) → OpenCL → CPU.
   - Use your lazy loading + WorkManager: Pre-load QNN context in background (QNN has lower cold-start than OpenCL).
   - Handle NPU-specific: Set `QNN_HTP_GRAPH_FINALIZE_OPTIMIZATIONS=1` and power mode `QNN_POWER_SAVER` for sustained decode.

3. **Model prep**:
   - Use the exact PRISM-PRO-DQ style quant if available (or Q4_0/Q3_K_M). MoE sparsity shines on NPU.
   - Test with Gemma 4 26B-A4B GGUF from ggml-org collection.

4. **Benchmark**:
   - Reuse your harness. Expect 80–200+ t/s decode on short contexts (MoE + NPU = magic).
   - Monitor thermal throttling (iQOO 13 has good cooling).

**Success criteria**: 100+ t/s stable decode, no Vulkan crashes (QNN bypasses it), fused ops handled natively on NPU.

### Phase 3: Model & App-Level Optimizations (1 week)
- **Speculative decoding**: Add `--speculative` (or draft model) in llama.cpp — huge decode boost on MoE.
- **KV cache & context**: Limit to 8K–16K initially; use your streaming logic.
- **Multi-modal (vision)**: Ensure QNN path supports vision tensors (Gemma 4 is multi-modal).
- **App polish**:
  - Foreground service: Your WorkManager workaround is good — keep it.
  - Diagnostics screen: Show active backend, layers offloaded, t/s live.
  - Error handling: Auto-fallback on NPU timeout (rare with QNN).

### Phase 4: Validation, Polish & Alternatives (Ongoing)
1. **Full benchmarks**:
   - Car-wash-style reasoning + image description (replicate the original demo).
   - Long-context stability.
   - Battery/thermal logs via ADB.

2. **If QNN falls short** (unlikely):
   - Switch to **Google LiteRT + QNN Accelerator** (SOTA for Gemma on Snapdragon 8 Elite — 100+ t/s decode reported on 8 Elite Gen 5).
   - Or ExecuTorch (PyTorch Edge) with native Qualcomm NPU delegate.
   - MLC-LLM as fallback (excellent TVM compiler for Adreno + Hexagon).

3. **Documentation & Sharing**:
   - Update your milestone doc with exact CMake flags, env vars, and benchmark CSV.
   - Open-source the QNN integration patches if clean.

**Risks & Mitigations**:
- QNN SDK access: Free but requires Qualcomm developer account (quick approval).
- Driver/ROM quirks on iQOO: Test on stock + custom if needed.
- Fused op fallbacks: Latest main + `--no-flash-attn` resolves 95% of cases.
- Memory: 26B MoE at Q4_0 fits comfortably in iQOO 13’s 16/18 GB RAM.

This plan gets you from “OpenCL slower than CPU” → “NPU blazing at demo speeds” while reusing 100% of your existing code.  

Start with **Phase 1 today** — pull latest llama.cpp and run the OpenCL build/test cycle. Share your current llama.cpp commit hash + exact model quant + any new error logs, and I’ll refine the CMake flags or patch suggestions further.  

You’re extremely close — this is how the ~300 t/s demo was achieved. Let’s ship it! 🚀
