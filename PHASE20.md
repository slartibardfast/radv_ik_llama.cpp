# Phase 20: Vulkan Token Generation Performance

## Status: IN PROGRESS

## Problem

Vulkan token generation is 2-6x slower than ROCm on identical hardware, leaving most memory bandwidth unused.

### Measured bandwidth utilization (6800 XT, 512 GB/s peak)

| Model | Theoretical | ROCm tok/s | Vulkan tok/s | ROCm BW% | Vulkan BW% |
|-------|------------|-----------|-------------|----------|------------|
| TinyLlama 1.1B Q2_K | 1305 | 380 | 62 | 29% | 4.8% |
| Llama-2-7B Q8_0 | 73 | 69 | 31 | 95% | 42% |

ROCm achieves near-perfect bandwidth utilization on 7B (95%). Vulkan achieves 42% — a **2.3x gap** on large models. On small models the gap widens to 6x due to dispatch overhead.

### Vega (484 GB/s HBM2 peak)

| Model | ROCm tok/s | Vulkan tok/s | ROCm advantage |
|-------|-----------|-------------|----------------|
| TinyLlama 1.1B Q2_K | 244 | 31 | 7.9x |
| Llama-2-7B Q8_0 | — (OOM) | 8 | — |

Vega's Vulkan performance is worse than 6800 XT relative to ROCm, suggesting additional GCN5-specific shader issues.

## Root Cause Analysis

### Large models (7B+): shader memory access inefficiency

The `mul_mat_vec` kernel achieves 42% bandwidth on 6800 XT. The `test-backend-ops perf` micro-benchmark shows the kernel DOES saturate bandwidth on large matrices:

```
q8_0, m=128256, n=1, k=3072: 2159 GB/s effective (saturated)
q8_0, m=16,     n=1, k=256:     9 GB/s effective (dispatch-bound)
```

But during real inference, the actual weight matrices are moderate-sized (4096×4096 for 7B). At these sizes the kernel doesn't fully saturate bandwidth — memory access patterns, workgroup sizing, or reduction strategy may be suboptimal.

### Small models (<2B): dispatch overhead dominates

TinyLlama has 511 graph nodes per token. At 62 tok/s, each token takes 16 ms = **31 µs per node average**. With Vulkan dispatch overhead of ~3-5 µs per `vkCmdDispatch`, dispatch alone consumes ~2 ms (12%). But the bigger issue is that small weight matrices (2048×2048, ~1.4 MB for Q2_K) take only 2.7 µs to read at peak bandwidth — the kernel can't amortize launch overhead at this size.

ROCm's HIP has ~10x lower dispatch overhead than Vulkan's command buffer model, explaining the 6x gap on small models.

## Optimization Targets

### 1. mul_mat_vec memory coalescing (high impact, large models)

Current shader may have suboptimal memory access patterns for the A matrix (quantized weights). Ensuring coalesced reads across the workgroup could close the 42% → 90%+ gap.

### 2. Workgroup size tuning (medium impact, all models)

The fork uses a fixed workgroup size. Upstream llama.cpp has a 3D pipeline array `[DMMV_WG_SIZE_COUNT][type][cols]` that selects workgroup sizes based on matrix dimensions. Porting this heuristic could improve efficiency.

### 3. Kernel fusion (high impact, small models)

Fusing consecutive small ops (e.g., RMS_NORM + MUL_MAT, or MUL + ADD) reduces dispatch count. The fork already has FUSED_UP_GATE (Phase 13). Additional fusion targets:
- FUSED_RMS_NORM + MUL_MAT (norm then project)
- Fused QKV projection (single dispatch for Q, K, V)

### 4. Vega-specific: Rapid Packed Math (medium impact, Vega only)

Vega's RPM processes two FP16 values per instruction via `f16vec2`. The mul_mat_vec dequant path currently works in F32. Using F16 packed math could improve Vega throughput — but this is secondary to the memory access fixes that benefit all GPUs.

### 5. Async command buffer submission (medium impact, all)

Vulkan allows pre-recording command buffers. If the current implementation records and submits per-dispatch, batching into a single command buffer per token could eliminate most dispatch overhead.

## Approach

1. Profile mul_mat_vec with `VK_EXT_calibrated_timestamps` or radv perf counters to identify the bottleneck (ALU-bound vs memory-bound vs launch-bound)
2. Compare workgroup/dispatch parameters with upstream
3. Implement workgroup size heuristic from upstream
4. Measure impact on 7B token gen
5. If still >2x gap, investigate memory access patterns in the shader

## Verify by

Vulkan 7B Q8_0 token gen on 6800 XT improves from 31 tok/s toward 60+ tok/s (>80% bandwidth utilization).

## References

- Benchmark data: [BENCHMARKS.md](BENCHMARKS.md)
- Upstream workgroup heuristic: `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` line ~3978 (3D pipeline array)
- AMD GCN5 ISA: Rapid Packed Math `v_pk_*` instructions
- RADV: `VK_KHR_shader_float16_int8` for `float16_t` / `f16vec2`
