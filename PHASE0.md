# Phase 0: Backend-Ops Test Failure Fixes

## Status: COMPLETE

## Overview

Fix all 10 backend-ops test failures on RADV Vulkan (Vega 56 / RX 6800 XT). All failures were caused by a single bug in 5 K-quant dequantization shaders.

## Test Failures (Vulkan0 — RADV VEGA10, initial baseline 2026-03-11)

| # | Test | NMSE | Threshold | Category |
|---|------|------|-----------|----------|
| 1 | CPY(f32→iq4_nl, ne=[256,4,4,4], permute=[0,0,0,0]) | 0.00117 | 0.000001 | Prealloc contamination |
| 2 | CPY(f32→iq4_nl, ne=[256,2,3,4], permute=[0,2,1,3]) | 0.00120 | 0.000001 | Prealloc contamination |
| 3 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,1], nr=[1,1]) | 1.887 | 0.0005 | Direct K-quant dequant bug |
| 4 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,1], nr=[2,1]) | 1.786 | 0.0005 | Direct K-quant dequant bug |
| 5 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,10], nr=[1,1]) | 1.983 | 0.0005 | Direct K-quant dequant bug |
| 6 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,10], nr=[2,1]) | 1.997 | 0.0005 | Direct K-quant dequant bug |
| 7 | MUL_MAT(iq3_xxs×f32, m=16,n=1,k=256, bs=[1,1], nr=[1,1]) | 0.000539 | 0.0005 | Prealloc contamination |
| 8 | MUL_MAT(iq4_xs×f32, m=16,n=1,k=256, bs=[1,1], nr=[1,1]) | 0.032 | 0.0005 | Prealloc contamination |
| 9 | MUL_MAT(bf16×f32, m=16,n=1,k=1, bs=[1,1], nr=[1,1]) | 9.59 | 0.0005 | Prealloc contamination |
| 10 | MUL_MAT_ID(iq4_xs×f32, n_mats=4,n_used=2, m=512,n=1,k=256) | 0.035 | 0.0005 | Prealloc contamination |

## Root Cause

**Single bug**: 5 K-quant dequant shaders (`dequant_q2_k.comp` through `dequant_q6_k.comp`) used `p.M * p.K / QUANT_K` for bounds checking instead of `p.nel / QUANT_K`.

- `p.M = ne01`, `p.K = ne10` — these are 2D dimensions covering only one batch
- `p.nel = ggml_nelements(src0)` — covers all elements across all batches
- For multi-batch tensors (`ne02 > 1` or `ne03 > 1`), only the first batch was dequantized
- Remaining batches were left as **uninitialized garbage in the prealloc buffer**

### Direct failures (#3-6)

The 4 q4_K×f16 batched MUL_MAT tests used the dequant+matmul fallback path (no native q4_K×f16 pipeline without coopmat2). The dequantization step only processed the first batch, producing garbage output for subsequent batches. NMSE ~2.0 (completely uncorrelated).

### Indirect failures (#1-2, #7-10)

The test suite runs all operations sequentially using the same Vulkan context with shared prealloc buffers. When K-quant batched dequantization left uninitialized garbage in the prealloc buffer, subsequent tests that reused the same buffer region could read contaminated data. These failures were non-deterministic (appeared to be flaky) because buffer reuse patterns varied between runs.

## Fix

Changed the bounds check in all 5 K-quant dequant shaders from:
```glsl
if (ib >= p.M * p.K / QUANT_K) { return; }
```
to:
```glsl
if (ib >= p.nel / QUANT_K) { return; }
```

Files changed:
- `ggml/src/vulkan-shaders/dequant_q2_k.comp`
- `ggml/src/vulkan-shaders/dequant_q3_k.comp`
- `ggml/src/vulkan-shaders/dequant_q4_k.comp`
- `ggml/src/vulkan-shaders/dequant_q5_k.comp`
- `ggml/src/vulkan-shaders/dequant_q6_k.comp`

## Verification

3 consecutive clean runs on both GPUs, zero failures:

```
=== Vulkan0 (AMD Vega 56, RADV VEGA10, warp=64) ===
Run 1: 2/2 backends passed
Run 2: 2/2 backends passed
Run 3: 2/2 backends passed

=== Vulkan1 (AMD RX 6800 XT, RADV NAVI21, warp=32) ===
Run 1: 2/2 backends passed
Run 2: 2/2 backends passed
Run 3: 2/2 backends passed
```

Tests covered: MUL_MAT, MUL_MAT_ID, CPY (all operations that had failures).

Upstream comparison: upstream llama.cpp (same hardware) has zero failures for these operations, confirming the fix brings us to parity.
