# Phase 22: Backend-Ops Test Failure Fixes

## Status: In Progress

## Overview

Fix remaining backend-ops test failures on RADV Vulkan (Vega 56 / RX 6800 XT). These are pre-existing failures inherited from upstream, not regressions from our multi-GPU work.

## Test Failures (Vulkan0 — RADV VEGA10, 2026-03-11)

| # | Test | NMSE | Threshold | Category |
|---|------|------|-----------|----------|
| 1 | CPY(f32→iq4_nl, ne=[256,4,4,4], permute=[0,0,0,0]) | 0.00117 | 0.000001 | Quant error vs tight threshold |
| 2 | CPY(f32→iq4_nl, ne=[256,2,3,4], permute=[0,2,1,3]) | 0.00120 | 0.000001 | Quant error vs tight threshold |
| 3 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,1], nr=[1,1]) | 1.887 | 0.0005 | Garbage — broken pipeline path |
| 4 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,1], nr=[2,1]) | 1.786 | 0.0005 | Garbage — broken pipeline path |
| 5 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,10], nr=[1,1]) | 1.983 | 0.0005 | Garbage — broken pipeline path |
| 6 | MUL_MAT(q4_K×f16, m=16,n=16,k=256, bs=[10,10], nr=[2,1]) | 1.997 | 0.0005 | Garbage — broken pipeline path |
| 7 | MUL_MAT(iq3_xxs×f32, m=16,n=1,k=256, bs=[1,1], nr=[1,1]) | 0.000539 | 0.0005 | Borderline precision |
| 8 | MUL_MAT(iq4_xs×f32, m=16,n=1,k=256, bs=[1,1], nr=[1,1]) | 0.032 | 0.0005 | Broken dequant/mul_mat_vec |
| 9 | MUL_MAT(bf16×f32, m=16,n=1,k=1, bs=[1,1], nr=[1,1]) | 2.312 | 0.0005 | Garbage — broken pipeline path |
| 10 | MUL_MAT_ID(iq4_xs×f32, n_mats=4,n_used=2, m=512,n=1,k=256) | 0.037 | 0.0005 | Same root cause as #8 |

## Root Cause Analysis

### Group A: CPY f32→iq4_nl (#1-2)

NMSE ~0.001 for a 4-bit quantization format. The CPY shader (`pipeline_cpy_f32_quant[IQ4_NL]`) correctly quantizes f32 to iq4_nl, but 4-bit quantization inherently has ~0.1% relative error. The test threshold of 1e-6 is appropriate for lossless copies, not lossy quantization. This is a **test threshold issue**, not a shader bug.

### Group B: MUL_MAT q4_K×f16 batched (#3-6)

NMSE ~1.8-2.0 (uncorrelated output). All failing cases have `bs=[10,...]` (batched). Pipeline selection flow:

1. `get_mul_mat_mat_pipeline(q4_K, F16)` → returns `nullptr` (line 4463: `src1_type != F32 && !coopmat2`)
2. Fallback: `qx_needs_dequant = true`, dequant q4_K→F16, then F16×F16 matmul
3. The dequant+matmul fallback path likely has a batching bug — buffer offsets or strides are wrong when `ne12 > 1`

Key observation: only batched cases (`bs=[10,...]`) fail. The non-batched q4_K×f16 case presumably passes (not in failure list), suggesting the per-batch offset calculation is wrong in the dequant+matmul fallback.

### Group C: MUL_MAT iq4_xs/iq3_xxs×f32 mat-vec (#7-8, #10)

These use `n=1`, hitting the `mul_mat_vec` path (`ggml_vk_mul_mat_vec_q_f16`). NMSE ranges from borderline (iq3_xxs: 0.000539) to clearly wrong (iq4_xs: 0.032). The IQ quant dequant shaders may have precision issues in the vec path, or the `ggml_vk_get_dequantize_mul_mat_vec` pipeline for these types has a bug.

### Group D: MUL_MAT bf16×f32 (#9)

NMSE ~2.3 (garbage). Pipeline flow:
1. `f16_type = BF16` (src0 is bf16)
2. `y_non_contig = true` (src0 is bf16 && src1 is not bf16)
3. `get_pipeline(bf16, BF16)` → `pipeline_matmul_bf16` (OK)
4. `qy_needs_dequant = true` → Y converted from F32 to BF16 via `ggml_vk_get_cpy_pipeline(src1, nullptr, BF16)`
5. With `m=16, n=1, k=1` this is a tiny matmul. The issue is likely in the F32→BF16 copy or the bf16 matmul for edge-case dimensions.

Note: `supports_op` correctly rejects `bf16×f16` but accepts `bf16×f32` — however the conversion path may be broken.

## Plan

1. Fix Group B (q4_K×f16 batched) — trace the dequant+matmul fallback path for batch offset bugs
2. Fix Group D (bf16×f32) — verify F32→BF16 copy pipeline and bf16 matmul for small dimensions
3. Fix Group C (iq4_xs/iq3_xxs vec) — investigate dequant precision in mul_mat_vec path
4. Fix Group A (CPY iq4_nl threshold) — adjust test threshold or mark as expected quantization error
5. Verify fixes on both GPUs
