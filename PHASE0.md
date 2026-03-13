# Phase 0: Backend-Ops Test Failure Fixes

## Status: IN PROGRESS (2 MUL_MAT + 1 MUL_MAT_ID + 2 CPY failures remaining)

## Overview

Fix all backend-ops test failures on RADV Vulkan (Vega 56 / RX 6800 XT). Originally 10 failures; 6 now fixed (4 by K-quant dequant, 1 by push constant alignment, 1 by IQK bf16 scalar fallback), 4 remain (+1 marginal iq3_xxs).

## Round 1: K-Quant Dequant Bounds Bug (2026-03-11)

**Root cause**: 5 K-quant dequant shaders (`dequant_q2_k.comp` through `dequant_q6_k.comp`) used `p.M * p.K / QUANT_K` for bounds checking instead of `p.nel / QUANT_K`. For multi-batch tensors, only the first batch was dequantized; remaining batches were uninitialized garbage in the prealloc buffer.

**Fixed**: 4 direct failures (q4_K x f16 batched MUL_MAT) + some indirect contamination failures.

Files changed: `dequant_q2_k.comp` through `dequant_q6_k.comp`

## Round 2: Push Constant & get_offsets Alignment (2026-03-12)

**Changes applied**:
1. **Buffer sizing**: `x_ne = ggml_nelements(src0)` instead of `ne01*ne00` (covers all batches/experts). Same for y_ne, d_sz.
2. **Push constant struct alignment**: Added `fusion_flags` (always 0) and `base_work_group_y` / `expert_i1` / `nbi1` fields to match upstream layout.
3. **get_offsets alignment**: Now does `batch_stride_a / QUANT_K` inside `get_offsets()`, matching upstream. Removed external `a_offset /= QUANT_K` from `mul_mat_vec.comp` and `a_offset / QUANT_K` from 12 specialized shaders.
4. **MUL_MAT_ID struct**: Added `fusion_flags`, `expert_i1`, `nbi1` fields to match upstream dispatch.

**Fixed**: iq3_xxs MUL_MAT and MUL_MAT_ID (marginal NMSE now consistently passes).

Files changed:
- `ggml/src/ggml-vulkan.cpp` — C++ push constant structs and dispatch code
- `ggml/src/vulkan-shaders/mul_mat_vec_base.comp` — push constant layout, get_offsets
- `ggml/src/vulkan-shaders/mul_mat_vec.comp` — removed a_offset /= QUANT_K
- 12 specialized `mul_mat_vec_*.comp` shaders — removed a_offset / QUANT_K

## Round 3: Structural Alignment with Upstream (2026-03-13)

**Changes applied**:
1. **Fuse0/Fuse1 buffer bindings**: Added `data_fuse0` (binding 3) and `data_fuse1` (binding 4) to `mul_mat_vec_base.comp`. Moved IDS binding from 3 to 5 for MUL_MAT_ID.
2. **Pipeline binding counts**: Changed from 3→5 (non-ID) and 4→6 (ID) in all `ggml_vk_create_pipeline` calls for mul_mat_vec pipelines.
3. **Dispatch subbuffers**: Updated dispatch calls to pass 5 subbuffers (non-ID) or 6 (ID), with d_D as dummy Fuse0/Fuse1 bindings.
4. **spirv-opt exclusions**: Disabled `-O` for bf16 and rope shaders in `vulkan-shaders-gen.cpp` (matching upstream issues #15344 and #16860).

**Result**: No test failures were fixed by these changes. However, these changes are kept for structural correctness — they make the fork's SPIR-V and pipeline layout match upstream.

Files changed:
- `ggml/src/vulkan-shaders/mul_mat_vec_base.comp` — Fuse0/Fuse1 bindings, IDS binding moved
- `ggml/src/ggml-vulkan.cpp` — pipeline creation (binding counts), dispatch (subbuffer counts)
- `ggml/src/vulkan-shaders/vulkan-shaders-gen.cpp` — spirv-opt exclusions for bf16 and rope

## Round 3: What Was Ruled Out

1. **Descriptor binding count (3 vs 5)**: Made fork match upstream's 5/6 binding layout. SPIR-V confirmed to contain Fuse0/Fuse1 declarations. No effect on test outcomes.

2. **spirv-opt on bf16**: Upstream disables spirv-opt for bf16 (issue #15344). Analysis showed fork's bf16 shaders were already effectively unoptimized (17044 bytes matches no-opt output; optimized would be 17436 bytes). Not the root cause.

3. **Subgroup vs shmem reduction**: Upstream explicitly disables subgroup reduction on AMD GCN (`use_subgroups = ... && architecture != AMD_GCN`). Both fork and upstream use shmem-only reduction on Vega. No difference.

## Round 4: IQK CPU bf16 Scalar Fallback (2026-03-13)

**Root cause**: bf16 MUL_MAT k=1 was NOT a GPU bug. The GPU output was correct; the CPU reference (IQK) produced garbage. IQK's `iqk_set_kernels_float` requires `ne00 % 32 == 0` for bf16 — when k=1 this returns false, and `iqk_mul_mat_4d` returns false. The ggml.c fallback should have handled it, but the `ggml_vec_dot_bf16` fallback also produced zeros for k=1 (separate issue). The test framework compared correct GPU output against broken CPU reference.

**Discovery method**: Wrote a minimal standalone test that ran bf16 k=1 on both GPU and CPU separately, printing actual values. GPU showed correct `{1.0, 2.0, ..., 16.0}`, CPU showed all zeros.

**Fix**: Added `mul_mat_bf16_scalar<FloatY, nrc_y>()` — a scalar bf16 mul_mat function in IQK that handles arbitrary `ne00` including 1. Three-tier dispatch in `iqk_set_kernels_float`:
1. AVX512BF16 native path: `ne00 % 32 == 0`
2. Generic SIMD path: `ne00 % k_step == 0` (8 on AVX2, 16 on AVX512)
3. Scalar fallback: any `ne00`

Files changed: `ggml/src/iqk/iqk_gemm_floats.cpp`

## Remaining Failures (4)

| # | Test | NMSE | Threshold | Notes |
|---|------|------|-----------|-------|
| 1 | MUL_MAT(iq4_xs x f32, m=16,n=1,k=256) | 0.012-0.040 | 0.0005 | Variable |
| 2 | MUL_MAT(iq3_xxs x f32, m=16,n=1,k=256) | 0.00065 | 0.0005 | Marginal |
| 3 | MUL_MAT_ID(iq4_xs x f32, n_mats=4,n_used=2,m=512,n=1,k=256) | 0.025-0.033 | 0.0005 | Variable |
| 4-5 | CPY(f32→iq4_nl) × 2 | 0.0012-0.0014 | 0.000001 | |

## Remaining Differences from Upstream

1. **iq4_xs dequantize4**: Fork reads 4 individual bytes (`data_a[...].qs[iq+0..3]`); upstream reads packed 32-bit word (`data_a_packed32[...].qs[iq/4]` + `unpack8`). Fork lacks `data_a_packed32` buffer alias.

2. **No A-buffer type aliases**: Fork's `mul_mat_vec_base.comp` only declares `data_a[]`. Upstream's `mul_mat_vec_iface.glsl` also declares `data_a_v4[]`, `data_a_packed16[]`, `data_a_packed32[]` aliases at binding 0.

## Build System Caveat

Changing `vulkan-shaders-gen.cpp` does NOT automatically trigger shader regeneration. The tool is built as a CMake ExternalProject with cached stamp files. After source changes:
1. `rm -rf build/ggml/src/vulkan-shaders-gen-prefix`
2. `cmake --build build --target vulkan-shaders-gen`
3. `build/Release/vulkan-shaders-gen --input-dir ... --output-dir ...`
4. Then rebuild the main target
