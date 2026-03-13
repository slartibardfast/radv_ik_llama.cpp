# MEMORY.md

## Submodule Branch Layout
- `dc/vulkan-split-mode-graph` — 6 commits: multi-GPU split mode graph for Vulkan (Phases 1-12)
- `dc/iqk-scalar-fallbacks` — 1 commit: graceful scalar fallbacks + AVX2 compile flags for non-AVX2 x86
- Branch prefix `dc/` follows upstream convention (`ik/`, `fcp/`, `s6/`)
- Commit messages use `vulkan:` or `iqk:` prefix (no phase references)
- Code comments must not reference phase numbers

## All Phases Complete
- Phases 1-10, 12 implemented. Phase 11 skipped (no opportunity with 3 graph splits).
- dmabuf zero-copy (Phase 12) dropped 7B overhead from ~15 ms/tok to ~1 ms/tok
- 13B multi-GPU faster than single-GPU for token generation (doubled memory bandwidth)

## CLAUDE.md Review (2026-03-08)
- Removed 28 "Phase N:" references from code comments across ggml-vulkan.cpp and ggml-vulkan-multigpu.cpp
- Moved AVX2 CMake compile flags from vulkan branch to IQK branch
- dmabuf detection mismatch in print_gpu_info accepted as cosmetic (device init does full probe)

## Hardware
- **Local (zen2)**: Ryzen 9 3950X, RX 6800 XT + Vega 64, AVX2
- **Alternate (retro)**: Xeon X5650, Polaris 12 + lavapipe, no AVX2 — available if needed

## Phase 13: FUSED_UP_GATE
- Shader bug found and fixed: gate accumulation loop in mul_mm_fused_up_gate.comp was missing cache_b reload. With WNITER>1 and N<BN, gate pass reused stale zeros from UP pass, producing all-zero output.
- CPU backend ABORTs on FUSED_UP_GATE (no implementation). Tests use decomposed reference: separate mul_mat + fused_mul_unary on CPU, then compare via NMSE.
- K-quant types (Q4_K, Q6_K) have block size 256; test K dimensions must be multiples of block size, not just 32.
- GGML_VK_VISIBLE_DEVICES=0 limits to single GPU (VEGA10) — avoids dual-GPU init overhead in tests.
- 50/50 FUSED_UP_GATE tests pass, 1187/1187 standard backend-ops tests pass on RADV VEGA10.

## Bug Fixes Found During Testing (2026-03-09)
- **Empty-graph fence hang**: `graph_compute` set `compute_pending=true` unconditionally. Test framework sentinel nodes (GGML_OP_NONE) produce graphs with zero GPU submissions. Next `synchronize()` spins forever on unsignaled fence. Fix: guard with `submit_count > 0`.
- **MULTI_ADD descriptor range**: `ggml_vk_op_f32` has an incontiguous-op block that overwrites `x_sz` with `ggml_nbytes(src0)`. For MULTI_ADD's strided view_2d, this gives only the view's logical size, not the full expert data span the shader reads. With `robustBufferAccess`, out-of-range reads silently return 0 — shader appeared to sum only expert 0. Fix: override x_sz *after* the incontiguous block.
- **FUSED_UP_GATE M=1 NMSE instability**: Single-element output near zero produces huge NMSE from tiny absolute errors. Fixed by increasing K from 32 to 64 so the output has more signal.
- Final test counts: 1190 standard + 143 FUSED_UP_GATE + 12 MULTI_ADD all pass on RADV VEGA10.

## Nemotron Architecture Confusion (2026-03-09)
- Target model is **Nemotron-3-Nano-30B-A3B** which uses `nemotron_h_moe` (hybrid Mamba2+Attention+MoE), NOT `LLM_ARCH_DECI`.
- DECI is a pure transformer variant (Nemotron-51B, Ultra-253B). Different architecture entirely.
- `nemotron_h_moe` is not recognized by our ik_llama.cpp fork — needs architecture registration, graph builder, and 5 missing Vulkan ops (SSM_CONV, SSM_SCAN, SWIGLU, ADD_ID, SET_ROWS).
- Upstream llama.cpp (added as `llama.cpp` submodule) has full support including Vulkan shaders for all required ops.
- MUL_MAT_ID (expert matmul) is already in our fork's Vulkan backend.

## Submodule Layout Update
- `ik_llama.cpp` — our fork with multi-GPU split mode
- `llama.cpp` — upstream reference for nemotron_h_moe and other missing architectures

## Vulkan REDUCE Op (2026-03-09)
- REDUCE is a cross-device collective ADD used by split-mode graph (`-sm graph`). CUDA uses P2P `cudaMemcpyPeerAsync`; Vulkan has no P2P.
- Implemented as CPU-mediated host staging: `ggml_vk_buffer_read` from each device → CPU ADD → `ggml_vk_buffer_write` back.
- The scheduler's special REDUCE handling (identity tensor_id_copy, no n_inputs increment) was NOT changed — CUDA depends on it. Instead, REDUCE is handled entirely in the Vulkan backend.
- `ggml_vk_reduce()` is called from `graph_compute` before the dryrun/build loop, since REDUCE splits are always single-node graphs.
- Performance: CPU round-trip is slow for graph-split (193 splits, 6.5 tok/s). Layer-split (default, 3 splits, 18 tok/s) doesn't use REDUCE. Future: dmabuf GPU→GPU + ADD shader.

## Phase 18: dmabuf REDUCE (2026-03-10)
- Replaced CPU-mediated REDUCE with dmabuf GPU-to-GPU copy + ADD shader dispatch on destination device.
- Graph-split prompt eval: 9→47 tok/s (5.3×). Token gen: 6.5→7.8 tok/s (+20%).
- Token gen improvement modest because per-REDUCE data is small (6KB F16) — fence latency dominates, not bandwidth.
- Can't bind dmabuf import buffer directly as storage buffer (only eTransferSrc|eTransferDst). Must copy to temp device-local buffer first.
- Separate descriptor pool (1 set) for REDUCE's ADD dispatch avoids entangling with graph pipeline's descriptor management.

## SUM_ROWS Bug (2026-03-10)
- Three bugs: GPU descriptor range only covered ne00×ne01 (not all rows), shader had no bounds check for extra dispatch workgroups, CPU had `ne0` (=1) instead of `ne02` in row index calculation.
- All fixed in one commit. SUM_ROWS now passes on both Vega and 6800 XT.

## Vega 10 (GCN5) Optimization Research
- No native DP4A — integer dot product emulation via float math (dequant+FMA at 4-8 cycles) outperforms "correct" integer emulation. The real gap vs DP4A hardware is 2-4× not 8×.
- Unexploited: Vega's Rapid Packed Math (f16vec2 packed FP16 arithmetic). llama.cpp Vulkan shaders don't use RPM in fallback paths.
- Optimization strategy: f16vec2 packed arithmetic + careful VGPR budgeting (≤32 regs for max occupancy) + full 64-lane wavefront subgroup reductions.
- AMDVLK is extinct; RADV is the sole Vulkan driver for GCN. RADV continues improving but dedicated GCN shader optimization in inference frameworks is an unfilled niche.

## Phase 0: K-Quant Dequant Bounds Check Bug (2026-03-11)
- 5 K-quant dequant shaders (q2_k through q6_k) used `p.M * p.K / QUANT_K` for bounds checking. For multi-batch tensors, `p.M×p.K` only covers one batch — remaining batches left as uninitialized garbage in prealloc buffer.
- Fix: change to `p.nel / QUANT_K` (nel = total elements across all batches).
- This single fix resolved 7 of 10 test-backend-ops failures: 4 direct (q4_K×f16 batched) + 3 indirect (prealloc contamination for iq3_xxs).
- Key insight: "flaky" test failures in GPU backends can be prealloc buffer contamination from a completely different operation's bug.

## Phase 0 Round 2: Push Constant & get_offsets Alignment (2026-03-12)
- Aligned fork's mul_mat_vec push constant struct with upstream: added fusion_flags, base_work_group_y (non-ID) / expert_i1, nbi1 (ID).
- Moved `batch_stride_a / QUANT_K` division into get_offsets() (matching upstream), removed inline `a_offset / QUANT_K` from mul_mat_vec.comp and 12 specialized shaders.
- Fixed iq3_xxs MUL_MAT and MUL_MAT_ID (both now pass).
- 3 failures remain: iq4_xs MUL_MAT (NMSE=0.040), bf16 k=1 MUL_MAT (NMSE=4.2), iq4_xs MUL_MAT_ID (NMSE=0.032). All reproduce in isolation — NOT contamination.
- bf16 is the deepest puzzle: GLSL is truly identical to upstream, push constants now match, yet massive errors. Only remaining structural difference is 2 extra descriptor bindings (Fuse0/Fuse1) in upstream SPIR-V.

## Vega Inference Bug: get_tensor_async Race Condition (2026-03-11)
- **Root cause**: `get_tensor_async` calls `buffer_read_2d_async` which does a synchronous memcpy from host-visible (rBAR) GPU buffer. But `get_tensor_async` is called BEFORE `synchronize()`, so GPU compute may still be in flight. On Vega (slower), the CPU reads stale data; on 6800 XT (faster), the GPU usually finishes in time — pure timing-dependent race.
- **Symptoms**: garbage output when ngl > n_layers (output matmul on GPU). ngl <= n_layers works (output stays on CPU). All backend-ops tests pass (they use synchronous reads).
- **Misleading clue**: inserting submit+fence_wait mid-graph "fixed" the issue — this forced the compute to complete before the next `get_tensor_async` read.
- **Fix**: for host-visible non-UMA buffers, `get_tensor_async` records a deferred memcpy in `ctx->pending_host_memcpys` (new field on backend context) instead of copying immediately. `synchronize()` processes these after `sync_compute` waits for the compute fence.
- **Why transfer_ctx->out_memcpys didn't work**: `graph_cleanup` in `sync_compute` clears `gc.contexts`, destroying the transfer context. The weak_ptr `ctx->transfer_ctx` expires, so `synchronize()` exits early without processing `out_memcpys`.
- **Key insight**: `buffer_read` (synchronous) checks `eHostVisible && uma` before direct memcpy, but `buffer_read_2d_async` only checked `eHostVisible`. The UMA check prevented the issue on UMA devices where compute and host share coherent memory. For discrete GPUs with rBAR, the async path was broken.

## Phase 0 Round 3: Structural Alignment with Upstream (2026-03-13)
- Added Fuse0/Fuse1 buffer bindings (binding 3, 4) to mul_mat_vec_base.comp; moved IDS from binding 3 to 5. Pipeline creation 3→5 (non-ID), 4→6 (ID). Dispatch passes dummy Fuse subbuffers.
- Disabled spirv-opt for bf16 (issue #15344) and rope (issue #16860) shaders in vulkan-shaders-gen.cpp. BUT: bf16 was already effectively unoptimized (17044 bytes matches no-opt output). Not the root cause.
- **Neither change fixed any test failure.** Kept for structural correctness (SPIR-V and pipeline layout now match upstream).
- Remaining 5 failures unchanged: bf16 k=1 (NMSE~3-6), iq4_xs MUL_MAT (NMSE~0.02), iq4_xs MUL_MAT_ID (NMSE~0.03), 2x CPY f32→iq4_nl.
- Key remaining upstream differences: (1) iq4_xs dequantize4 reads individual bytes vs upstream's packed32+unpack8, (2) no data_a_packed32/v4/packed16 buffer aliases, (3) bf16 k=1 passes alone but fails in full suite despite using own buffer (not prealloc).
- Build system caveat: changing vulkan-shaders-gen.cpp does NOT auto-trigger shader regen. Must manually rm -rf build/ggml/src/vulkan-shaders-gen-prefix, rebuild tool, then re-run it.

## Phase 0 Round 4: bf16 k=1 Was a CPU Bug, Not GPU (2026-03-13)
- **Critical lesson**: bf16 MUL_MAT k=1 (NMSE=2.88) was NOT a GPU bug. The GPU produced correct output. The CPU reference (IQK) produced garbage because `iqk_set_kernels_float` requires `ne00 % 32 == 0` for bf16 — k=1 fails this check, IQK returns false, fallback also broken.
- **Fix**: Added scalar bf16 mul_mat fallback in IQK for `ne00 < k_step`. Three-tier dispatch: AVX512BF16 (ne00%32), generic SIMD (ne00%k_step), scalar (any ne00).
- **Methodology that found it**: Wrote minimal standalone test printing GPU vs CPU values side-by-side. GPU: correct. CPU: all zeros. Took 10 minutes.
- **Methodology that failed**: 6+ hours exhaustively comparing GPU dispatch code, push constants, SPIR-V bytecode, spec constants vs upstream. Everything matched. The bug wasn't there.
- **Rule**: ALWAYS verify CPU reference output before debugging GPU. Large NMSE (>1.0) is a red flag for CPU reference bugs, not GPU bugs. Edge-case dimensions (k=1) break CPU alignment assumptions.
- Remaining 4 failures: iq4_xs MUL_MAT, iq3_xxs MUL_MAT (marginal), iq4_xs MUL_MAT_ID, 2x CPY f32→iq4_nl.

## Build Notes
- Use clang (GCC 15 has -Wtemplate-body errors)
- `-DGGML_IQK_FLASH_ATTENTION=OFF` on non-AVX2 hosts
- Remote: use absolute `-S`/`-B` paths with cmake (SSH starts in /home/llm)
