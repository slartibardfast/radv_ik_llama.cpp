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

## Phase 0 Round 5: iq4_xs Was Also a CPU Bug (2026-03-13)
- **Same pattern as bf16**: iq4_xs MUL_MAT (NMSE=0.027) was NOT a GPU bug. GPU matched float64 expected at machine epsilon (1.8e-14). IQK's AVX2 `DequantizerIQ4XS` kernel has a systematic computation error in its unsigned-value bias compensation.
- **Fix**: Added scalar iq4_xs×Q8_K dot product (`mul_mat_iq4_xs_q8_K_scalar`) using signed `kvalues_iq4nl` directly. Used on non-AVX512 systems. CPU NMSE: 0.027 → 2.97e-05.
- **Collateral fix**: iq3_xxs MUL_MAT (marginal, NMSE=0.00065) now passes — likely IQK precision improvement from the same scalar fallback approach.
- **iq4_xs dequantize4 byte-by-byte vs packed32**: Verified mathematically equivalent (same bits after shift+mask). NOT a bug despite the code difference.
- **ggml-cpu fallback path is broken**: When IQK returns false from `iqk_set_kernels_kquants`, ggml-cpu's standard `ggml_vec_dot` fallback produces zeros. Must fix IQK internally, not disable it.
- Remaining 3 failures: MUL_MAT_ID(iq3_xxs) marginal, 2x CPY f32→iq4_nl.
- Also discovered: `ggml_internal_get_type_traits(GGML_TYPE_IQ4_XS).to_float` requires `ggml_init()` first — the `ggml_table_f32_f16` lookup table is populated lazily during init. Standalone tests that skip init get zeros from dequantize functions.

## Phase 20g: Multi-GPU MoE root cause = GROUPED_TOPK CPU fallback (2026-04-08)

Target: Qwen3.5-35B-A3B-UD-IQ3_XXS (12.17 GiB / 34.66 B params, MoE 256 experts × 8 active, 40 layers, n_kv_head=2, head_dim=256). Dual-GPU rig: RDNA2 6800 XT 16 GiB + Vega 64 8 GiB.

Measured baselines (single-GPU 6800 XT, post-Phase-20f reduce fix):
- pp256 / tg64 = **0.71 / 0.31 t/s** with default flags
- pp256 / tg64 = **3.42 / 0.31 t/s** with `-fmoe 0` (5× pp gain, tg unchanged because Phase 20b already skips fused-moe for N=1)
- Compare dense Llama-2-13B Q8_0 same GPU, similar VRAM (12.88 GiB): **511 / 27.7 t/s** — MoE is **150-720× slower per token**.

Multi-GPU layer-split (`-sm layer -ts 0.67/0.33`) gives **0.63 / 0.27 t/s** — same as single-GPU. **Multi-GPU is not the bottleneck.**

Root cause found via `llama-cli ... 2>&1`: prints **`graph splits = 322`** for Qwen3.5 MoE on Vulkan (compare ~3 splits for dense models). 322 splits = ~161 CPU↔GPU round-trip boundaries × ~10 ms each = ~3.2 sec/token of pure sync overhead, matching measured wall time. GPU-side compute is healthy (~15 ms/token via `GGML_VK_PERF_LOGGER=1` per-op timings).

The split source: **`GGML_OP_GROUPED_TOPK` is not in the Vulkan supports_op list** at `ggml/src/ggml-vulkan.cpp:9905-9923`. The op is defined in `ggml.h:658` and used by `llm_build_moe_ffn` at `llama-build-context.cpp:1131` for expert routing (`ggml_grouped_topk`). Every layer's expert selection therefore falls back to CPU, splitting the graph into hundreds of fragments.

**Phase 20h scope** (next plan): implement `GROUPED_TOPK` on the Vulkan backend (one new compute shader + dispatch wiring). Expected outcome: graph_splits drops from 322 → ~3, tg perf goes from 0.31 → ~25-40 t/s for Qwen3.5-35B-A3B IQ3_XXS, matching dense-model expectations.

Bug A (separate): `-fmoe=1` (default) makes pp eval 5× slower than `-fmoe=0` for MoE on Vulkan. Phase 20b's N=1 skip works for tg but the prompt-eval path still goes through fused MoE. Either fix the fused-moe pp path or change the default to `-fmoe=0` for MoE models on Vulkan.

## Phase 20h: Vulkan MOE_FUSED_UP_GATE shader (2026-04-08)

Bug A fixed by adding the missing GGML_OP_MOE_FUSED_UP_GATE Vulkan implementation. The op was defined in ggml.h and implemented for CPU+CUDA, but not Vulkan, so multi-token MoE prompt-eval went through unfused fallback.

Implementation: extended the existing dense `mul_mm_fused_up_gate.comp` with a `MUL_MAT_ID` build flag mirroring the upstream `mul_mm.comp` MUL_MAT_ID branches. Push constants are layout-shared between dense and MoE — macro aliases (`p_nei0`/`p_nei1`/`p_nbi1`/`p_ne11`) re-interpret the dense `ne0X/broadcast` slots as the MoE indirection fields, so a single C struct serves both. Backported the full IQ family (IQ1_S..IQ4_XS) into the dense fused_up_gate gen as well, since IQ3_XXS is the target model.

The non-obvious bug: a buffer subbuffer scoping mistake. The dispatch passed `b_sz = sizeof(float) * ne10 * ne11` to the b binding, which is one token's worth of data. The shader strides across tokens via `row_idx.y * batch_stride_b`, so it needs the FULL tensor visible to the binding. Fix: `b_sz = sizeof(float) * ne10 * ne11 * b->ne[2]`. Single-token cases passed (only token 0 was needed), but multi-token cases produced zero output for tokens > 0 because the GPU's bounds-protected reads outside the bound subbuffer returned zero. Diagnosed by reading `buf_b` shared memory back into `dst` from the write loop and observing col 1 = 0 even though the load formula was correct.

Headline result: Qwen3.5-35B-A3B-UD-IQ3_XXS pp256 on 6800 XT with default `-fmoe=1`: 3.47 t/s (vs the broken 0.71 t/s pre-fix and the unfused 3.42 t/s baseline). The fused path is now at parity with unfused on this model — the inference graph no longer falls back to CPU for the fused MoE FFN op. Test coverage: 40 new test_moe_fused_up_gate cases pass on both Vega 64 (RDNA-GCN5) and 6800 XT (RDNA2) on top of the existing 1237/1237 baseline and 143/143 dense FUSED_UP_GATE cases.

Note for Phase 20i: the headline `tg32 = 0.31 t/s` is unchanged because token gen is bottlenecked by `GROUPED_TOPK` falling back to CPU (322 graph splits per token). 20h was a prerequisite (so the fused MoE path is GPU-resident), but the tg headline win has to wait for Phase 20i.

## Phase 20i: Vulkan GROUPED_TOPK shader (2026-04-08)

Added the GGML_OP_GROUPED_TOPK Vulkan implementation as a single fused compute shader (`grouped_topk_f32.comp`), one workgroup per token row. All stages (per-group sort + sum, group selection, masking, global sort, write top-k indices) run in shared memory with no temp buffers. 7/7 test cases pass on both Vega 64 and 6800 XT covering Qwen3.5 and DeepSeek-style shapes.

**However**: this op is NOT what was bottlenecking Qwen3.5-35B-A3B-UD-IQ3_XXS. The Phase 20g diagnosis was wrong on the root cause. With `GGML_SCHED_DEBUG=2` it's clear that Qwen3.5 uses ordinary `ARGSORT` (which is already on Vulkan) for expert routing, not `GROUPED_TOPK`. The 322 graph splits per layer come from two other ops:

- `GGML_OP_MUL_MULTI_ADD` (80 CPU instances) — the fork's multi-expert gather-and-add op, used to combine the routed expert outputs (`routed_out` = sum over experts of `ffn_moe_down * ffn_moe_weights_norm`). Defined in `ggml.h:623`, implemented in CPU at `ggml.c:6213` and CUDA at `ggml-cuda.cu:3475/4656`. **Not in Vulkan supports_op at all** — `grep MUL_MULTI_ADD ggml-vulkan.cpp` returns nothing. This is a 1-shader implementation that should be straightforward to port.

- `GGML_OP_FUSED_MUL_UNARY` (40 CPU instances) — the dense FUSED_MUL_UNARY op IS in Vulkan, but for the `ffn_shexp_gated` (shared expert gate) calls the supports_op check at `ggml-vulkan.cpp:11785` rejects them because `op->src[0]` (`shared_expert_ga…`) and `op->src[1]` (`ffn_shexp_out…`) don't pass the `ggml_are_same_shape` check. Likely a broadcast pattern that needs separate support.

Phase 20i still ships GROUPED_TOPK because it's correct, well-tested, and the right op for any model that uses it (DeepSeek-V3, BailingMoE, etc.). The Qwen3.5 headline gain has to wait for Phase 20j (MUL_MULTI_ADD) and Phase 20k (FUSED_MUL_UNARY broadcast).

f16acc dispatch counter (Phase 20c instrumentation) on this MoE model showed `hits=0 fallbacks=43704`. Hits=0 is correct on RDNA2 (f16acc is Vega-only). On Vega the same 43704 dispatches would hit the f16acc path — but the bottleneck is graph_splits, not compute, so the f16acc work cannot move the needle until Phase 20h lands.

## Build Notes
- Use clang (GCC 15 has -Wtemplate-body errors)
- `-DGGML_IQK_FLASH_ATTENTION=OFF` on non-AVX2 hosts
- Remote: use absolute `-S`/`-B` paths with cmake (SSH starts in /home/llm)
