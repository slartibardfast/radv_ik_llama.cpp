# Phase 18: GPU-Accelerated REDUCE via dmabuf

## Goal

Replace the CPU-mediated REDUCE with a GPU-side implementation using dmabuf zero-copy and the existing ADD shader. Eliminate the CPU round-trip that makes graph-split mode slow.

## Current State (Phase 17)

Each REDUCE does 3 synchronous operations through the CPU:

```
GPU A (Vega) ──read──> CPU ──add──> CPU ──write──> GPU B (6800 XT)
```

For Nemotron-Nano-4B: 64 REDUCE ops/token × 3 sync transfers each = **192 GPU↔CPU round-trips per token**. Result: 6.9 tok/s (vs 47 tok/s single-GPU).

## Target Architecture

Use dmabuf shared memory to keep everything on the GPU:

```
GPU A: copy partial_sum → dmabuf_export_buffer     [GPU A transfer queue]
GPU B: wait fence → copy dmabuf_import → local_tmp  [GPU B transfer queue]
GPU B: ADD(local_partial, local_tmp) → result        [GPU B compute queue]
```

No CPU reads, no CPU writes, no CPU element-wise add.

## Existing Infrastructure

All pieces already exist:

| Component | Location | Status |
|---|---|---|
| dmabuf export | `ggml_vk_create_dmabuf_exportable_buffer()` | Working (Phase 12) |
| dmabuf import | `ggml_vk_import_dmabuf_buffer()` | Working (Phase 12) |
| dmabuf staging cache | `ggml_vk_get_dmabuf_staging()` | Working — caches per device pair |
| ADD shader | `vulkan-shaders/add.comp` | Working — F16/F32, broadcast support |
| ADD pipeline | `pipeline_add[s0][s1][d]` | Working — 8 variants (F16/F32 combos) |
| Fence sync | `dmabuf_shared_staging::fence` | Working — per-peer fence |

## Implementation Plan

### Step 1: Rewrite `ggml_vk_reduce()` to use dmabuf + ADD shader

Replace the CPU-mediated loop with:

1. For each remote source `src[j]` on a different device:
   a. Get/create dmabuf staging between src device and dst device
   b. Submit GPU copy: `src[j]` device-local → dmabuf export buffer (on src device)
   c. Submit fence on src device
   d. Wait fence on dst device
   e. Submit GPU copy: dmabuf import buffer → temporary local buffer (on dst device)
   f. Dispatch ADD shader: `result = result + tmp` (on dst device compute queue)

2. Fall back to existing CPU path if dmabuf is unavailable (e.g. lavapipe, Windows).

**Verify by**: graph-split inference produces same output, no crashes.

### Step 2: Benchmark

Compare before/after:

| Mode | Before (CPU REDUCE) | Target |
|---|---|---|
| Graph-split gen tok/s | 6.9 | >15 |
| Graph-split prompt tok/s | 9 | >30 |

**Verify by**: `llama-cli -sm graph` benchmark numbers.

### Step 3: Async pipelining (stretch)

Overlap REDUCE transfers with subsequent GPU compute by using the async cross-device copy pipeline instead of synchronous fence waits. This requires integrating REDUCE into the `pending_xdev_copies` flow rather than handling it as an early return in `graph_compute`.

**Verify by**: further benchmark improvement.

## Key Design Decisions

1. **ADD shader reuse**: The existing `add.comp` already handles F16×F16→F16 and F32×F32→F32 with proper stride support. No new shader needed.

2. **Temporary buffer for imported data**: The dmabuf import buffer is shared staging — we can't dispatch ADD directly on it while the next REDUCE might reuse it. Copy to a local temp buffer first, then ADD.

3. **Fallback**: Keep the CPU path for non-dmabuf systems. The dmabuf path is an optimization, not a requirement.

4. **Two-GPU simplification**: With 2 GPUs, each REDUCE has exactly 1 remote source. The loop handles N sources but the common case is N=1.

## Files to Modify

| File | Change |
|---|---|
| `ggml/src/ggml-vulkan.cpp` | Rewrite `ggml_vk_reduce()` — dmabuf copy + ADD shader dispatch |

## Risk

- **Fence overhead**: Each REDUCE still needs a fence wait (GPU A done → GPU B starts). With 64 REDUCEs per token, fence latency could dominate even without CPU copies.
- **Mitigation**: If fence overhead is the bottleneck, Step 3 (async pipelining) batches the waits.
