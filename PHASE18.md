# Phase 18: GPU-Accelerated REDUCE via dmabuf

## Goal

Replace the CPU-mediated REDUCE with a GPU-side implementation using dmabuf zero-copy and the existing ADD shader. Eliminate the CPU round-trip that makes graph-split mode slow.

## Before (Phase 17)

Each REDUCE does 3 synchronous operations through the CPU:

```
GPU B: read local_partial → CPU       [fence wait]
GPU A: read remote_partial → CPU      [fence wait]
CPU: element-wise ADD
GPU B: write result ← CPU             [fence wait]
```

64 REDUCE ops/token × 3 fence waits = **192 GPU↔CPU round-trips per token**.

## After (Phase 18)

dmabuf shared memory keeps everything on the GPU:

```
GPU A: copy partial_sum → dmabuf_export_buffer     [fence wait]
GPU B: copy dmabuf_import → local_tmp               ─┐
GPU B: ADD(local_partial, local_tmp) → result        ─┘ [single fence wait]
```

64 REDUCE ops/token × 2 fence waits = **128 GPU round-trips per token**. No CPU data movement.

## Implementation

### Changes to `ggml_vk_reduce()` in `ggml/src/ggml-vulkan.cpp`

**dmabuf fast path** (when both devices support dmabuf):

1. Get/create dmabuf staging between src and dst devices (cached per device pair)
2. GPU A: `vkCmdCopyBuffer` from src device-local → dmabuf export buffer, submit with fence
3. Wait fence on src device
4. GPU B: single command buffer that:
   - Copies dmabuf import → `reduce_temp_buffer` (device-local, lazily allocated)
   - Pipeline barrier (transfer → compute)
   - Dispatches `add_norepeat` shader: `dst[i] = dst[i] + tmp[i]`
5. Submit on compute queue, wait fence

**CPU fallback** (no dmabuf): same as Phase 17 implementation, restructured per-source.

### New device state

| Field | Purpose |
|---|---|
| `reduce_temp_buffer` | Device-local buffer for receiving dmabuf data |
| `reduce_descriptor_pool` | Single-descriptor-set pool for ADD shader |
| `reduce_descriptor_set` | Pre-allocated descriptor set, reused across REDUCE calls |

### Key design decisions

1. **ADD shader reuse**: `add.comp` with `pipeline_add_norepeat[f16][f16][f16]` handles same-shape contiguous F16 ADD. No new shader needed.

2. **In-place ADD**: Binding 0 (src0) and binding 2 (dst) point to the same buffer. The shader reads `data_a[idx]` before writing `data_d[idx]`, so in-place is safe.

3. **Descriptor set management**: Separate pool (1 descriptor set) avoids entangling with the normal graph pipeline's descriptor allocation.

4. **Temp buffer**: Can't dispatch ADD directly on the dmabuf import buffer — it lacks `eStorageBuffer` usage flag (only `eTransferSrc | eTransferDst`). Copy to a regular device-local temp buffer first.

## Results

Nemotron-Nano-4B-Q4_K_M, Vega + 6800 XT, `-c 2048`, 18-token prompt + 49 generated:

| Mode | Prompt (tok/s) | Gen (tok/s) |
|---|---|---|
| Single GPU (6800 XT) | 291 | 48.4 |
| Layer split (`-sm layer`) | 137 | 18.5 |
| **Graph split, dmabuf REDUCE** | **47.5** | **7.8** |
| Graph split, CPU REDUCE (before) | 9 | 6.5 |

### Analysis

- **Prompt eval 5.3× faster** (9 → 47.5 tok/s): CPU element-wise ADD on multi-token batches was the dominant bottleneck.
- **Token gen +20%** (6.5 → 7.8 tok/s): Modest gain because the per-token REDUCE data is small (6KB F16) — fence latency dominates, not data movement.
- **Remaining gap**: Graph split is still 2.4× slower than layer split for token gen. The 128 fence waits per token (2 per REDUCE × 64 REDUCEs) cost ~20-50μs each ≈ 2.5-6.4ms overhead per token. The rest is actual GPU compute overhead from 193 splits.

## Future Optimization

- **Async pipelining (Step 3)**: Overlap REDUCE transfers with subsequent GPU compute. Would require integrating REDUCE into the `pending_xdev_copies` flow instead of synchronous fence waits.
- **Batched fence waits**: Coalesce multiple REDUCE fence waits where possible.
- Graph split mode is primarily useful for models that don't fit on one GPU — for models that fit, layer split is always faster.
