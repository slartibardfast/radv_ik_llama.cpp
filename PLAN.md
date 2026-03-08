# Vulkan Multi-GPU Split Mode Graph

## Design Principles

- **Vendor-neutral**: All changes use Vulkan spec features only (no driver-specific extensions). Works on RADV, NVIDIA proprietary, ANV (Intel), lavapipe, etc.
- **NVIDIA-safe**: No changes to CUDA/NCCL paths. Vulkan backend changes use standard Vulkan 1.2+ APIs (timeline semaphores, host-visible staging). NVIDIA's Vulkan driver supports all of these.
- **Upstreamable**: Minimal, surgical changes. Phase 1 (async+events) is a standalone improvement that benefits single-GPU Vulkan too. Each phase can be upstreamed independently.
- **Host-staging model**: Cross-device transfers go through host-visible memory (GPU A → host → GPU B). This is the only portable approach in the Vulkan spec. Avoids driver-specific P2P extensions like `VK_KHR_external_memory_fd`.

## File Layout

- `ggml/src/ggml-vulkan.cpp` — direct submodule edits for vtable wiring, event functions (Phase 1)
- `ggml/src/ggml-vulkan-multigpu.cpp` — **new file** for split buffer type, cross-device staging, topology (Phases 2-5)
- `ggml/include/ggml-vulkan.h` — public API additions (split buffer type)
- `src/llama.cpp` — integration points for split mode graph

## Phases

1. [PHASE 1: Enable Async Interface and Events](PHASE1.md) — **COMPLETE**
2. [PHASE 2: Implement Vulkan Split Buffer Type](PHASE2.md) — **COMPLETE**
3. [PHASE 3: Cross-Device Tensor Copies via Host Staging](PHASE3.md) — **COMPLETE**
4. PHASE 4: Async Execution and Pipeline Parallelism
5. PHASE 5: Topology Discovery and Performance Tuning

## Upstreamability Notes

| Phase | Upstream Risk | Notes |
|-------|---------------|-------|
| 1 | Low | Enables existing disabled code + adds standard event API |
| 2 | Medium | New split buffer type, mirrors CUDA pattern exactly |
| 3 | Medium | Cross-device copy via staging, clean fallback path |
| 4 | Medium | Async pipeline, needs careful testing on multiple vendors |
| 5 | Low | Read-only topology query, no behavioral change |
