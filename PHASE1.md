# PHASE 1: Enable Async Interface and Events

**Goal**: Un-NULL the async and event function pointers so the scheduler can use async execution with Vulkan backends.

## Steps

1. Fix `cpy_tensor_async` signature to match interface `(backend_src, backend_dst, src, dst)` → verify: compiles
2. Wire 4 async functions into vtable (un-NULL them) → verify: compiles, single-device works
3. Implement 5 event functions using timeline semaphores → verify: compiles
4. Wire event functions into vtable → verify: compiles, single-device regression test

## Status

- [ ] Step 1: Fix cpy_tensor_async signature
- [ ] Step 2: Wire async functions
- [ ] Step 3: Implement event functions
- [ ] Step 4: Wire event functions
