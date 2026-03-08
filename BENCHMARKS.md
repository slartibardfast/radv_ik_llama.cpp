# Benchmarks

## Test Hardware

### System A: Retro Computing Baseline

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon X5650 (Westmere, 2010) — 6C/12T, no AVX/AVX2, DDR3-1333 triple-channel |
| GPU 0 | AMD Radeon Pro WX 2100 (Polaris 12, 2017) — 512 SPs, 2 GB GDDR5, PCIe 3.0 x16 |
| GPU 1 | lavapipe (Mesa llvmpipe) — CPU-based Vulkan software rasterizer |
| Driver | Mesa RADV (Polaris12), Mesa llvmpipe |
| OS | Arch Linux, kernel 6.18.16-2-lts |

Deliberately terrible hardware to stress-test the implementation at the low end. The Polaris 12 is a passively-cooled workstation card with roughly the compute power of a 2013 midrange GPU. lavapipe has no actual GPU hardware — every Vulkan operation runs on CPU.

### System B: Real Multi-GPU

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 9 3950X (Zen 2, 2019) — 16C/32T, AVX2/FMA, DDR4 |
| GPU 0 | AMD Radeon RX 6800 XT (RDNA 2, Navi 21) — 16 GB GDDR6, PCIe 4.0 x16 |
| GPU 1 | AMD Radeon RX Vega 56/64 (GCN 5, Vega 10) — 8 GB HBM2, PCIe 3.0 x16 |
| Driver | Mesa RADV (Navi21 + Vega10) |
| OS | Arch Linux, kernel 6.12.74-1-lts |

Mixed-generation discrete AMD GPUs over PCIe. The 6800 XT (RDNA 2) is roughly 2x the compute throughput of the Vega (GCN 5). Different warp sizes (32 vs 64) and different PCIe generations test asymmetric device handling.

## Models

| Model | Quantization | Size | Parameters | Vulkan Graph Splits |
|-------|-------------|------|------------|-------------------|
| TinyLlama 1.1B Chat v1.0 | Q2_K | 459 MiB | 1.1 B | 2 (optimal) |
| Llama-2-7B | Q8_0 | 6.7 GiB | 6.7 B | 2 (optimal) |

Sources: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, `TheBloke/Llama-2-7B-GGUF`

### Model Compatibility Notes

Not all models work efficiently on the ik_llama.cpp Vulkan backend. Models with unsupported ops create excessive graph splits, causing constant GPU↔CPU bouncing:

| Model | Arch | Graph Splits | Status |
|-------|------|-------------|--------|
| TinyLlama 1.1B | llama (v1) | 2 | Fast |
| Llama-2-7B | llama (v2) | 2 | Fast |
| Llama-2-13B | llama (v2) | 2 | Fast |
| Llama-3.2-3B | llama | 58 | Very slow (unsupported ops per layer) |
| Llama-3.1-8B | llama | 66 | Very slow (same issue as 3.2) |
| Qwen3.5-4B | qwen35 (SSM+attn) | 306 | Very slow |
| GLM-4.7-Flash | deepseek2 (MoE) | 186 | Very slow |

Graph splits > ~5 on single GPU indicates ops falling back to CPU — a performance disaster regardless of GPU speed.

## Results

### System A: Retro Baseline (Polaris 12 + lavapipe)

#### Phase Evolution (multi-GPU, 2 token prompt, 4 eval tokens)

| Phase | Prompt eval | Token gen | Total | Key change |
|-------|------------|-----------|-------|------------|
| Phase 6 (sync copies) | 15378 ms | 685 ms/tok | 18127 ms | Baseline |
| Phase 7 (async copies) | 862 ms | 560 ms/tok | 3110 ms | Deferred fence wait |
| Phase 8 (staging pool) | 893 ms | 537 ms/tok | 3051 ms | Per-copy staging |
| Phase 9 (parallel upload) | ~1112 ms load | 537 ms/tok | ~3035 ms | Threaded uploads |

Key finding: Phase 7 delivered a **17.8x prompt eval speedup** by eliminating synchronous fence waits during cross-device copies.

#### Load Time: Serial vs Parallel Upload (Phase 9)

| Mode | Average (3 trials) |
|------|-------------------|
| Serial upload | 1117 ms |
| Parallel upload | 1112 ms |

No improvement because lavapipe's "uploads" are instant memcpy. Benefit requires two discrete GPUs.

### System B: Real Multi-GPU (RX 6800 XT + Vega)

#### TinyLlama 1.1B Q2_K (125 token prompt, 4 eval tokens)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 53 ms (2372 tok/s) | 10.96 ms/tok (91 tok/s) | 149 ms |
| Vega alone | 92 ms (1354 tok/s) | 13.12 ms/tok (76 tok/s) | 375 ms |
| Both GPUs (smgs 1:1) | 131 ms (958 tok/s) | 15.78 ms/tok (63 tok/s) | 271 ms |

#### Llama-2-7B Q8_0 (127 token prompt, 4 eval tokens, best of 3)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 166 ms (767 tok/s) | 25.95 ms/tok (38.5 tok/s) | 1919 ms |
| Vega alone | 306 ms (416 tok/s) | 33.07 ms/tok (30.2 tok/s) | 5293 ms |
| Both GPUs (smgs 1:1) | 396 ms (321 tok/s) | 41.12 ms/tok (24.3 tok/s) | 3554 ms |

#### Analysis

Multi-GPU is **slower** than single GPU for both models. This is expected and correct:

1. **Both models fit on one GPU**: TinyLlama (459 MiB) and Llama-2-7B (6.7 GiB) both fit in 16 GB (6800 XT). No need to split.
2. **Transfer overhead dominates**: Every token requires cross-device copies through host staging (GPU A → host → GPU B). For models this size, the copy time (~15 ms/tok) exceeds the compute savings from splitting.
3. **Asymmetric GPUs**: The 6800 XT is ~1.3-1.8x faster than Vega depending on workload. With 1:1 split, the faster GPU idles waiting for the slower one.
4. **Graph splits = 3**: Minimal parallelism opportunity. Each split boundary requires a synchronization point.

#### Cross-device transfer overhead

Measured consistently across both models:

| Model | Single GPU best | Multi-GPU | Overhead per token |
|-------|----------------|-----------|-------------------|
| TinyLlama 1.1B | 11.0 ms/tok | 15.8 ms/tok | ~5 ms |
| Llama-2-7B | 26.0 ms/tok | 41.1 ms/tok | ~15 ms |

The overhead scales with model size (more data to transfer per split boundary). For a 70B model where single-token eval takes ~200-500ms, this ~15 ms overhead would be negligible (<5%).

#### When Multi-GPU Helps

Multi-GPU split mode graph is designed for models that **don't fit** on a single GPU:

- **70B models** across 2x 24GB GPUs: each GPU gets ~35B worth of layers. Without splitting, the model simply doesn't load.
- **Large batch prompt processing**: With enough compute per split, the transfer overhead becomes negligible relative to matmul time.
- **Memory-bandwidth-bound workloads**: Splitting across GPUs effectively doubles available memory bandwidth.

The overhead measured here would be negligible for larger models where compute time per token is much higher.

#### Llama-2-13B Q8_0 (127 token prompt, 4 eval tokens, best of 3)

13 GB model — fits on 6800 XT alone (14.6 GB with KV cache), but a natural candidate for splitting.

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 275 ms (462 tok/s) | 37.64 ms/tok (26.6 tok/s) | 4068 ms |
| Both GPUs smgs 1:1 | 547 ms (232 tok/s) | 63.52 ms/tok (15.7 tok/s) | 7058 ms |
| Both GPUs smgs 2:1 | 502 ms (253 tok/s) | 59.28 ms/tok (16.9 tok/s) | 5998 ms |

Weights distribution (smgs 1:1): Vulkan0 = 6750 MiB, Vulkan1 = 6274 MiB. KV cache split: Vulkan0 = 840 MiB, Vulkan1 = 760 MiB.

Multi-GPU is still slower because the model fits on one GPU. The 2:1 split is ~7% faster than 1:1 — weighting toward the faster 6800 XT reduces idle time.

#### Transfer overhead scaling

| Model | Size | Single GPU | Multi-GPU 1:1 | Overhead/tok |
|-------|------|-----------|--------------|-------------|
| TinyLlama 1.1B | 459 MiB | 11.0 ms/tok | 15.8 ms/tok | ~5 ms |
| Llama-2-7B | 6.7 GiB | 26.0 ms/tok | 41.1 ms/tok | ~15 ms |
| Llama-2-13B | 13.0 GiB | 37.6 ms/tok | 63.5 ms/tok | ~26 ms |

The overhead scales roughly with model size. For models large enough to genuinely require splitting (>16 GB), the compute savings from not running out of VRAM far outweigh this transfer cost — the alternative is not being able to run at all.

## Runtime Flags

| Flag | Purpose |
|------|---------|
| `-smgs` | Enable split mode graph scheduling |
| `-ts 1,1` | Tensor split ratio (equal across 2 devices) |
| `-ngl 99` | Offload all layers to GPU |
| `-no-fa` | Disable flash attention (required on non-AVX2) |
| `-no-fug` | Disable fused up-gate (required on non-AVX2) |
| `-c 2048` | Context size (reduce for VRAM-limited configs) |
| `GGML_VK_VISIBLE_DEVICES=0,1` | Include non-discrete devices like lavapipe |
