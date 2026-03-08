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

## Model

| Model | Quantization | Size | Parameters |
|-------|-------------|------|------------|
| TinyLlama 1.1B Chat v1.0 | Q2_K (Medium) | 459 MiB | 1.1 B |

Source: `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`

Small model chosen to fit in all test configurations including the 2 GB Polaris 12. Not representative of real multi-GPU workloads (model fits on a single GPU), but validates correctness and measures transfer overhead.

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

#### Single GPU vs Multi-GPU (125 token prompt, 4 eval tokens)

| Configuration | Prompt eval | Token gen | Load time |
|--------------|------------|-----------|-----------|
| RX 6800 XT alone | 53 ms (2372 tok/s) | 10.96 ms/tok (91 tok/s) | 149 ms |
| Vega alone | 92 ms (1354 tok/s) | 13.12 ms/tok (76 tok/s) | 375 ms |
| Both GPUs (smgs 1:1) | 131 ms (958 tok/s) | 15.78 ms/tok (63 tok/s) | 271 ms |

#### Analysis

Multi-GPU is **slower** than single GPU for this model. This is expected and correct:

1. **Model fits on one GPU**: TinyLlama at 459 MiB fits easily in 16 GB (6800 XT) or 8 GB (Vega). No need to split.
2. **Transfer overhead dominates**: Every token requires cross-device copies through host staging (GPU A → host → GPU B). For a model this small, the copy time exceeds the compute savings from splitting.
3. **Asymmetric GPUs**: The 6800 XT is ~1.75x faster than Vega for this workload. With 1:1 split, the faster GPU idles waiting for the slower one.
4. **Graph splits = 3**: Minimal parallelism opportunity. Each split boundary requires a synchronization point.

#### When Multi-GPU Helps

Multi-GPU split mode graph is designed for models that **don't fit** on a single GPU:

- **70B models** across 2x 24GB GPUs: each GPU gets ~35B worth of layers. Without splitting, the model simply doesn't load.
- **Large batch prompt processing**: With enough compute per split, the transfer overhead becomes negligible relative to matmul time.
- **Memory-bandwidth-bound workloads**: Splitting across GPUs effectively doubles available memory bandwidth.

The overhead measured here (~16 ms/tok for cross-device copies) would be negligible for a 70B model where single-token eval takes ~200-500ms on one GPU.

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
