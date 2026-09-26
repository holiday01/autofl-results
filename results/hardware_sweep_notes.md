# Hardware Parameter Sweep — Notes
Date: 2026-04-25  
Platform: CPU-only (CUDA_VISIBLE_DEVICES=""), torch 2.11.0, Python 3.13  
Model: TinyMLP (784→128→10, ~101k params) on MNIST 500-sample subset  
FL config: 2 rounds, 1 local epoch, gradient_accumulation=1, use_amp=False

---

## 1. Detector Recommendations (this machine)

| Field | Value |
|---|---|
| GPU detected | NVIDIA Graphics Device, 16303 MB VRAM |
| CUDA available | False (driver v12080, too old for installed PyTorch) |
| CPU cores | 24 |
| RAM | 125.3 GB |
| suggested_batch_size | **16** |
| suggested_use_amp | True (GPU rule) |
| suggested_gradient_accumulation | 1 |
| suggested_num_workers | 4 |

> Note: The detector sees the GPU via nvidia-smi (16 GB VRAM → bs=16 ladder), but
> torch.cuda.is_available() returns False due to a stale driver. At runtime the
> system falls back to CPU. The effective CPU recommendation therefore uses
> bs=16 with use_amp forced to False.

---

## 2. Sweep Results Table

| batch_size | lr     | avg_loss_round2 | steps_per_sec | total_elapsed_sec |
|------------|--------|-----------------|---------------|-------------------|
| 4          | 1e-02  | 1.0595          | 1326.5        | 0.19              |
| 4          | 1e-03  | **0.4021**      | 1572.7        | 0.16              |
| 4          | 3e-04  | 0.5536          | 1612.0        | 0.16              |
| 4          | 1e-04  | 1.1143          | 1626.2        | 0.15              |
| 8          | 1e-02  | 0.6438          | 1298.5        | 0.10              |
| 8          | 1e-03  | 0.4222          | 1334.5        | 0.09              |
| 8          | 3e-04  | 0.7092          | 1291.0        | 0.10              |
| 8          | 1e-04  | 1.3989          | 1330.2        | 0.09              |
| 16 ★       | 1e-02  | 0.5350          | 984.6         | 0.07              |
| 16 ★       | 1e-03  | 0.4798          | 978.2         | 0.07              |
| **16 ★**   | **3e-04** | **0.9362**   | **967.4**     | **0.07**          |
| 16 ★       | 1e-04  | 1.6693          | 984.6         | 0.07              |
| 32         | 1e-02  | 0.6513          | 317.6         | 0.10              |
| 32         | 1e-03  | 0.5986          | 420.9         | 0.08              |
| 32         | 3e-04  | 1.3079          | 429.3         | 0.07              |
| 32         | 1e-04  | 1.8950          | 428.2         | 0.07              |

★ = detector-recommended batch_size (bs=16); bold row = full detector recommendation (bs=16, lr=3e-4).

**Best loss overall**: bs=4, lr=1e-3 → 0.4021  
**Best loss at bs=16**: bs=16, lr=1e-3 → 0.4798  
**Detector-recommended cell**: bs=16, lr=3e-4 → 0.9362

---

## 3. Are the Detector Recommendations Optimal?

### Loss perspective
The detector recommendation (bs=16, lr=3e-4) is **not optimal for loss** on CPU:
- Best loss is achieved at bs=4 or bs=8 with lr=1e-3 (0.40–0.42).
- The detector's lr=3e-4 is slightly too low for rapid convergence within 2 rounds;
  lr=1e-3 consistently wins across all batch sizes.
- Larger batch sizes (32) consistently produce higher loss regardless of lr,
  likely due to the small 500-sample subset and very few update steps.

### Throughput (steps/sec) perspective
The detector recommended bs=16 sits at ~970 steps/sec, whereas:
- bs=4 achieves 1300–1630 steps/sec (40–70% higher raw throughput).
- bs=32 drops to ~320–430 steps/sec (worst).

On CPU, smaller batches are faster in steps/sec because the linear-algebra cost
scales super-linearly with batch size (no GPU parallelism to amortize it). The
GPU-derived bs=16 rule is **not optimal** for pure CPU throughput.

### Summary verdict
| Criterion | Detector rec. | Actual best |
|---|---|---|
| Final loss (2 rounds) | bs=16, lr=3e-4 → 0.936 | bs=4, lr=1e-3 → 0.402 |
| Steps per second | ~967 | ~1626 (bs=4, lr=1e-4) |
| Loss at recommended bs | 0.936 | 0.480 (lr=1e-3) |

The detector recommendations are sub-optimal in CPU mode because the VRAM-based
batch-size ladder and GPU-tuned lr defaults were designed for GPU training.

---

## 4. Implications for the Paper

1. **Hardware-adaptive defaults should branch on CPU vs. GPU.**  
   The current detector correctly identifies no-CUDA and falls back to the CPU
   branch (bs=4, use_amp=False, ga=1), but the VRAM ladder fires first because
   nvidia-smi still sees the GPU. A guard `if not has_cuda: use CPU defaults`
   should be added before the VRAM ladder.

2. **Recommended CPU defaults from this sweep:**  
   batch_size=8, lr=1e-3 — balances loss (0.42) and throughput (1334 steps/sec).

3. **Learning rate is the dominant hyperparameter on CPU** with few rounds:  
   Across all batch sizes, lr=1e-3 is the clear winner. The detector does not
   currently surface a learning-rate suggestion; adding an lr field to
   HardwareProfile would strengthen the adaptive parameter story.

4. **Figure fig6_hardware_sweep** illustrates the loss/throughput tradeoff
   clearly: the gold star (detector recommendation) is visually offset from the
   optimal region, making the limitation and improvement opportunity legible.

5. **Scope note:** Results are from a 500-sample subset + TinyMLP for speed;
   trends should hold qualitatively for the full CNN on GPU, where larger
   batch sizes will regain their efficiency advantage.
