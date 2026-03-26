# Phase 4 — Baseline Full Training Run

## Execution Date
2026-03-26

## Hardware
Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128, Lightning 2.6.1

## Purpose
First full training run on the 10k QM9 subset using the custom HDNNPModel pipeline built
in Phase 3. Establishes a baseline before any hyperparameter tuning.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Subset size | 10 000 molecules |
| Train / Val / Test | 8 000 / 1 000 / 1 000 |
| Batch size | 32 |
| Model | HDNNPModel — 186K params |
| d_model | 128 |
| n_interactions | 3 |
| n_rbf | 50 |
| r_cutoff | 5.0 Å |
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Gradient clip | 1.0 |
| Loss | Weighted MAE: λ_e·MAE(E) + λ_d·MAE(μ), λ=1.0 each |
| Max epochs | 300 |
| Early stopping | patience=30, monitor=val/mae_energy |
| Seed | 42 |

---

## Normalisation Statistics (10k subset, seed=42)

| Target | Raw mean | Raw std |
|---|---|---|
| energy_U0 (eV) | −76.191 | 10.184 |
| dipole_moment (Debye) | 2.661 | 1.476 |

Normalised training set: mean ≈ −0.005, std ≈ 0.997 (energy); mean ≈ 0.003, std ≈ 1.004 (dipole).

---

## Training Outcome

- **Stopped at epoch:** 226
- **Best checkpoint:** epoch 195

| Metric | Val (best) | Test |
|---|---|---|
| MAE energy_U0 | 0.2149 eV | 0.2240 eV |
| MAE dipole_moment | 0.7569 D | 0.7401 D |

Targets not yet met (< 0.100 eV, < 0.200 D). Model is learning meaningfully — untrained
baseline was **25.46 eV** (val/mae_energy at epoch 0), so the trained model is **118× better**
than random initialisation.

---

## Diagnosis

Two root causes identified for the gap to target:

**1. Insufficient model capacity.**
d_model=128 with 3 interaction blocks (186K params) is too shallow to capture the complexity
of molecular PES over 8k diverse QM9 molecules. Longer-range interactions are missed.

**2. Imbalanced loss weighting.**
Both tasks used λ=1.0. Dipole is harder (requires spatially consistent per-atom charge
assignments). Equal weighting let the easier energy task dominate gradients.

These are addressed in Phase 5.

---

## Comparison with Phase 2

| | Phase 2 (SchNetPack baseline) | Phase 4 (this run) |
|---|---|---|
| Framework | SchNetPack 2.2 | PyG + HDNNPModel |
| Hardware | Apple MPS | Colab T4 |
| Training molecules | 200 | 8 000 |
| Energy target | Total DFT (~−1000s eV) | Atomisation (~−76 eV) |
| Normalisation | None | z-score |
| Loss | MSE | Weighted MAE |
| **Energy score** | **RMSE ≈ 7.96 eV** | **MAE = 0.2149 eV** |
| **Dipole score** | Not predicted | MAE = 0.7569 D |

Phase 2 was an overfit sanity check (200 molecules, no normalisation, total DFT energy).
Phase 4 is the first proper generalisation run with a held-out test set.
