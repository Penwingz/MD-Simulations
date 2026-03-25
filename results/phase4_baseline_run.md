# Phase 4 — Baseline Training Run

## Execution Date
2026-03-26

## Hardware
Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128, Lightning 2.6.1

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

## Normalisation Statistics (10k subset, seed=42)

| Target | Raw mean | Raw std |
|---|---|---|
| energy_U0 (eV) | −76.191 | 10.184 |
| dipole_moment (Debye) | 2.661 | 1.476 |

Normalised training set: mean ≈ −0.005, std ≈ 0.997 (energy); mean ≈ 0.003, std ≈ 1.004 (dipole).

## Training Outcome

- **Stopped at epoch:** 226
- **Best checkpoint:** epoch 195

| Metric | Val (best) | Test |
|---|---|---|
| MAE energy_U0 | 0.2149 eV | 0.2240 eV |
| MAE dipole_moment | 0.7569 D | 0.7401 D |

Targets not yet met (< 0.100 eV, < 0.200 D). Model is learning meaningfully — untrained
baseline was 25.46 eV (val/mae_energy at epoch 0).

## Comparison with Phase 2 Baseline

Phase 2 used SchNetPack on Apple Silicon MPS, 200 training molecules, 50 epochs, no
normalisation, predicting raw total DFT energy (index 7, ~−1000s eV range).

| | Phase 2 | Phase 4 |
|---|---|---|
| Training molecules | 200 | 8 000 |
| Val molecules | 40 | 1 000 |
| Energy target | Total DFT (~−1000s eV) | Atomisation (~−76 eV) |
| Normalisation | None | z-score |
| Final val score | 63.4 eV² (MSE) → RMSE ≈ 7.96 eV | MAE = 0.224 eV |
| Approx. MSE | 63.4 eV² | ~0.050–0.075 eV² |

The approximate MSE improvement is ~850–1 200×. Phase 2 was an overfit sanity check;
Phase 4 is a proper generalisation run on held-out molecules.

## Improvement Log

| Date | Change | Val MAE Energy | Val MAE Dipole | Notes |
|---|---|---|---|---|
| 2026-03-26 | Baseline (d_model=128, n_int=3) | 0.2149 eV | 0.7569 D | Phase 4 first run |

## Next Steps (Phase 5)

- Increase `n_interactions`: 3 → 4 or 5
- Increase `d_model`: 128 → 256
- Tune `lambda_dipole` — dipole is 3.7× further from target than energy; may benefit from upweighting
- Experiment with `r_cutoff`: 5.0 Å → 6.0 Å
- Consider larger subset if memory permits
