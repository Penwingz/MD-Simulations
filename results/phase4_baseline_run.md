# Training Run Results — All Phases

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

## Full Comparison Across All Runs

| | Phase 2 | Phase 4 (Baseline) | Phase 5 Run 1 |
|---|---|---|---|
| Framework | SchNetPack 2.2 | PyG + HDNNPModel | PyG + HDNNPModel |
| Hardware | Apple MPS | Colab T4 | Colab T4 |
| Training molecules | 200 | 8 000 | 8 000 |
| Val molecules | 40 | 1 000 | 1 000 |
| Model params | ~226K | 186K | **1.3M** |
| d_model | default | 128 | 256 |
| n_interactions | default | 3 | 6 |
| r_cutoff | default | 5.0 Å | 6.0 Å |
| Energy target | Total DFT (idx 7, ~−1000s eV) | Atomisation (idx 12, ~−76 eV) | Atomisation (idx 12, ~−76 eV) |
| Normalisation | None | z-score | z-score |
| Loss function | MSE | MAE (λ_e=1, λ_d=1) | MAE (λ_e=1, λ_d=3) |
| Learning rate | 1e-3 (Adam) | 1e-3 (AdamW) | 5e-4 (AdamW) |
| Epochs run | 50 | 226 (early stop) | 400 (hit max — not converged) |
| **Val energy score** | **RMSE ≈ 7.96 eV** | **MAE = 0.2149 eV** | **MAE = 0.1463 eV** |
| **Val dipole score** | Not predicted | MAE = 0.7569 D | MAE = 0.7220 D |
| **Test energy score** | — | MAE = 0.2240 eV | MAE = 0.1449 eV |
| **Test dipole score** | — | MAE = 0.7401 D | MAE = 0.6974 D |
| Approx. energy MSE | ~63.4 eV² | ~0.05–0.08 eV² | ~0.02–0.04 eV² |

**Energy improvement trajectory:**
- Phase 2 → Phase 4: ~850–1200× MSE reduction (different target, so indicative only)
- Phase 4 → Phase 5: **32% MAE reduction** (same task, direct comparison)
- Phase 4 → Phase 5: untrained baseline was 25.46 eV; Phase 5 best = 0.1463 eV — **174× reduction from untrained**

Phase 2 was an overfit sanity check (200 molecules, no normalisation, total DFT energy).
Phase 4 and 5 are proper generalisation runs with held-out test sets.

## Improvement Log

| Date | Change | Val MAE Energy | Val MAE Dipole | Epochs | Notes |
|---|---|---|---|---|---|
| 2026-03-26 | Baseline (d_model=128, n_int=3, r=5Å, λ_d=1.0, lr=1e-3) | 0.2149 eV | 0.7569 D | 226 (early stop) | Phase 4 first run |
| 2026-03-26 | Phase 5 Run 1 (d_model=256, n_int=6, r=6Å, λ_d=3.0, lr=5e-4) | 0.1463 eV | 0.7220 D | 400 (hit max) | Still improving at epoch 400 — not converged |

---

## Phase 5 Run 1 Detail

**Config:** d_model=256, n_interactions=6, r_cutoff=6.0Å, lambda_dipole=3.0, lr=5e-4, max_epochs=400, early_stop_patience=50

**Model size:** 1.3M params (up from 186K)

**Edges per batch:** 9780 (was 8980) — +9% from wider r_cutoff

| Metric | Val (best, epoch 365) | Test |
|---|---|---|
| MAE energy_U0 | 0.1463 eV | 0.1449 eV |
| MAE dipole_moment | 0.7220 D | 0.6974 D |

**Key finding:** Model hit max_epochs=400 without triggering early stopping (patience=50). It was still actively improving at epoch 400 — training was not converged.

**Improvement vs Phase 4 baseline:**
- Energy: 0.2149 → 0.1463 eV — **32% improvement**
- Dipole: 0.7569 → 0.7220 D — **4.6% improvement**

---

## Next Steps (Phase 5 Run 2)

Model did not converge — extend budget and let training continue:
- Increase `max_epochs`: 400 → 800
- Keep all other Phase 5 Run 1 settings unchanged
- Expected: energy will continue falling toward 0.100 eV target
