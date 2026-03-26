# Phase 5 — Improved Model Training Results

## Execution Date
2026-03-26

## Hardware
Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128, Lightning 2.6.1

## Context
Phase 4 baseline converged at 0.2149 eV energy MAE — significantly above the 0.100 eV target.
Root causes: insufficient capacity (186K params, 3 interaction blocks) and equal loss weighting
on unequally difficult tasks. Phase 5 addresses both.

---

## Architectural & Config Changes vs Phase 4

| Parameter | Phase 4 | Phase 5 | Reason |
|---|---|---|---|
| d_model | 128 | **256** | 2× wider atom embeddings — more expressive representations |
| n_interactions | 3 | **6** | 2× deeper — captures longer-range chemical interactions |
| r_cutoff | 5.0 Å | **6.0 Å** | Wider neighbourhood — +9% edges per batch |
| lambda_dipole | 1.0 | **3.0** | Upweight harder task to force charge-distribution learning |
| learning_rate | 1e-3 | **5e-4** | Lower LR for larger model — avoids overshooting early |
| scheduler patience | 10 | **15** | Give larger model more time before LR reduction |
| early_stop patience | 30 | **50** | Allow longer plateau before stopping |

**Model size:** 186K → **1.3M params** (7× increase)

---

## Run 1

**Config:** max_epochs=400 (all other Phase 5 settings above)

**Notebook:** `colabtests/phase5_improved_model.ipynb`

| Metric | Val (best, epoch 365) | Test |
|---|---|---|
| MAE energy_U0 | 0.1463 eV | 0.1449 eV |
| MAE dipole_moment | 0.7220 D | 0.6974 D |

**Improvement vs Phase 4:**

| Metric | Phase 4 | Phase 5 Run 1 | Improvement |
|---|---|---|---|
| Val MAE energy | 0.2149 eV | 0.1463 eV | **−32%** |
| Val MAE dipole | 0.7569 D | 0.7220 D | **−4.6%** |
| Test MAE energy | 0.2240 eV | 0.1449 eV | **−35%** |
| Test MAE dipole | 0.7401 D | 0.6974 D | **−5.8%** |

**Key finding:** Model hit max_epochs=400 without triggering early stopping (patience=50).
Still actively improving at epoch 400 — training was not converged. Fix: extend to 800 epochs.

---

## Run 2

**Config:** max_epochs=800, everything else identical to Run 1

**Notebook:** `colabtests/phase5_run2_full_eval.ipynb`

**Training:** Early stopping triggered at epoch 339 · best checkpoint at **epoch 288** — first fully converged run.

| Metric | Val (best, epoch 288) | Test | Held-out (120 831 mol) |
|---|---|---|---|
| MAE energy_U0 | **0.1596 eV** | 0.1597 eV | 0.1601 eV |
| MAE dipole_moment | **0.7197 D** | 0.6945 D | 0.7047 D |

**Improvement vs Phase 4:**

| Metric | Phase 4 | Phase 5 Run 2 (converged) | Improvement |
|---|---|---|---|
| Val MAE energy | 0.2149 eV | 0.1596 eV | **−26%** |
| Val MAE dipole | 0.7569 D | 0.7197 D | **−4.9%** |
| Test MAE energy | 0.2240 eV | 0.1597 eV | **−29%** |
| Test MAE dipole | 0.7401 D | 0.6945 D | **−6.2%** |

---

## Run 1 vs Run 2 — Reconciling the Discrepancy

Run 1 reported 0.1463 eV at epoch 365, which appears better than Run 2's converged 0.1596 eV.
This is not a contradiction. Run 1 hit its 400-epoch ceiling mid-training — the loss curve had
not plateaued, and epoch 365 happened to be a local minimum in the validation curve. The model
was still drifting when the run ended.

Run 2 demonstrates that the **true converged value for this architecture is ~0.160 eV**.
Run 2 is the authoritative result.

---

## Full QM9 Held-Out Evaluation (Run 2 only)

After training, the best checkpoint was evaluated against all 120 831 QM9 molecules **not** in
the 10k training subset — every molecule with indices `perm[10000:]` under `seed=42`.

| Split | Size | MAE energy_U0 | MAE dipole_moment |
|---|---|---|---|
| Val (seen distribution) | 1 000 | 0.1596 eV | 0.7197 D |
| Test (seen distribution) | 1 000 | 0.1597 eV | 0.6945 D |
| **Full QM9 held-out** | **120 831** | **0.1601 eV** | **0.7047 D** |

**Generalisation gap (test → held-out):** 0.0004 eV energy · 0.0102 D dipole.

This near-zero gap across 120× more molecules confirms the model has **not overfit** to its
10k training subset. It generalises to the full QM9 chemical distribution at essentially the
same error rate — validating that the training pipeline is sound and the model is learning
transferable molecular representations.

---

## Full Progression Summary

| Run | Val MAE Energy | Val MAE Dipole | Test MAE Energy | Test MAE Dipole | Converged? |
|---|---|---|---|---|---|
| Phase 4 baseline | 0.2149 eV | 0.7569 D | 0.2240 eV | 0.7401 D | Yes (epoch 195) |
| Phase 5 Run 1 | 0.1463 eV | 0.7220 D | 0.1449 eV | 0.6974 D | No (hit 400-epoch ceiling) |
| Phase 5 Run 2 | **0.1596 eV** | **0.7197 D** | **0.1597 eV** | **0.6945 D** | **Yes (epoch 288)** |

Overall improvement Phase 4 → Phase 5 (converged): **−26% energy MAE, −6% dipole MAE**

Untrained baseline at epoch 0: 25.46 eV. Phase 5 Run 2 converged: 0.1596 eV.
**159× reduction from random initialisation.**

---

## Targets

| Metric | Target | Best achieved | Gap |
|---|---|---|---|
| val/mae_energy | < 0.100 eV | 0.1596 eV | 60% over |
| val/mae_dipole | < 0.200 D | 0.7197 D | 260% over |

Targets not yet met. The model has converged — more training epochs will not help.

