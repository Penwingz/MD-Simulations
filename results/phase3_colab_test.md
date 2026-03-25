# Phase 3 — Architecture & Colab Pipeline Test

## Execution Date
2026-03-25

## Purpose
Document the Phase 3 architecture design, the differences vs the Phase 2
SchNetPack baseline, and the results of the Colab T4 pipeline validation test
(Phase 1: data pipeline verification).

---

## Colab Test Results (Phase 1 — Data Pipeline)

**Hardware:** Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128

**Subset used for test:** 200 molecules (project training default: 10 000)

**All 6 pipeline tests: PASS**

| Test | Result |
|---|---|
| `RadiusGraphTransform` (pure-PyTorch, no torch-cluster) | PASS |
| QM9 download + `stats.json` saved | PASS |
| `QM9DataModule.setup()` | PASS |
| Batch tensor shapes (z, pos, y, batch, edge_index) | PASS |
| Normalised target range | PASS |
| Split sizes + zero overlap | PASS |

**Normalisation statistics (200-molecule subset, seed=42):**

| Target | Raw mean | Raw std | Norm mean (train) | Norm std (train) |
|---|---|---|---|---|
| energy_U0 (eV) | −76.089 | 10.783 | 0.0003 | 0.963 |
| dipole_moment (Debye) | 2.640 | 1.413 | 0.024 | 1.018 |

Normalised means ≈ 0 and stds ≈ 1 confirm the normalisation pipeline is correct.

**DataLoader batch shapes verified (batch_size=16):**

```
batch.z.shape          = (276,)       dtype=int64   — atomic numbers
batch.pos.shape        = (276, 3)     dtype=float32 — 3D coordinates
batch.y.shape          = (16, 19)     dtype=float32 — all QM9 targets
batch.batch.shape      = (276,)       dtype=int64   — molecule index per atom
batch.edge_index.shape = (2, 4196)    dtype=int64   — radius-graph edges
```

**Colab test config:**
```yaml
seed: 42
data:
  root: /content/data/qm9
  subset_size: 200
  r_cutoff: 5.0
  batch_size: 16
  num_workers: 0
  pin_memory: true        # auto-activates on CUDA
  val_fraction: 0.1
  test_fraction: 0.1
  target_indices: [12, 0] # energy_U0_atom (eV), dipole_moment (Debye)
```

---

## Comparison with Phase 2 Baseline

### Phase 2 (SchNetPack baseline)

| Property | Value |
|---|---|
| Library | SchNetPack 2.2.0 |
| Model | SchNet (default config, ~226K params) |
| Training samples | 200 |
| Validation samples | 40 |
| Batch size | 4 |
| Target | energy_U0 only (single-task) |
| Target index | 7 (total DFT energy, ~−1000s eV) |
| Normalisation | None |
| Loss | MSE |
| Optimiser | Adam (lr=1e-3) |
| Epochs | 50 |
| Hardware | Apple Silicon MPS |
| **Final val_loss** | **~63.4 (MSE, un-normalised eV²)** |

### Phase 3 (HDNNP — this project)

| Property | Value |
|---|---|
| Library | PyTorch Geometric 2.7.0 |
| Model | Custom HDNNPModel (SchNet interactions + charge head) |
| Training samples | 8 000 (80% of 10 000 subset) |
| Validation samples | 1 000 |
| Test samples | 1 000 |
| Batch size | 32 |
| Targets | energy_U0 + dipole_moment (multi-task) |
| Target indices | 12 (atomisation energy, ~−76 eV mean), 0 (dipole, ~2.6 D mean) |
| Normalisation | z-score per target (mean=0, std=1) |
| Loss | Weighted MAE: λ_e·MAE(E) + λ_d·MAE(μ) |
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Gradient clipping | 1.0 |
| Hardware | Apple Silicon MPS / CUDA (T4 verified) |
| **Trained val MAE** | **TBD — Phase 3 training not yet run** |

---

## Why a Direct Numeric Comparison Isn't Valid (Yet)

The Phase 2 val_loss of **63.4** is:
- MSE (squared errors), not MAE
- On **raw total DFT energy** (~−1000 to −5000 eV range), not atomisation energy
- Over only **40 validation molecules**

An MSE of 63.4 eV² on un-normalised total energies translates to an RMSE of ~7.96 eV,
which is very large — this is consistent with an overfit test on 200 molecules with no
normalisation and a single-target objective.

Phase 3 reports MAE in physical units (eV, Debye) on normalised targets with 1 000
validation molecules. Comparison will be meaningful once Phase 3 training completes.

---

## Architectural Improvements in Phase 3

| Dimension | Phase 2 | Phase 3 | Why it matters |
|---|---|---|---|
| Target quantity | 1 (energy) | 2 (energy + dipole) | Multi-task forces better representations |
| Energy reference | Total DFT (~−1000s eV) | Atomisation (~−76 eV mean) | Smaller dynamic range → easier to learn |
| Normalisation | None | z-score (mean=0, std=1) | Stable gradients, comparable loss scales |
| Loss | MSE | MAE | Robust to outliers; physically interpretable |
| Dipole computation | Not predicted | Charge-based: μ = Σ q_i·r_i | Grounded in electrostatics; guarantees charge neutrality |
| Training set size | 200 | 8 000 | 40× more molecules |
| Optimiser | Adam | AdamW + LR scheduler | Weight decay + adaptive LR = better generalisation |
| Graph edges | ASENeighborList (cached) | RadiusGraph(r=5Å) via torch.cdist | Pure-PyTorch, MPS/CUDA safe |
| Data format | ASE SQLite (.db) | PyG HDF5 (pre-processed .pt) | Faster random access during training |

---

## Expected Performance Targets (after Phase 3 training)

| Metric | Target |
|---|---|
| val MAE energy_U0 | < 0.100 eV |
| val MAE dipole_moment | < 0.200 Debye |
| Peak memory (batch=32) | < 8 GB |

Stretch goals: energy MAE < 0.050 eV, dipole MAE < 0.100 Debye.

---

## Colab Test Results (Phase 3 — Training Loop)

**Hardware:** Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128, Lightning 2.6.1

**Subset used for test:** 200 molecules (project training default: 10 000)

**All 4 pipeline tests: PASS**

| Test | Result |
|---|---|
| Test 1 — Data pipeline (shapes, normalisation) | PASS |
| Test 2 — Model forward (shapes, charge neutrality) | PASS |
| Test 3 — Training loop (fast_dev_run, finite loss) | PASS |
| Test 4 — Checkpoint save/load (identical outputs) | PASS |

**Test 1 — Batch tensor shapes (batch_size=16):**

```
batch.z.shape          = (276,)       dtype=int64   — atomic numbers
batch.pos.shape        = (276, 3)     dtype=float32 — 3D coordinates
batch.y.shape          = (16, 19)     dtype=float32 — all QM9 targets
batch.batch.shape      = (276,)       dtype=int64   — molecule index per atom
batch.edge_index.shape = (2, 4196)    dtype=int64   — radius-graph edges
energy (norm): mean=-0.0000, std=1.0000
dipole (norm): mean=-0.0000, std=1.0000
```

**Test 2 — Model forward pass (HDNNPModel, 186K params):**

```
device:            cuda
energy.shape:      (16,)
dipole.shape:      (16,)
charges.shape:     (276,)
energy range:      [1.202, 2.594]
dipole range:      [0.322, 1.354]
charge neutrality: all 16 molecules sum to ~0
```

**Test 3 — Untrained baseline (fast_dev_run=True, 1 batch):**

```
train/loss:        4.7362
val/mae_energy:    25.4643 eV   (untrained — expected to be large)
val/mae_dipole:    1.1232 Debye
```

**Test 4 — Checkpoint:**

```
Checkpoint saved and reloaded successfully
Energy predictions match: True
Dipole predictions match: True
```

---

## Next Step

Run full training on the 10 000-molecule subset. Record final val/mae_energy and val/mae_dipole.
