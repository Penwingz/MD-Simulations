# Phase 6 — PaiNN Equivariant Architecture Results

## Execution Date
2026-03-29

## Hardware
Google Colab, Tesla T4 GPU (15 360 MiB), CUDA 12.8, PyTorch 2.10.0+cu128, Lightning 2.6.1

## Motivation
Phase 5 SchNet (1.3M params) converged at 0.1596 eV energy MAE — 60% above the 0.100 eV target.
Root cause: SchNet is rotationally invariant only — it uses pairwise distances but no directional
information. The dipole head (charge MLP → Σ qᵢrᵢ → norm) computes charges from scalar-only
features and has no directional signal in the hidden representation.

PaiNN (Schütt et al., ICML 2021) adds an equivariant vector channel v_i (N, F, 3) alongside
scalar features s_i (N, F). Messages propagate both scalars and direction-weighted vectors,
enabling physically consistent dipole prediction via a direct vector readout.

---

## Architecture: PaiNN F=128, L=3

| Component | Detail |
|---|---|
| Scalar features s_i | (N, 128) — invariant |
| Vector features v_i | (N, 128, 3) — equivariant (rotate with molecule) |
| Message block | W_s·φ_s (scalar) + W_v·v_j + (W_r·φ_r)⊗r̂_ij (vector) |
| Update block | Gated: a_net([s, ‖Vv‖²]) gates both s and v channels |
| Energy readout | Scalar head → per-atom MLP → scatter_add |
| Dipole readout | w·v → (N,3) → scatter_add → (B,3) → norm — no charge assignment |
| Parameters | **672K** |

**vs Phase 5 SchNet:** 672K equivariant vs 1.3M invariant — **half the parameters**.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Subset | 10 000 molecules, train/val/test = 8 000/1 000/1 000 |
| Batch size | 32 |
| r_cutoff | 6.0 Å |
| Learning rate | 5e-4 (AdamW) |
| lambda_energy / lambda_dipole | 1.0 / 3.0 |
| max_epochs | 400 |
| Early stopping patience | 50 |
| Seed | 42 |

---

## Results

**Converged at epoch 261 · early-stopped at epoch 312**

| Metric | Val (best, epoch 261) | Test |
|---|---|---|
| MAE energy_U0 | **0.1070 eV** | 0.1179 eV |
| MAE dipole_moment | **0.6486 D** | 0.6398 D |

---

## Improvement vs Phase 5 SchNet (converged baseline)

| Metric | Phase 5 SchNet (1.3M) | Phase 6 PaiNN (672K) | Improvement |
|---|---|---|---|
| Val MAE energy | 0.1596 eV | **0.1070 eV** | **−33%** |
| Val MAE dipole | 0.7197 D | **0.6486 D** | **−10%** |
| Test MAE energy | 0.1597 eV | **0.1179 eV** | **−26%** |
| Test MAE dipole | 0.6945 D | **0.6398 D** | **−8%** |
| Parameters | 1 300K | **672K** | **−48%** |

**PaiNN with half the parameters outperforms SchNet by 33% on energy.** This directly validates
the equivariant architecture hypothesis: directional message passing is structurally more
informative per parameter than distance-only message passing.

---

## Targets Status

| Metric | Target | Phase 6 val | Gap |
|---|---|---|---|
| val/mae_energy | < 0.100 eV | 0.1070 eV | 7% over |
| val/mae_dipole | < 0.200 D | 0.6486 D | 224% over |

**Energy target is close — 7% above the 0.100 eV threshold.**
The dipole remains far from target. The equivariant vector readout improved dipole by 10% vs
SchNet, but the gap is still large. This suggests the dipole bottleneck is primarily a data/capacity
issue, not an architecture issue — the model has not seen enough molecular diversity to learn
the full electronic charge distribution from equilibrium geometries alone.

---

## Full Progression Summary

| Run | Params | Val MAE Energy | Val MAE Dipole | Test MAE Energy | Converged? |
|---|---|---|---|---|---|
| Phase 4 baseline (SchNet) | 186K | 0.2149 eV | 0.7569 D | 0.2240 eV | Yes (ep 195) |
| Phase 5 Run 2 (SchNet) | 1.3M | 0.1596 eV | 0.7197 D | 0.1597 eV | Yes (ep 288) |
| Phase 6 PaiNN F=128 L=3 | **672K** | **0.1070 eV** | **0.6486 D** | **0.1179 eV** | Yes (ep 261) |

Energy improvement Phase 4 → Phase 6: **0.2149 → 0.1070 eV — 50% total reduction**

---

## Analysis

### Why PaiNN outperforms SchNet with fewer parameters

SchNet propagates scalar features gated by pairwise distances. Two molecules with identical
distance matrices but different angles look identical to SchNet. QM9 molecules with the same
connectivity but different geometries (stereoisomers, conformers) are not distinguishable.

PaiNN's message block includes `(W_r·φ_r) ⊗ r̂_ij` — a per-edge vector weighted by both
the RBF filter and a scalar gate. This injects unit direction vectors into the feature space.
After aggregation, each atom's vector feature v_i knows the directionality of its neighbourhood.
The same number of scalar gates from neighbours now encodes both distance and direction —
roughly doubling the geometric information per message.

### Dipole — why the improvement is modest

The equivariant vector readout (w·v → Σ → norm) is theoretically correct for dipole. The 10%
improvement over SchNet's charge-based approach is real but small. Two reasons:

1. **Data limitation:** The model sees only 8k training molecules from a 10k fixed subset. The
   electronic charge distribution underlying the dipole requires diverse chemical context to learn.
   More training data would help more than architecture changes here.

2. **Magnitude scale:** The dipole MAE is still ~0.64 D. The mean dipole in QM9 is 2.66 D with
   std=1.48 D — our MAE in normalised units is ~0.43 std, meaning the model is not learning
   dipole well at all. Energy in normalised units is MAE/std ≈ 0.1070/10.184 ≈ 0.010 std —
   the model has learned energy very well relative to its variance, but dipole barely.

---

## Next Steps (Phase 7)

The energy target (0.100 eV) is within reach — 7% away with a converged model. Options:

| Intervention | Expected impact | Basis |
|---|---|---|
| **Scale PaiNN** (F=256, L=6 → ~5M params) | Energy ~0.06–0.08 eV | Capacity scaling |
| **Force supervision** (F = −∇E in loss) | Energy ~0.04–0.08 eV | Behler 2014/2016: forces are critical |
| **Larger dataset** (50k molecules) | Dipole likely improves most | Data limitation is primary dipole bottleneck |
| **TorchMD-Net / attention-based** | ~0.03–0.05 eV | 2022 state-of-the-art on QM9 |

Force supervision requires force labels. Standard QM9 does not include forces. The next practical
step without switching datasets is **scaled-up PaiNN** (F=256, L=6) — same architecture, more
capacity, no new data requirements.
