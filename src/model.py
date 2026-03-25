"""
HDNNP model for the Phase 3 pipeline.

Multi-target graph neural network predicting energy_U0 and dipole_moment
jointly using SchNet-style message passing and a charge-based dipole head.

See .claude/INSTRUCTIONS.md § src/model.py for the full architecture spec.
See .claude/SCHEMA.md § 2 for all intermediate tensor shapes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig

from src.utils import rbf_expansion, cosine_cutoff, scatter_add, shifted_softplus


class _ShiftedSoftplus(nn.Module):
    """Thin nn.Module wrapper around the shifted_softplus functional.

    Enables use of shifted_softplus inside nn.Sequential without a lambda
    (which is not picklable).
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply shifted softplus element-wise.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of the same shape as x.
        """
        return shifted_softplus(x)


class _FilterNetwork(nn.Module):
    """Two-layer MLP that maps RBF edge features to per-edge filter weights.

    Architecture: Linear(n_rbf → d_model) → ShiftedSoftplus → Linear(d_model → d_model)

    Args:
        n_rbf:   Number of radial basis functions (input size).
        d_model: Hidden and output dimensionality.
    """

    def __init__(self, n_rbf: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, d_model),
            _ShiftedSoftplus(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, e_ij: Tensor) -> Tensor:
        """Map edge RBF features to filter weights.

        Args:
            e_ij: Edge RBF features, shape (E, n_rbf).

        Returns:
            Filter weights, shape (E, d_model).
        """
        return self.net(e_ij)


class _InteractionBlock(nn.Module):
    """Single SchNet-style interaction block.

    Computes filtered messages from neighbours, aggregates them, and applies a
    two-layer update MLP with a residual connection.

    Architecture:
        W_ij  = filter_net(e_ij)                    (E, d_model)
        m_ij  = W_ij * x[dst_idx]                   (E, d_model)  element-wise
        m_i   = scatter_add(m_ij, src_idx, N)        (N, d_model)
        h_i   = update_net(m_i)                      (N, d_model)
        out_i = x + h_i                              (N, d_model)  residual

    Args:
        d_model: Feature dimensionality.
        n_rbf:   Number of radial basis functions fed into the filter network.
    """

    def __init__(self, d_model: int, n_rbf: int) -> None:
        super().__init__()
        self.filter_net = _FilterNetwork(n_rbf=n_rbf, d_model=d_model)
        self.update_net = nn.Sequential(
            nn.Linear(d_model, d_model),
            _ShiftedSoftplus(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor, edge_index: Tensor, e_ij: Tensor) -> Tensor:
        """Run one interaction step.

        Args:
            x:          Atom feature matrix, shape (N, d_model).
            edge_index: Edge connectivity, shape (2, E).  Row 0 = source
                        indices, row 1 = destination indices.
            e_ij:       Edge RBF features (already multiplied by cutoff),
                        shape (E, n_rbf).

        Returns:
            Updated atom features, shape (N, d_model).
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        W_ij = self.filter_net(e_ij)          # (E, d_model)
        m_ij = W_ij * x[dst_idx]              # (E, d_model) — element-wise product
        m_i = scatter_add(m_ij, src_idx, N)   # (N, d_model)

        return x + self.update_net(m_i)        # (N, d_model) — residual, no in-place


class HDNNPModel(nn.Module):
    """High-Dimensional Neural Network Potential with SchNet message passing.

    Jointly predicts molecular energy (energy_U0) and dipole moment
    (dipole_moment) from atomic numbers and 3D positions.

    Architecture:
        1. Embedding: z → x  (N, d_model)
        2. RBF expansion + cosine cutoff on pairwise distances
        3. n_interactions × _InteractionBlock
        4. Energy head: per-atom MLP → scatter_add → E  (B,)
        5. Charge head: per-atom MLP → charge neutralisation → dipole norm  (B,)

    Args:
        config: OmegaConf DictConfig with a ``model`` sub-config containing:
            - d_model (int):          Embedding / hidden dimensionality.
            - n_interactions (int):   Number of interaction blocks.
            - n_rbf (int):            Number of radial basis functions.
            - r_cutoff (float):       Cutoff radius in Angstroms.
            - max_atomic_num (int):   Size of the atomic-number embedding table.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        cfg = config.model
        self.d_model: int = cfg.d_model
        self.n_rbf: int = cfg.n_rbf
        self.r_cutoff: float = cfg.r_cutoff

        # Embedding layer — map atomic number to dense feature vector.
        # Index 0 is unused (atomic numbers start at 1); +1 for safe indexing.
        self.embedding = nn.Embedding(cfg.max_atomic_num + 1, cfg.d_model)

        # Stack of SchNet interaction blocks.
        self.interactions = nn.ModuleList(
            [_InteractionBlock(d_model=cfg.d_model, n_rbf=cfg.n_rbf)
             for _ in range(cfg.n_interactions)]
        )

        # Energy head: per-atom MLP → scalar energy contribution.
        self.energy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            _ShiftedSoftplus(),
            nn.Linear(cfg.d_model // 2, 1),
        )

        # Charge head: per-atom MLP → scalar partial charge.
        self.charge_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            _ShiftedSoftplus(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, batch) -> dict[str, Tensor]:
        """Run forward pass on a PyG batch.

        Args:
            batch: torch_geometric.data.Batch with fields:
                   z (N,)            — atomic numbers
                   pos (N, 3)        — Cartesian coordinates (Angstroms)
                   edge_index (2, E) — directed edges within r_cutoff
                   batch (N,)        — molecule index for each atom

        Returns:
            dict with keys:
                "energy":  FloatTensor (B,) — predicted energy (normalised units)
                "dipole":  FloatTensor (B,) — predicted dipole norm (normalised units)
                "charges": FloatTensor (N,) — per-atom partial charges (charge-neutral)
        """
        z: Tensor = batch.z                    # (N,)
        pos: Tensor = batch.pos                # (N, 3)
        edge_index: Tensor = batch.edge_index  # (2, E)
        batch_idx: Tensor = batch.batch        # (N,)
        B: int = int(batch_idx.max().item()) + 1

        # ── 1. Embed atomic numbers ───────────────────────────────────────────
        x = self.embedding(z)                  # (N, d_model)

        # ── 2. Compute edge features ──────────────────────────────────────────
        src_idx, dst_idx = edge_index[0], edge_index[1]

        if edge_index.size(1) > 0:
            r_ij = (pos[dst_idx] - pos[src_idx]).norm(dim=-1)          # (E,)
            e_ij = rbf_expansion(r_ij, self.n_rbf, 0.0, self.r_cutoff) # (E, n_rbf)
            fc = cosine_cutoff(r_ij, self.r_cutoff)                     # (E,)
            e_ij = e_ij * fc.unsqueeze(-1)                              # (E, n_rbf)
        else:
            # No edges — create empty tensors with correct shapes so the
            # interaction blocks receive valid (but vacuous) inputs.
            e_ij = x.new_zeros(0, self.n_rbf)

        # ── 3. Interaction blocks ─────────────────────────────────────────────
        for block in self.interactions:
            x = block(x, edge_index, e_ij)

        # ── 4. Energy head ────────────────────────────────────────────────────
        eps_i = self.energy_head(x).squeeze(-1)              # (N,)
        energy = scatter_add(eps_i, batch_idx, dim_size=B)   # (B,)

        # ── 5. Charge head (charge-neutralised dipole) ────────────────────────
        q_raw = self.charge_head(x).squeeze(-1)              # (N,)

        # Neutralise: subtract per-molecule mean charge from each atom.
        ones_N = torch.ones_like(q_raw)
        mol_q = scatter_add(q_raw, batch_idx, dim_size=B)    # (B,) total charge per mol
        n_atoms = scatter_add(ones_N, batch_idx, dim_size=B) # (B,) atom count per mol
        mean_q = (mol_q / n_atoms)[batch_idx]                # (N,) broadcast mean
        q_i = q_raw - mean_q                                 # (N,) neutralised — no in-place

        # Dipole: sum of charge-weighted positions, then take L2 norm.
        mu_vec = scatter_add(
            q_i.unsqueeze(-1) * pos, batch_idx, dim_size=B
        )                                                     # (B, 3)
        dipole = mu_vec.norm(dim=-1)                          # (B,)

        return {"energy": energy, "dipole": dipole, "charges": q_i}
