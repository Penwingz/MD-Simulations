"""
HDNNP model: multi-target GNN predicting energy_U0 and dipole_moment
using SchNet-style message passing and a charge-based dipole head.
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

        x = self.embedding(z)                  # (N, d_model)

        src_idx, dst_idx = edge_index[0], edge_index[1]

        if edge_index.size(1) > 0:
            r_ij = (pos[dst_idx] - pos[src_idx]).norm(dim=-1)          # (E,)
            e_ij = rbf_expansion(r_ij, self.n_rbf, 0.0, self.r_cutoff) # (E, n_rbf)
            fc = cosine_cutoff(r_ij, self.r_cutoff)                     # (E,)
            e_ij = e_ij * fc.unsqueeze(-1)                              # (E, n_rbf)
        else:
            # No edges — empty tensors so interaction blocks still receive valid input.
            e_ij = x.new_zeros(0, self.n_rbf)

        for block in self.interactions:
            x = block(x, edge_index, e_ij)

        eps_i = self.energy_head(x).squeeze(-1)              # (N,)
        energy = scatter_add(eps_i, batch_idx, dim_size=B)   # (B,)

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


# ─────────────────────────────────────────────────────────────────────────────
# PaiNN — Polarizable Atom Interaction Neural Network
# Schütt et al., "Equivariant message passing for the prediction of tensorial
# properties and molecular spectra", ICML 2021.
# ─────────────────────────────────────────────────────────────────────────────


def _vec_linear(v: Tensor, weight: Tensor) -> Tensor:
    """Apply a bias-free linear transform along the feature dimension of a vector tensor.

    Equivalent to nn.Linear(F_in, F_out, bias=False) applied to dim=1 of a
    (N, F_in, 3) tensor, producing (N, F_out, 3).

    Uses reshape + matmul instead of einsum for MPS compatibility.

    Args:
        v:      Vector features, shape (N, F_in, 3).
        weight: Weight matrix from nn.Linear(bias=False), shape (F_out, F_in).

    Returns:
        Transformed vector features, shape (N, F_out, 3).
    """
    N, F_in, _ = v.shape
    return (
        v.permute(0, 2, 1)           # (N, 3, F_in)
         .reshape(N * 3, F_in)       # (N*3, F_in)
         @ weight.t()                # (N*3, F_out)
    ).reshape(N, 3, -1).permute(0, 2, 1)   # (N, F_out, 3)


class _PaiNNMessageBlock(nn.Module):
    """Equivariant message block for PaiNN.

    Aggregates scalar and vector messages from neighbouring atoms, injecting
    directional information via per-edge unit vectors r̂_ij.

    Message equations (E = edges, F = n_features):
        W_s, W_v, W_r = filter_net(e_ij).chunk(3)   — (E, F) each
        φ_s, φ_v, φ_r = phi_net(s[dst]).chunk(3)     — (E, F) each
        m_s  = W_s * φ_s                              — (E, F) scalar
        m_v  = W_v[:,None]*v[dst] + (W_r*φ_r)[:,None]*r̂[:,None]  — (E, F, 3) vector

    Args:
        n_features: Feature dimension F.
        n_rbf:      Number of radial basis functions.
    """

    def __init__(self, n_features: int, n_rbf: int) -> None:
        super().__init__()
        self.phi_net = nn.Linear(n_features, 3 * n_features)
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, n_features),
            _ShiftedSoftplus(),
            nn.Linear(n_features, 3 * n_features),
        )

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        e_ij: Tensor,
        r_hat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute and aggregate equivariant messages.

        Args:
            s:          Scalar features, shape (N, F).
            v:          Vector features, shape (N, F, 3).
            edge_index: Edge connectivity (2, E). Row 0 = src, row 1 = dst.
            e_ij:       Cutoff-weighted RBF edge features, shape (E, n_rbf).
            r_hat:      Unit direction vectors (src→dst), shape (E, 3).

        Returns:
            Tuple (delta_s, delta_v):
                delta_s: aggregated scalar messages, shape (N, F).
                delta_v: aggregated vector messages, shape (N, F, 3).
        """
        N = s.size(0)
        F = s.size(1)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        W_s, W_v, W_r = self.filter_net(e_ij).chunk(3, dim=-1)       # each (E, F)
        phi_s, phi_v, phi_r = self.phi_net(s[dst_idx]).chunk(3, dim=-1)  # each (E, F)

        m_s = W_s * phi_s                                              # (E, F)

        # Vector from neighbour vectors: (E, F, 1) * (E, F, 3) = (E, F, 3)
        m_v_from_v = W_v.unsqueeze(-1) * v[dst_idx]

        # Vector from scalar projected onto unit direction:
        # (E, F, 1) * (E, 1, 3) = (E, F, 3)
        m_v_from_s = (W_r * phi_r).unsqueeze(-1) * r_hat.unsqueeze(-2)

        m_v = m_v_from_v + m_v_from_s                                  # (E, F, 3)

        delta_s = scatter_add(m_s, src_idx, dim_size=N)                # (N, F) — 2D ✓

        # 3D vector scatter — inline because utils.scatter_add only handles 2D index expansion
        delta_v = m_v.new_zeros(N, F, 3)
        delta_v.scatter_add_(0, src_idx.view(-1, 1, 1).expand_as(m_v), m_v)

        return delta_s, delta_v


class _PaiNNUpdateBlock(nn.Module):
    """Gated equivariant update block for PaiNN.

    Mixes scalar and vector channels through a learned gate that depends on
    both scalar features and the (invariant) per-channel vector norms.
    No bias on vector linear transforms — preserves equivariance.

    Update equations (N = atoms, F = n_features):
        Uv, Vv = LinearNoBias(v), LinearNoBias(v)        — (N, F, 3)
        a_ss, a_sv, a_vv = MLP([s, ‖Vv‖²]).chunk(3)      — (N, F)
        inner = Σ_xyz Uv*Vv                               — (N, F)
        s_new = s + a_ss + a_sv * inner
        v_new = v + a_vv[:,None] * Uv

    Args:
        n_features: Feature dimension F.
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.U_linear = nn.Linear(n_features, n_features, bias=False)
        self.V_linear = nn.Linear(n_features, n_features, bias=False)
        self.a_net = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            _ShiftedSoftplus(),
            nn.Linear(n_features, 3 * n_features),
        )

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Apply gated equivariant update.

        Args:
            s: Scalar features, shape (N, F).
            v: Vector features, shape (N, F, 3).

        Returns:
            Tuple (s_new, v_new):
                s_new: updated scalars, shape (N, F).
                v_new: updated vectors, shape (N, F, 3).
        """
        Uv = _vec_linear(v, self.U_linear.weight)        # (N, F, 3)
        Vv = _vec_linear(v, self.V_linear.weight)        # (N, F, 3)

        Vv_norm_sq = (Vv * Vv).sum(dim=-1)               # (N, F)

        a_ss, a_sv, a_vv = (
            self.a_net(torch.cat([s, Vv_norm_sq], dim=-1))  # (N, 3F)
            .chunk(3, dim=-1)
        )                                                 # each (N, F)

        inner = (Uv * Vv).sum(dim=-1)                    # (N, F)

        s_new = s + a_ss + a_sv * inner                  # (N, F)  — no in-place
        v_new = v + a_vv.unsqueeze(-1) * Uv              # (N, F, 3)  — no in-place

        return s_new, v_new


class PaiNNModel(nn.Module):
    """Polarizable Atom Interaction Neural Network (PaiNN).

    Extends SchNet-style message passing with equivariant vector features,
    enabling directional message propagation. The dipole moment is predicted
    directly from summed equivariant vectors — no indirect charge assignment.

    Architecture:
        1. Embedding: z → s (N, F);  v = 0 (N, F, 3)
        2. Edge geometry: r̂_ij (unit directions) + cutoff-weighted RBF
        3. n_interactions × (_PaiNNMessageBlock + _PaiNNUpdateBlock)
        4. Energy head: scalar MLP → scatter_add → E (B,)
        5. Dipole head: w·v → per-atom (N,3) → scatter_add → (B,3) → norm (B,)

    Args:
        config: OmegaConf DictConfig with a ``model`` sub-config containing:
            - n_features (int):     Feature dimension F.
            - n_interactions (int): Number of (Message+Update) block pairs.
            - n_rbf (int):          Number of radial basis functions.
            - r_cutoff (float):     Cutoff radius in Angstroms.
            - max_atomic_num (int): Size of the atomic-number embedding table.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        cfg = config.model
        self.n_features: int = cfg.n_features
        self.n_rbf: int = cfg.n_rbf
        self.r_cutoff: float = cfg.r_cutoff

        self.embedding = nn.Embedding(cfg.max_atomic_num + 1, cfg.n_features)

        self.message_blocks = nn.ModuleList([
            _PaiNNMessageBlock(n_features=cfg.n_features, n_rbf=cfg.n_rbf)
            for _ in range(cfg.n_interactions)
        ])
        self.update_blocks = nn.ModuleList([
            _PaiNNUpdateBlock(n_features=cfg.n_features)
            for _ in range(cfg.n_interactions)
        ])

        self.energy_head = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.n_features // 2),
            _ShiftedSoftplus(),
            nn.Linear(cfg.n_features // 2, 1),
        )

        # Learned linear mix of F equivariant vector channels → per-atom (N, 3)
        # bias=False: a constant bias vector would break rotational equivariance
        self.dipole_proj = nn.Linear(cfg.n_features, 1, bias=False)

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
                "charges": FloatTensor (N,) — zeros (PaiNN has no charge head)
        """
        z: Tensor = batch.z
        pos: Tensor = batch.pos
        edge_index: Tensor = batch.edge_index
        batch_idx: Tensor = batch.batch
        B: int = int(batch_idx.max().item()) + 1

        s = self.embedding(z)                                          # (N, F)
        v = s.new_zeros(s.size(0), s.size(1), 3)                       # (N, F, 3)

        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        if edge_index.size(1) > 0:
            r_vec = pos[dst_idx] - pos[src_idx]                        # (E, 3)
            r_norm = r_vec.norm(dim=-1)                                # (E,)
            r_hat = r_vec / r_norm.clamp(min=1e-8).unsqueeze(-1)       # (E, 3) — no in-place
            e_ij = rbf_expansion(r_norm, self.n_rbf, 0.0, self.r_cutoff)  # (E, n_rbf)
            fc = cosine_cutoff(r_norm, self.r_cutoff)                  # (E,)
            e_ij = e_ij * fc.unsqueeze(-1)                             # (E, n_rbf)
        else:
            e_ij = s.new_zeros(0, self.n_rbf)
            r_hat = s.new_zeros(0, 3)

        for msg_block, upd_block in zip(self.message_blocks, self.update_blocks):
            delta_s, delta_v = msg_block(s, v, edge_index, e_ij, r_hat)
            s = s + delta_s                                            # (N, F)
            v = v + delta_v                                            # (N, F, 3)
            s, v = upd_block(s, v)

        eps_i = self.energy_head(s).squeeze(-1)                        # (N,)
        energy = scatter_add(eps_i, batch_idx, dim_size=B)             # (B,)

        # Equivariant dipole: project F vector channels → per-atom (N, 3) → aggregate
        w = self.dipole_proj.weight[0]                                 # (F,)
        mu_i = (v * w.view(1, -1, 1)).sum(dim=1)                       # (N, 3)
        mu_mol = scatter_add(mu_i, batch_idx, dim_size=B)              # (B, 3)
        dipole = mu_mol.norm(dim=-1)                                   # (B,)

        charges = s.new_zeros(s.size(0))                               # (N,) API compat

        return {"energy": energy, "dipole": dipole, "charges": charges}
