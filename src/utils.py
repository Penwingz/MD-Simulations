"""
Utility functions for the Phase 3 HDNNP pipeline.

Implements device selection, activation functions, radial basis functions,
and cutoff functions used across the model.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data


# RadiusGraph transform — shared between data/download.py and src/dataset.py

class RadiusGraphTransform:
    """Build radius-graph edges using torch.cdist (no torch-cluster dependency).

    For each ordered pair (i, j) where i != j and dist(i, j) < r, adds a
    directed edge i → j.  Equivalent to
    ``torch_geometric.transforms.RadiusGraph`` but pure-PyTorch, MPS-safe.

    Args:
        r:    Cutoff radius in Angstroms.
        loop: If True, include self-loops (i == j).  Default False.
    """

    def __init__(self, r: float, loop: bool = False) -> None:
        self.r = r
        self.loop = loop

    def __call__(self, data: Data) -> Data:
        """Apply transform to a single molecular graph.

        Args:
            data: PyG Data object with ``pos`` attribute (N, 3).

        Returns:
            Same Data object with ``edge_index`` (2, E) and
            ``edge_attr`` (E, 1) appended.
        """
        pos = data.pos
        dist = torch.cdist(pos.float(), pos.float())  # (N, N)

        mask = dist < self.r
        if not self.loop:
            n = pos.size(0)
            eye = torch.eye(n, dtype=torch.bool, device=pos.device)
            mask = mask & ~eye

        edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)
        src, dst = edge_index[0], edge_index[1]
        data.edge_index = edge_index
        data.edge_attr = dist[src, dst].unsqueeze(-1)  # (E, 1)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r}, loop={self.loop})"


def get_device() -> torch.device:
    """Return the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def shifted_softplus(x: Tensor) -> Tensor:
    """Shifted softplus activation: log(0.5 * exp(x) + 0.5).

    Numerically stable implementation via:
        softplus(x) - log(2)
    which equals log(exp(x) + 1) - log(2) = log(0.5 * exp(x) + 0.5).

    Smooth, non-saturating activation suitable for energy prediction.
    Gradient-friendly on MPS. No in-place ops.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of the same shape as x.
    """
    return F.softplus(x) - math.log(2)


def rbf_expansion(r: Tensor, n_rbf: int, r_min: float, r_max: float) -> Tensor:
    """Expand pairwise distances into Gaussian radial basis functions.

    Centres are evenly spaced between r_min and r_max with spacing
    ``(r_max - r_min) / n_rbf``.  The width parameter gamma is set to
    ``0.5 / spacing ** 2`` so neighbouring Gaussians overlap at half-height.

    Args:
        r:     Pairwise distances, shape (E,).
        n_rbf: Number of basis functions.
        r_min: Minimum centre position (Angstroms).
        r_max: Maximum centre position / cutoff (Angstroms).

    Returns:
        Basis function values, shape (E, n_rbf).
    """
    k = torch.arange(n_rbf, device=r.device, dtype=r.dtype)
    spacing = (r_max - r_min) / n_rbf
    centers = r_min + k * spacing          # (n_rbf,)
    gamma = 0.5 / (spacing ** 2)
    return torch.exp(-gamma * (r.unsqueeze(-1) - centers) ** 2)  # (E, n_rbf)


def cosine_cutoff(r: Tensor, r_cutoff: float) -> Tensor:
    """Cosine cutoff envelope: 0.5 * (cos(pi * r / r_cutoff) + 1) for r < r_cutoff, else 0.

    Smoothly reduces interactions to zero at the cutoff boundary.
    No in-place ops (MPS safe).

    Args:
        r:        Pairwise distances, shape (E,).
        r_cutoff: Cutoff radius in Angstroms.

    Returns:
        Envelope values in [0, 1], shape (E,).
    """
    cutoff_vals = 0.5 * (torch.cos(torch.pi * r / r_cutoff) + 1.0)
    return torch.where(r < r_cutoff, cutoff_vals, torch.zeros_like(cutoff_vals))


def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Scatter-add src values into an output tensor indexed by index.

    Aggregates values from ``src`` at positions given by ``index`` into an
    output buffer of size ``(dim_size, *src.shape[1:])``.  The operation is:

        out[index[i], ...] += src[i, ...]

    The in-place ``scatter_add_`` is called on a freshly allocated zero tensor,
    which is safe for autograd (gradients flow through correctly).

    Args:
        src:      Source tensor of shape (N,) or (N, D, ...).
        index:    1-D LongTensor of shape (N,) mapping each row of src to an
                  output row.  Values must be in [0, dim_size).
        dim_size: Size of the output's first dimension (e.g. number of
                  molecules B or number of atoms N).

    Returns:
        Output tensor of shape (dim_size,) when src is 1-D, or
        (dim_size, *src.shape[1:]) when src is multi-dimensional.
    """
    if src.dim() == 1:
        out = src.new_zeros(dim_size)
        out.scatter_add_(0, index, src)
    else:
        out_shape = (dim_size,) + src.shape[1:]
        out = src.new_zeros(out_shape)
        expanded_index = index.unsqueeze(-1).expand_as(src)
        out.scatter_add_(0, expanded_index, src)
    return out
