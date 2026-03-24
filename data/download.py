"""
Phase 3 data download and statistics computation script.

Downloads QM9 via torch_geometric, builds RadiusGraph edges (pre_transform),
computes normalisation statistics for energy_U0 and dipole_moment on the
configured subset, and saves them to data/stats.json.

Usage:
    python data/download.py --config configs/default.yaml

See .claude/INSTRUCTIONS.md § data/download.py for the full spec.

NOTE: This uses torch_geometric.datasets.QM9 — NOT schnetpack.
      The schnetpack-based data lives in data/qm9.db and is managed by
      src/custom_loader.py (Phase 2 — do not modify).

NOTE on torch-cluster:
    torch_geometric.transforms.RadiusGraph depends on torch-cluster, which
    has no pre-built wheel for Python 3.14 / torch 2.10.  We use a custom
    pure-PyTorch implementation (RadiusGraphTransform) that is equivalent,
    MPS-safe, and has no external C-extension dependency.
"""

import argparse
import json
import logging
import os
import sys

import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# QM9 target indices used in this project (PyG QM9 ordering)
# Index 12 = U0_atom: atomisation energy at 0 K (eV) — ~-5 to -30 eV per molecule
# Index  0 = mu:      dipole moment norm (Debye)
# NOTE: index 7 is total DFT energy (~-1000s eV); index 1 is polarizability.
TARGET_ENERGY_U0 = 12   # atomisation energy at 0 K (eV)
TARGET_DIPOLE = 0       # dipole moment norm (Debye)


class RadiusGraphTransform:
    """Build radius-graph edges using torch.cdist (no torch-cluster dependency).

    For each ordered pair (i, j) where i != j and dist(i, j) < r_cutoff,
    adds a directed edge i → j.  Equivalent to
    torch_geometric.transforms.RadiusGraph but pure-PyTorch, MPS-safe.

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
            ``edge_attr`` (E, 1) set.
        """
        pos = data.pos  # (N, 3)
        dist = torch.cdist(pos.float(), pos.float())  # (N, N)

        mask = dist < self.r
        if not self.loop:
            n = pos.size(0)
            eye = torch.eye(n, dtype=torch.bool, device=pos.device)
            mask = mask & ~eye

        # COO edge_index: (2, E) — [source_row, target_col]
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()  # (2, E)

        src, dst = edge_index[0], edge_index[1]
        data.edge_index = edge_index
        data.edge_attr = dist[src, dst].unsqueeze(-1)  # (E, 1)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r}, loop={self.loop})"


def compute_stats(dataset, subset_indices: list[int], seed: int) -> dict:
    """Compute mean and std for energy_U0 and dipole_moment over a subset.

    Args:
        dataset:        Full PyG QM9 dataset.
        subset_indices: Indices of molecules in the chosen subset.
        seed:           Seed used to select the subset (for provenance).

    Returns:
        Stats dict matching SCHEMA.md § Normalisation Stats File format.
    """
    log.info("Computing normalisation statistics over %d molecules …", len(subset_indices))

    energy_vals: list[float] = []
    dipole_vals: list[float] = []

    for idx in subset_indices:
        y = dataset[idx].y  # (1, 19)
        energy_vals.append(y[0, TARGET_ENERGY_U0].item())
        dipole_vals.append(y[0, TARGET_DIPOLE].item())

    energy_t = torch.tensor(energy_vals, dtype=torch.float64)
    dipole_t = torch.tensor(dipole_vals, dtype=torch.float64)

    return {
        "energy_U0": {
            "mean":         float(energy_t.mean()),
            "std":          float(energy_t.std()),
            "unit":         "eV",
            "target_index": TARGET_ENERGY_U0,  # 12 = U0_atom
        },
        "dipole_moment": {
            "mean":         float(dipole_t.mean()),
            "std":          float(dipole_t.std()),
            "unit":         "Debye",
            "target_index": TARGET_DIPOLE,     # 0 = mu
        },
        "computed_on_subset": len(subset_indices),
        "seed": seed,
    }


def main() -> None:
    """Download QM9, build radius graphs, compute and save normalisation stats."""
    parser = argparse.ArgumentParser(
        description="Download QM9 and compute normalisation stats."
    )
    parser.add_argument("--config", required=True, help="Path to configs/default.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        log.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = OmegaConf.load(args.config)
    seed: int = config.seed
    root: str = config.data.root
    subset_size: int = config.data.subset_size
    r_cutoff: float = config.data.r_cutoff

    log.info(
        "Config loaded — seed=%d, subset_size=%d, r_cutoff=%.1f Å",
        seed, subset_size, r_cutoff,
    )

    # Remove any empty/partial processed directory so PyG re-processes cleanly.
    processed_dir = os.path.join(root, "processed")
    if os.path.isdir(processed_dir) and not os.listdir(processed_dir):
        log.info("Removing empty processed directory (previous failed run) …")
        os.rmdir(processed_dir)

    log.info("Downloading / loading QM9 into '%s' …", root)
    pre_transform = RadiusGraphTransform(r=r_cutoff, loop=False)
    dataset = QM9(root=root, pre_transform=pre_transform)

    log.info("Full dataset size: %d molecules", len(dataset))

    if subset_size > len(dataset):
        log.warning(
            "subset_size=%d exceeds dataset size=%d; clamping.",
            subset_size, len(dataset),
        )
        subset_size = len(dataset)

    # Deterministic seeded subset: shuffle all indices, take first subset_size.
    torch.manual_seed(seed)
    perm = torch.randperm(len(dataset)).tolist()
    subset_indices: list[int] = perm[:subset_size]
    log.info("Subset: %d molecules selected (seed=%d)", len(subset_indices), seed)

    # Verify a sample molecule has edges
    sample = dataset[subset_indices[0]]
    log.info(
        "Sample molecule — atoms: %d, edges: %d",
        sample.pos.size(0),
        sample.edge_index.size(1),
    )

    stats = compute_stats(dataset, subset_indices, seed)

    log.info("")
    log.info("=== Normalisation Statistics ===")
    log.info(
        "  energy_U0      mean = %+.4f eV      std = %.4f eV",
        stats["energy_U0"]["mean"], stats["energy_U0"]["std"],
    )
    log.info(
        "  dipole_moment  mean = %+.4f Debye    std = %.4f Debye",
        stats["dipole_moment"]["mean"], stats["dipole_moment"]["std"],
    )
    log.info("================================")
    log.info("")

    # Save stats adjacent to the dataset root: data/stats.json
    stats_path = os.path.join(os.path.dirname(os.path.abspath(root)), "stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    log.info("Saved normalisation stats → %s", stats_path)


if __name__ == "__main__":
    main()
