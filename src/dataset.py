"""
QM9 data module for the Phase 3 HDNNP pipeline.

Uses torch_geometric.datasets.QM9 (PyG HDF5 format) — NOT schnetpack.
Do NOT import from src/custom_loader.py here; that is a Phase 2 artifact.

See .claude/INSTRUCTIONS.md § src/dataset.py for the full spec.
"""

import json
import logging
import os
from typing import Optional

import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from src.utils import RadiusGraphTransform

log = logging.getLogger(__name__)


class QM9DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the QM9 dataset (PyG format).

    Handles graph construction, target normalisation, deterministic splitting,
    and DataLoader creation for the Phase 3 HDNNP pipeline.

    Targets normalised:
        - energy_U0  (index 12, atomisation energy in eV)
        - dipole_moment (index 0, Debye)

    Args:
        config: OmegaConf DictConfig loaded from configs/default.yaml.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.config = config
        self.train_dataset: Optional[list] = None
        self.val_dataset: Optional[list] = None
        self.test_dataset: Optional[list] = None
        self._stats: Optional[dict] = None

    # ── LightningDataModule interface ──────────────────────────────────────────

    def prepare_data(self) -> None:
        """Download QM9 and build radius-graph edges (called once, main process only).

        If ``data/qm9/processed/`` already exists the dataset is loaded from
        the cache; no network access or re-processing occurs.
        """
        pre_transform = RadiusGraphTransform(r=self.config.data.r_cutoff, loop=False)
        QM9(root=self.config.data.root, pre_transform=pre_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load, normalise, and split the dataset subset.

        Args:
            stage: ``'fit'``, ``'validate'``, ``'test'``, or ``None``
                   (sets up all splits).
        """
        self._stats = self._load_stats()

        pre_transform = RadiusGraphTransform(r=self.config.data.r_cutoff, loop=False)
        dataset = QM9(root=self.config.data.root, pre_transform=pre_transform)

        subset_size = self.config.data.subset_size
        if subset_size > len(dataset):
            log.warning(
                "subset_size=%d > dataset size=%d; clamping.",
                subset_size, len(dataset),
            )
            subset_size = len(dataset)

        # Seeded deterministic subset: same indices every run
        torch.manual_seed(self.config.seed)
        perm = torch.randperm(len(dataset)).tolist()
        subset_indices: list[int] = perm[:subset_size]

        # Split sizes (integer arithmetic; train absorbs rounding remainder)
        n = len(subset_indices)
        n_val = int(n * self.config.data.val_fraction)
        n_test = int(n * self.config.data.test_fraction)
        n_train = n - n_val - n_test

        train_idx = subset_indices[:n_train]
        val_idx = subset_indices[n_train : n_train + n_val]
        test_idx = subset_indices[n_train + n_val :]

        log.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        self.train_dataset = self._build_split(dataset, train_idx)
        self.val_dataset = self._build_split(dataset, val_idx)
        self.test_dataset = self._build_split(dataset, test_idx)

    def train_dataloader(self) -> DataLoader:
        """Return shuffled DataLoader for the training split."""
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for the validation split (no shuffle)."""
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for the test split (no shuffle)."""
        return self._make_loader(self.test_dataset, shuffle=False)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Normalisation statistics loaded from ``data/stats.json``.

        Available after :meth:`setup` has been called.

        Raises:
            RuntimeError: If called before :meth:`setup`.
        """
        if self._stats is None:
            raise RuntimeError("stats not available — call setup() first.")
        return self._stats

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_stats(self) -> dict:
        """Load normalisation statistics from ``data/stats.json``.

        Returns:
            Parsed JSON dict with ``energy_U0`` and ``dipole_moment`` entries.

        Raises:
            FileNotFoundError: If ``data/stats.json`` does not exist.
        """
        stats_path = os.path.join(
            os.path.dirname(os.path.abspath(self.config.data.root)), "stats.json"
        )
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"stats.json not found at {stats_path}. "
                "Run `python data/download.py --config configs/default.yaml` first."
            )
        with open(stats_path) as fh:
            return json.load(fh)

    def _build_split(self, dataset, indices: list[int]) -> list:
        """Return a list of normalised Data objects for the given molecule indices.

        Clones each PyG Data object and normalises ``y`` for the two target
        properties in-place on the clone (no in-place ops on original tensors).

        Args:
            dataset: Full PyG QM9 dataset (InMemoryDataset, already loaded).
            indices: Molecule indices to include in this split.

        Returns:
            List of ``torch_geometric.data.Data`` objects with normalised ``y``.
        """
        e_idx = self._stats["energy_U0"]["target_index"]
        d_idx = self._stats["dipole_moment"]["target_index"]
        e_mean = torch.tensor(self._stats["energy_U0"]["mean"], dtype=torch.float32)
        e_std  = torch.tensor(self._stats["energy_U0"]["std"],  dtype=torch.float32)
        d_mean = torch.tensor(self._stats["dipole_moment"]["mean"], dtype=torch.float32)
        d_std  = torch.tensor(self._stats["dipole_moment"]["std"],  dtype=torch.float32)

        samples = []
        for idx in indices:
            data = dataset[idx].clone()
            # Build new y — avoids any in-place op (MPS restriction)
            y = data.y.clone()  # (1, 19)
            y[0, e_idx] = (y[0, e_idx] - e_mean) / e_std
            y[0, d_idx] = (y[0, d_idx] - d_mean) / d_std
            data.y = y
            samples.append(data)
        return samples

    def _make_loader(self, split: list, shuffle: bool) -> DataLoader:
        """Create a PyG DataLoader for a data split.

        ``pin_memory`` is only activated for CUDA devices; MPS does not
        support pinned memory.

        Args:
            split:   List of PyG Data objects.
            shuffle: Whether to shuffle at each epoch.

        Returns:
            ``torch_geometric.loader.DataLoader`` instance.
        """
        pin_memory = self.config.data.pin_memory and torch.cuda.is_available()
        num_workers = self.config.data.num_workers
        return DataLoader(
            split,
            batch_size=self.config.data.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
