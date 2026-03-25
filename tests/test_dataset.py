"""
Tests for src/dataset.py — QM9DataModule.

Uses a small subset (100 molecules) of the already-downloaded QM9 data so
that each test completes in well under 30 seconds.  No network access needed.

Run:
    python -m pytest tests/test_dataset.py -v --tb=short
"""

import os

import pytest
import torch
from omegaconf import OmegaConf

from src.dataset import QM9DataModule

# ── Tiny config fixture ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_config():
    """OmegaConf config using 100 molecules so tests run fast."""
    cfg = OmegaConf.create({
        "seed": 42,
        "data": {
            "root": "data/qm9",
            "subset_size": 100,
            "r_cutoff": 5.0,
            "batch_size": 16,
            "num_workers": 0,          # no multiprocessing in tests
            "pin_memory": False,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "target_indices": [12, 0],
        },
    })
    return cfg


@pytest.fixture(scope="module")
def datamodule(tiny_config):
    """Set-up QM9DataModule once for the whole test module."""
    dm = QM9DataModule(tiny_config)
    dm.setup()
    return dm


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_datamodule_setup(datamodule):
    """setup() produces batches with the correct tensor shapes.

    Checks:
        batch.z          : (N_atoms,)     LongTensor — atomic numbers
        batch.pos        : (N_atoms, 3)   FloatTensor — 3D coordinates
        batch.y          : (B, 19)        FloatTensor — all QM9 targets
        batch.batch      : (N_atoms,)     LongTensor  — molecule index per atom
        batch.edge_index : (2, E)         LongTensor  — COO edges
    """
    assert datamodule.train_dataset is not None, "train_dataset not set"
    assert datamodule.val_dataset   is not None, "val_dataset not set"
    assert datamodule.test_dataset  is not None, "test_dataset not set"

    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    # --- atomic numbers ---
    assert batch.z.dim() == 1,                   f"z.dim={batch.z.dim()}, want 1"
    assert batch.z.dtype == torch.long,           f"z.dtype={batch.z.dtype}, want long"

    # --- positions ---
    assert batch.pos.dim() == 2,                  f"pos.dim={batch.pos.dim()}, want 2"
    assert batch.pos.size(1) == 3,                f"pos.shape[1]={batch.pos.size(1)}, want 3"
    assert batch.pos.dtype == torch.float32,      f"pos.dtype={batch.pos.dtype}, want float32"

    # --- targets ---
    assert batch.y.dim() == 2,                   f"y.dim={batch.y.dim()}, want 2"
    assert batch.y.size(1) == 19,                f"y.shape[1]={batch.y.size(1)}, want 19"
    assert batch.y.dtype == torch.float32,       f"y.dtype={batch.y.dtype}, want float32"
    # batch dimension matches number of molecules in the batch
    assert batch.y.size(0) == batch.batch.max().item() + 1, \
        "y.size(0) should equal number of molecules in batch"

    # --- batch vector ---
    assert batch.batch.dim() == 1,               f"batch.dim={batch.batch.dim()}, want 1"
    assert batch.batch.dtype == torch.long,      f"batch.dtype={batch.batch.dtype}, want long"
    assert batch.batch.size(0) == batch.z.size(0), "batch length must equal number of atoms"

    # --- edges (built by RadiusGraphTransform pre_transform) ---
    assert batch.edge_index.dim() == 2,          f"edge_index.dim={batch.edge_index.dim()}, want 2"
    assert batch.edge_index.size(0) == 2,        f"edge_index.shape[0]={batch.edge_index.size(0)}, want 2"
    assert batch.edge_index.dtype == torch.long, f"edge_index.dtype={batch.edge_index.dtype}, want long"
    assert batch.edge_index.size(1) > 0,         "no edges found — RadiusGraph may have failed"


def test_normalisation(datamodule):
    """Normalised energy and dipole values are centred near 0 on the training split.

    With the training-set statistics used for normalisation and 80 training
    samples, the sample mean of normalised values should be roughly in [-2, 2]
    (not equal to zero because the stats were computed on a different random
    sample, but definitely not near the raw-unit magnitudes of ~-76 eV or ~2.7 D).
    """
    stats = datamodule.stats
    e_idx = stats["energy_U0"]["target_index"]    # 12
    d_idx = stats["dipole_moment"]["target_index"] # 0

    # Collect all normalised training targets
    energies, dipoles = [], []
    for data in datamodule.train_dataset:
        energies.append(data.y[0, e_idx].item())
        dipoles.append(data.y[0, d_idx].item())

    e_tensor = torch.tensor(energies)
    d_tensor = torch.tensor(dipoles)

    # Values must NOT be in raw-unit range (which would indicate normalisation failed)
    raw_energy_mean = stats["energy_U0"]["mean"]    # ~-76 eV
    assert not (e_tensor.mean().abs() > 50), \
        f"energy mean {e_tensor.mean():.2f} looks like raw (un-normalised) values"

    raw_dipole_mean = stats["dipole_moment"]["mean"]  # ~2.66 Debye
    assert not (d_tensor.mean() > 1.5 and d_tensor.mean() < 4.0 and d_tensor.std() < 2.0), \
        f"dipole mean {d_tensor.mean():.2f} looks like raw (un-normalised) values"

    # Normalised values should be order-of-magnitude ~ [-5, 5]
    assert e_tensor.abs().max() < 20, \
        f"normalised energy max {e_tensor.abs().max():.2f} seems too large"
    assert d_tensor.abs().max() < 10, \
        f"normalised dipole max {d_tensor.abs().max():.2f} seems too large"


def test_split_sizes(tiny_config, datamodule):
    """Train / val / test sizes match config fractions with no index overlap."""
    n = tiny_config.data.subset_size          # 100
    n_val  = int(n * tiny_config.data.val_fraction)   # 10
    n_test = int(n * tiny_config.data.test_fraction)  # 10
    n_train = n - n_val - n_test                       # 80

    assert len(datamodule.train_dataset) == n_train, \
        f"train size: got {len(datamodule.train_dataset)}, expected {n_train}"
    assert len(datamodule.val_dataset) == n_val, \
        f"val size: got {len(datamodule.val_dataset)}, expected {n_val}"
    assert len(datamodule.test_dataset) == n_test, \
        f"test size: got {len(datamodule.test_dataset)}, expected {n_test}"

    # No overlap: molecule index (data.idx) must be unique across all splits
    def get_indices(split):
        return {data.idx.item() for data in split}

    train_ids = get_indices(datamodule.train_dataset)
    val_ids   = get_indices(datamodule.val_dataset)
    test_ids  = get_indices(datamodule.test_dataset)

    assert train_ids.isdisjoint(val_ids),  "overlap between train and val"
    assert train_ids.isdisjoint(test_ids), "overlap between train and test"
    assert val_ids.isdisjoint(test_ids),   "overlap between val and test"

    # Union covers exactly subset_size distinct molecules
    all_ids = train_ids | val_ids | test_ids
    assert len(all_ids) == n, \
        f"total unique molecules {len(all_ids)} != subset_size {n}"
