"""
Tests for src/lightning_module.py — MLIPLightningModule.

Uses synthetic mock data — no real QM9 data needed. Completes in < 30 seconds.

Run:
    python -m pytest tests/test_training.py -v --tb=short
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.lightning_module import MLIPLightningModule


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_config() -> OmegaConf:
    """Tiny config for tests — small model, no real data paths needed."""
    return OmegaConf.create({
        "seed": 0,
        "data": {
            "root": "data/qm9",
            "subset_size": 100,
            "r_cutoff": 5.0,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "val_fraction": 0.1,
            "test_fraction": 0.1,
            "target_indices": [12, 0],
        },
        "model": {
            "d_model": 32,
            "n_interactions": 1,
            "n_rbf": 10,
            "r_cutoff": 5.0,
            "max_atomic_num": 10,
        },
        "training": {
            "max_epochs": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "gradient_clip_val": 1.0,
            "lambda_energy": 1.0,
            "lambda_dipole": 1.0,
            "scheduler": {
                "patience": 5,
                "factor": 0.5,
                "min_lr": 1.0e-6,
            },
        },
        # Mock normalisation stats — no stats.json needed in tests
        "stats": {
            "energy_U0": {
                "mean": -76.0,
                "std": 10.0,
                "unit": "eV",
                "target_index": 12,
            },
            "dipole_moment": {
                "mean": 2.5,
                "std": 1.5,
                "unit": "Debye",
                "target_index": 0,
            },
            "computed_on_subset": 100,
            "seed": 0,
        },
    })


def _make_mock_data(n_mols: int = 16, seed: int = 0) -> list[Data]:
    """Create synthetic PyG Data objects with valid graph structure.

    Each molecule has 4–7 atoms, positions drawn from N(0,1)*2 Å,
    and edges built by a 5 Å radius graph (via torch.cdist).

    Args:
        n_mols: Number of molecules to generate.
        seed:   Random seed for reproducibility.

    Returns:
        List of PyG Data objects ready for Batch.from_data_list.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    data_list = []
    for i in range(n_mols):
        N = int(torch.randint(4, 8, (1,), generator=rng).item())
        z = torch.randint(1, 9, (N,), generator=rng)
        pos = torch.randn(N, 3, generator=rng) * 2.0

        # Build radius graph (r < 5.0 Å, no self-loops)
        dists = torch.cdist(pos, pos)
        mask = (dists < 5.0) & ~torch.eye(N, dtype=torch.bool)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()

        # y: (1, 19) with all zeros except the two target columns
        y = torch.zeros(1, 19)
        y[0, 12] = torch.randn(1, generator=rng).item()  # normalised energy
        y[0, 0]  = torch.randn(1, generator=rng).item()  # normalised dipole

        data = Data(z=z, pos=pos, y=y, edge_index=edge_index,
                    idx=torch.tensor(i))
        data_list.append(data)
    return data_list


def _make_dataloader(n_mols: int = 16, batch_size: int = 4, seed: int = 0):
    """Build a PyG DataLoader from mock data.

    Args:
        n_mols:     Total number of mock molecules.
        batch_size: Molecules per batch.
        seed:       RNG seed for data generation.

    Returns:
        torch_geometric.loader.DataLoader instance.
    """
    from torch_geometric.loader import DataLoader
    data_list = _make_mock_data(n_mols=n_mols, seed=seed)
    return DataLoader(data_list, batch_size=batch_size, shuffle=False)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_train_one_epoch():
    """Training loop runs for one epoch without errors and produces finite loss.

    Verifies:
    - MLIPLightningModule instantiates with mock config.
    - Trainer.fit completes without raising.
    - The logged train/loss value is a finite positive number.
    """
    pl.seed_everything(0, workers=True)
    config = _make_mock_config()
    module = MLIPLightningModule(config)

    train_dl = _make_dataloader(n_mols=16, batch_size=4, seed=0)
    val_dl   = _make_dataloader(n_mols=8,  batch_size=4, seed=1)

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,          # suppress TensorBoard / CSV writer in tests
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Confirm a finite training loss was recorded
    logged = trainer.logged_metrics
    assert "train/loss_epoch" in logged, \
        f"Expected 'train/loss_epoch' in logged_metrics; got keys: {list(logged.keys())}"
    loss_val = float(logged["train/loss_epoch"])
    assert torch.isfinite(torch.tensor(loss_val)), \
        f"Training loss is not finite: {loss_val}"
    assert loss_val > 0, f"Training loss should be positive, got {loss_val}"


def test_checkpoint_save_load():
    """Checkpoint saved by Trainer reloads with identical forward-pass output.

    Verifies:
    - Trainer saves a checkpoint after training.
    - MLIPLightningModule.load_from_checkpoint loads it without errors.
    - Model predictions on a fixed batch are identical before and after reload.
    """
    pl.seed_everything(0, workers=True)
    config = _make_mock_config()
    module = MLIPLightningModule(config)

    train_dl = _make_dataloader(n_mols=16, batch_size=4, seed=0)
    val_dl   = _make_dataloader(n_mols=8,  batch_size=4, seed=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "last.ckpt"

        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=tmpdir,
                    filename="last",
                    save_last=True,
                    save_top_k=0,   # only keep last
                )
            ],
        )
        trainer.fit(module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        # Find the saved checkpoint
        saved_ckpts = list(Path(tmpdir).glob("*.ckpt"))
        assert len(saved_ckpts) >= 1, "No checkpoint was saved"
        saved_path = str(saved_ckpts[0])

        # Reload the module from checkpoint
        reloaded = MLIPLightningModule.load_from_checkpoint(saved_path)
        reloaded.eval()
        module.eval()

        # Run forward on a fixed batch and compare outputs
        data_list = _make_mock_data(n_mols=4, seed=42)
        batch = Batch.from_data_list(data_list)

        with torch.no_grad():
            out_orig     = module(batch)
            out_reloaded = reloaded(batch)

        torch.testing.assert_close(
            out_orig["energy"], out_reloaded["energy"],
            msg="Energy predictions differ after checkpoint reload"
        )
        torch.testing.assert_close(
            out_orig["dipole"], out_reloaded["dipole"],
            msg="Dipole predictions differ after checkpoint reload"
        )
