"""
PyTorch Lightning module for the HDNNP training loop.

Wraps HDNNPModel with multi-task weighted MAE loss, AdamW optimiser,
and ReduceLROnPlateau scheduler.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.model import HDNNPModel, PaiNNModel


class MLIPLightningModule(pl.LightningModule):
    """Lightning training wrapper for HDNNPModel.

    Handles multi-task loss computation, metric logging in physical units,
    optimiser and LR scheduler configuration.

    Args:
        config: OmegaConf DictConfig loaded from configs/default.yaml.
                Must include a ``stats`` sub-config (injected by train.py
                from data/stats.json) with normalisation mean/std for
                ``energy_U0`` and ``dipole_moment``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        # Serialise to plain dict so checkpoints are compatible with
        # torch.load(weights_only=True) in PyTorch >= 2.6.
        if isinstance(config, DictConfig):
            config_container = OmegaConf.to_container(config, resolve=True)
        else:
            config_container = config  # already a plain dict (loaded from ckpt)
        self.save_hyperparameters({"config": config_container})
        # Reconstruct DictConfig for dot-access throughout this instance.
        self.config: DictConfig = OmegaConf.create(config_container)

        model_type = str(self.config.model.get("type", "hdnnp")).lower()
        if model_type == "painn":
            self.model = PaiNNModel(self.config)
        elif model_type == "hdnnp":
            self.model = HDNNPModel(self.config)
        else:
            raise ValueError(f"Unknown model.type: {model_type!r}")

        # Loss weights from config — use self.config (DictConfig), not raw arg
        self.lambda_energy: float = float(self.config.training.lambda_energy)
        self.lambda_dipole: float = float(self.config.training.lambda_dipole)

        # Column indices into the 19-column batch.y tensor
        # config.data.target_indices = [energy_col, dipole_col]
        target_indices = list(self.config.data.target_indices)
        self.energy_col: int = int(target_indices[0])  # e.g. 12 for energy_U0_atom
        self.dipole_col: int = int(target_indices[1])  # e.g. 0 for dipole_moment

        # Normalisation stats — used to convert normalised MAE → physical units
        stats = self.config.stats
        self.energy_mean: float = float(stats.energy_U0.mean)
        self.energy_std: float  = float(stats.energy_U0.std)
        self.dipole_mean: float = float(stats.dipole_moment.mean)
        self.dipole_std: float  = float(stats.dipole_moment.std)

    def forward(self, batch) -> dict[str, Tensor]:
        """Delegate to HDNNPModel forward.

        Args:
            batch: PyG Batch object.

        Returns:
            Dict with keys ``energy`` (B,), ``dipole`` (B,), ``charges`` (N,).
        """
        return self.model(batch)

    def _get_targets(self, batch) -> tuple[Tensor, Tensor]:
        """Extract normalised energy and dipole targets from batch.y.

        Args:
            batch: PyG Batch with ``y`` of shape (B, 19).

        Returns:
            Tuple of (e_true, d_true), each of shape (B,).
        """
        e_true = batch.y[:, self.energy_col]  # (B,) — normalised energy
        d_true = batch.y[:, self.dipole_col]  # (B,) — normalised dipole
        return e_true, d_true

    def training_step(self, batch, batch_idx: int) -> Tensor:
        """Compute multi-task weighted MAE loss and log training metrics.

        Args:
            batch:     PyG Batch.
            batch_idx: Batch index (unused, required by Lightning).

        Returns:
            Scalar loss tensor.
        """
        preds = self(batch)
        e_pred: Tensor = preds["energy"]   # (B,) normalised
        d_pred: Tensor = preds["dipole"]   # (B,) normalised
        e_true, d_true = self._get_targets(batch)

        mae_e = F.l1_loss(e_pred, e_true)
        mae_d = F.l1_loss(d_pred, d_true)
        loss = self.lambda_energy * mae_e + self.lambda_dipole * mae_d

        B = e_pred.size(0)
        self.log("train/loss",       loss,  on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        self.log("train/mae_energy", mae_e, on_step=True, on_epoch=True, batch_size=B)
        self.log("train/mae_dipole", mae_d, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        """Compute and log validation MAE in physical units (eV, Debye).

        MAE in physical units = MAE_normalised * std
        (MAE is scale-equivariant; mean cancels).

        Args:
            batch:     PyG Batch.
            batch_idx: Batch index (unused).
        """
        preds = self(batch)
        e_pred: Tensor = preds["energy"]
        d_pred: Tensor = preds["dipole"]
        e_true, d_true = self._get_targets(batch)

        mae_e_norm = F.l1_loss(e_pred, e_true)
        mae_d_norm = F.l1_loss(d_pred, d_true)
        loss = self.lambda_energy * mae_e_norm + self.lambda_dipole * mae_d_norm

        # Convert to physical units for interpretability
        mae_e_phys = mae_e_norm * self.energy_std
        mae_d_phys = mae_d_norm * self.dipole_std

        B = e_pred.size(0)
        self.log("val/loss",       loss,        on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val/mae_energy", mae_e_phys,  on_epoch=True, prog_bar=True, batch_size=B)
        self.log("val/mae_dipole", mae_d_phys,  on_epoch=True,                batch_size=B)

    def test_step(self, batch, batch_idx: int) -> None:
        """Compute and log test MAE in physical units.

        Args:
            batch:     PyG Batch.
            batch_idx: Batch index (unused).
        """
        preds = self(batch)
        e_pred: Tensor = preds["energy"]
        d_pred: Tensor = preds["dipole"]
        e_true, d_true = self._get_targets(batch)

        mae_e_norm = F.l1_loss(e_pred, e_true)
        mae_d_norm = F.l1_loss(d_pred, d_true)

        mae_e_phys = mae_e_norm * self.energy_std
        mae_d_phys = mae_d_norm * self.dipole_std

        B = e_pred.size(0)
        self.log("test/mae_energy", mae_e_phys, on_epoch=True, batch_size=B)
        self.log("test/mae_dipole", mae_d_phys, on_epoch=True, batch_size=B)

    def configure_optimizers(self):
        """Return AdamW optimiser and ReduceLROnPlateau scheduler.

        Returns:
            Dict with ``optimizer`` and ``lr_scheduler`` keys per Lightning convention.
        """
        cfg = self.config.training
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(cfg.scheduler.patience),
            factor=float(cfg.scheduler.factor),
            min_lr=float(cfg.scheduler.min_lr),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/mae_energy",
                "interval": "epoch",
                "frequency": 1,
            },
        }
