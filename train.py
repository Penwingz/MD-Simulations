"""
Phase 3 training entry point.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml data.subset_size=5000
    python train.py --config configs/default.yaml trainer.fast_dev_run=true

NOTE: This is the Phase 3 entry point. For Phase 2 training, use:
    python src/train_overfit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from src.dataset import QM9DataModule
from src.lightning_module import MLIPLightningModule


def _load_config(config_path: str, overrides: list[str]) -> OmegaConf:
    """Load YAML config and apply CLI dot-list overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides:   List of ``key=value`` strings from the CLI.

    Returns:
        Merged OmegaConf DictConfig.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def main() -> None:
    """Parse arguments, set up trainer, and run training."""
    parser = argparse.ArgumentParser(description="Train HDNNP on QM9 (Phase 3)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    config = _load_config(args.config, overrides)

    pl.seed_everything(int(config.seed), workers=True)

    # ── Load normalisation stats and inject into config ─────────────────────────
    stats_path = Path("data/stats.json")
    if not stats_path.exists():
        print(
            f"ERROR: {stats_path} not found.\n"
            "Run `python data/download.py` first to compute normalisation stats.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(stats_path) as f:
        raw_stats = json.load(f)

    # Merge stats into config so they are saved with the checkpoint hparams.
    config = OmegaConf.merge(config, OmegaConf.create({"stats": raw_stats}))

    # ── Instantiate datamodule and model ────────────────────────────────────────
    datamodule = QM9DataModule(config)
    model = MLIPLightningModule(config)

    # ── Trainer flags ───────────────────────────────────────────────────────────
    fast_dev_run = OmegaConf.select(config, "trainer.fast_dev_run", default=False)

    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoint.dirpath,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=int(config.checkpoint.save_top_k),
            filename="epoch={epoch:03d}-val_mae_e={val/mae_energy:.4f}",
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=config.early_stopping.monitor,
            patience=int(config.early_stopping.patience),
            mode=config.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=int(config.training.max_epochs),
        accelerator="auto",
        devices=1,
        gradient_clip_val=float(config.training.gradient_clip_val),
        log_every_n_steps=int(config.logging.log_every_n_steps),
        callbacks=callbacks,
        fast_dev_run=bool(fast_dev_run),
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # ── Print final validation metrics ──────────────────────────────────────────
    if not fast_dev_run:
        print("\n=== Final Validation Metrics ===")
        val_results = trainer.validate(model, datamodule=datamodule, verbose=False)
        if val_results:
            for key, value in val_results[0].items():
                print(f"  {key}: {value:.4f}")
        print("================================")


if __name__ == "__main__":
    main()
