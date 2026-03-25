"""
Phase 3 evaluation entry point.

Loads a saved checkpoint and runs the test set, printing per-target MAE
in physical units (eV, Debye) with 95% bootstrap confidence intervals.

Usage:
    python evaluate.py --config configs/default.yaml --ckpt checkpoints/best.ckpt

See .claude/INSTRUCTIONS.md § evaluate.py for the full spec.

NOTE: This is Phase 3 only. It evaluates checkpoints produced by train.py.
"""

# ── Phase 3 — NOT YET IMPLEMENTED ─────────────────────────────────────────────
# Implement in TODO Phase 4 (Baseline Run & Evaluation).
#
# Steps (per INSTRUCTIONS.md):
#   1. Parse --config and --ckpt arguments
#   2. MLIPLightningModule.load_from_checkpoint(ckpt_path, config=config)
#   3. trainer.test(model, datamodule=QM9DataModule(config))
#   4. Print test/mae_energy (eV) and test/mae_dipole (Debye) with confidence intervals

raise NotImplementedError(
    "evaluate.py is not yet implemented.\n"
    "Complete train.py and run a full training first.\n"
    "See .claude/INSTRUCTIONS.md § evaluate.py for the spec."
)
