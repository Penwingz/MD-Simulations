# mdsims

A robust, memory-safe pipeline for molecular graph construction and neural network training on the QM9 dataset using SchNetPack and PyTorch Lightning, optimized for Apple Silicon (MPS backend).

## Features
- **Absolute pathing** and robust error handling for ASE/SQLite.
- **Configurable data splits** (Phase 2: 200 train / 40 val, batch size 4).
- **SchNet GNN** with NeuralNetworkPotential for energy_U0 prediction.
- **PyTorch Lightning** training loop with MPS acceleration.
- **Reproducible results** and configuration logging.

## Quickstart
1. Clone the repo and create a Python 3.14+ venv.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the loader to verify data pipeline:
   ```bash
   python src/custom_loader.py
   ```
4. Run the overfit training loop:
   ```bash
   python src/train_overfit.py
   ```

## Results
See `results/phase2_overfit_run.txt` for the full Phase 2 output and configuration.

## Methodology
- See `results/phase2_overfit_run.txt` for the exact configuration and terminal output for the successful Phase 2 run.
- All code is designed to be robust to database corruption and hardware memory constraints.

## License
MIT
