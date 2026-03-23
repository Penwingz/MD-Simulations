# Phase 2 Methodology

## Objective
- Overfit test on QM9 using SchNetPack and PyTorch Lightning, with strict hardware and memory constraints.

## Configuration
- Dataset: QM9 (SchNetPack 2.x)
- num_train: 200
- num_val: 40
- batch_size: 4
- Model: SchNet (default hidden channels)
- Potential: NeuralNetworkPotential (Atomwise, predicts energy_U0)
- Optimizer: Adam (lr=1e-3)
- Loss: MSE
- Trainer: PyTorch Lightning (accelerator='mps', devices=1, max_epochs=50)

## Hardware
- Apple Silicon (MPS backend, 8GB unified memory)

## Steps
1. Loader script (`src/custom_loader.py`) initializes QM9 with absolute pathing and robust error handling.
2. Training script (`src/train_overfit.py`) replicates loader setup, defines SchNet, wraps in NeuralNetworkPotential, and runs AtomisticTask with Lightning Trainer.
3. Results and configuration are saved in `results/phase2_overfit_run.txt`.

## Results
- Training completed for 50 epochs.
- Final validation loss: ~63.4
- See `results/phase2_overfit_run.txt` for full output.
