# Project Core
**Objective:** Phase 1 - Data Ingestion and Graph Construction. Convert 3D molecular coordinates from the QM9 dataset into PyTorch tensor graphs.
**Framework:** PyTorch & SchNetPack (version 2.x API).

# Hardware & Environment Constraints
* **System Limit:** Apple Silicon with a strict 8GB unified memory limit.
* **Hardware Acceleration:** Must utilize PyTorch's `mps` (Metal Performance Shaders) backend. 
* **Data Limits:** We must strictly limit data loading to a microscopic subset to prevent Out-Of-Memory (OOM) crashes.

# Database & Pathing Rules (CRITICAL)
* The `schnetpack.datasets.QM9` module relies on the Atomic Simulation Environment (ASE) SQLite backend. 
* ASE will throw an `AssertionError` if it attempts to open an empty, corrupted, or relative-pathed `.db` file. 
* **Rule 1:** All database paths must be dynamically resolved absolute paths using `os.path.abspath`. 
* **Rule 2:** If an `AssertionError` occurs from `ase/db/sqlite.py`, the agent must delete the `data/` directory and all hidden journal/wal files, then recreate the directory before attempting to run the script again.

# Current Task: `src/custom_loader.py`
The script must:
1. Guarantee an absolute path to `./data/qm9.db`.
2. Initialize the SchNetPack 2.0 QM9 dataset with `datapath` and `batch_size=4` as the first two positional arguments.
3. Apply `ASENeighborList` and `CastTo32` transforms.
4. Prepare the data, retrieve the dataloader, and print the tensor shapes of the first batch.