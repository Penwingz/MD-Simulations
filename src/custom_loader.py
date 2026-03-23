
import os
import shutil
import traceback

import torch
from schnetpack.datasets import QM9
from schnetpack.transform import ASENeighborList, CastTo32

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def reset_data_directory(data_dir):
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)


def try_load_dataset(db_path, data_dir, retry=False):
    try:
        print(f"Attempting to initialize QM9 at {db_path}")
        dataset = QM9(
            db_path,
            4,
            transforms=[ASENeighborList(cutoff=5.0), CastTo32()],
            num_train=200,
            num_val=40,
            num_test=0,
        )
        print("Preparing dataset (this will download QM9 if not present)...")
        dataset.prepare_data()
        dataset.setup()
        print("Dataset prepared and setup successfully.")
        return dataset
    except AssertionError as e:
        print("AssertionError during dataset setup:")
        traceback.print_exc()
        if not retry:
            print("Corrupted ASE database detected. Resetting data directory and retrying once...")
            reset_data_directory(data_dir)
            return try_load_dataset(db_path, data_dir, retry=True)
        else:
            print(
                "Retry failed after data directory reset. "
                "Check network connectivity and write permissions for data/."
            )
            raise e
    except Exception as e:
        print("Unexpected error during dataset setup:")
        traceback.print_exc()
        raise e

def main():
    device = get_device()
    print(f"Using device: {device}")

    data_dir = os.path.abspath("./data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(data_dir, "qm9.db"))
    print(f"Database path set to: {db_path}")

    dataset = try_load_dataset(db_path, data_dir)

    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))

    print("\n--- Batch Tensor Information ---")
    for key, value in batch.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}: <nested dict>")
        else:
            print(f"{key}: type={type(value)}")

if __name__ == "__main__":
    main()