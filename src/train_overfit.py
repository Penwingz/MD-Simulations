import os

import pytorch_lightning as pl
import torch
from torchmetrics import MeanSquaredError

import schnetpack.atomistic as spk_atomistic
import schnetpack.nn as spk_nn
from schnetpack.datasets import QM9
from schnetpack.model import NeuralNetworkPotential
from schnetpack.representation import SchNet
from schnetpack.task import AtomisticTask, ModelOutput
from schnetpack.transform import ASENeighborList, CastTo32


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataset() -> QM9:
    data_dir = os.path.abspath("./data")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(data_dir, "qm9.db"))

    dataset = QM9(
        db_path,
        4,
        transforms=[ASENeighborList(cutoff=5.0), CastTo32()],
        num_train=200,
        num_val=40,
        num_test=0,
    )
    dataset.prepare_data()
    dataset.setup()
    return dataset


def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    dataset = get_dataset()

    cutoff = 5.0
    n_atom_basis = 128
    n_interactions = 3
    radial_basis = spk_nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
    cutoff_fn = spk_nn.CosineCutoff(cutoff=cutoff)

    representation = SchNet(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    output_module = spk_atomistic.Atomwise(
        n_in=n_atom_basis,
        output_key=QM9.U0,
    )

    model = NeuralNetworkPotential(
        representation=representation,
        input_modules=[spk_atomistic.PairwiseDistances()],
        output_modules=[output_module],
    )

    output_energy = ModelOutput(
        name=QM9.U0,
        loss_fn=torch.nn.MSELoss(),
        metrics={"mse": MeanSquaredError()},
    )

    task = AtomisticTask(
        model=model,
        outputs=[output_energy],
        optimizer_cls=torch.optim.Adam,
        optimizer_args={"lr": 1e-3},
    )

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=50,
        log_every_n_steps=1,
    )
    trainer.fit(task, datamodule=dataset)


if __name__ == "__main__":
    main()
