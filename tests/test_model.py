"""
Tests for HDNNPModel.

Uses mock batches (5 atoms, 2 molecules) — no real QM9 data.
Run: python -m pytest tests/test_model.py -v
"""

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

from src.model import HDNNPModel
from src.utils import RadiusGraphTransform, get_device


TINY_CFG = OmegaConf.create({
    "model": {
        "d_model": 32,
        "n_interactions": 2,
        "n_rbf": 20,
        "r_cutoff": 5.0,
        "max_atomic_num": 10,
    }
})


def make_batch(z_list: list, pos_list: list, r_cutoff: float = 5.0) -> Batch:
    """Build a PyG Batch from lists of z-arrays and position arrays.

    Applies RadiusGraphTransform to add edges.

    Args:
        z_list:   List of atomic-number lists, one per molecule.
        pos_list: List of position arrays (list of [x, y, z] triples), one per molecule.
        r_cutoff: Cutoff radius passed to RadiusGraphTransform.

    Returns:
        A PyG Batch containing all molecules with edges assigned.
    """
    transform = RadiusGraphTransform(r=r_cutoff)
    data_list = [
        transform(
            Data(
                z=torch.tensor(z, dtype=torch.long),
                pos=torch.tensor(p, dtype=torch.float32),
            )
        )
        for z, p in zip(z_list, pos_list)
    ]
    return Batch.from_data_list(data_list)


def test_forward_pass() -> None:
    """Forward pass on mock batch returns correct output shapes.

    Uses 2 molecules:
        mol0: 3 atoms — C, H, H  (z=[6,1,1])
        mol1: 2 atoms — O, H     (z=[8,1])

    Checks:
        output["energy"]  : (B,)    FloatTensor, no NaN
        output["dipole"]  : (B,)    FloatTensor, no NaN
        output["charges"] : (N,)    FloatTensor, no NaN
    """
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = HDNNPModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert out["energy"].shape == (2,), f"Expected (2,), got {out['energy'].shape}"
    assert out["dipole"].shape == (2,), f"Expected (2,), got {out['dipole'].shape}"
    assert out["charges"].shape == (5,), f"Expected (5,), got {out['charges'].shape}"

    assert out["energy"].dtype == torch.float32
    assert out["dipole"].dtype == torch.float32
    assert out["charges"].dtype == torch.float32

    assert not torch.isnan(out["energy"]).any(), "NaN in energy"
    assert not torch.isnan(out["dipole"]).any(), "NaN in dipole"
    assert not torch.isnan(out["charges"]).any(), "NaN in charges"


def test_charge_neutrality() -> None:
    """Sum of per-atom charges per molecule is approximately zero.

    Verifies that the charge-neutralisation step in HDNNPModel.forward
    correctly removes the mean charge so each molecule sums to zero.
    """
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = HDNNPModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    charges = out["charges"]   # (5,)
    batch_idx = batch.batch    # (5,) — [0, 0, 0, 1, 1]

    # Molecule 0: atoms 0, 1, 2
    mol0_charge_sum = charges[batch_idx == 0].sum().item()
    # Molecule 1: atoms 3, 4
    mol1_charge_sum = charges[batch_idx == 1].sum().item()

    assert abs(mol0_charge_sum) < 1e-5, (
        f"Molecule 0 charge sum not neutral: {mol0_charge_sum:.2e}"
    )
    assert abs(mol1_charge_sum) < 1e-5, (
        f"Molecule 1 charge sum not neutral: {mol1_charge_sum:.2e}"
    )


def test_energy_extensivity() -> None:
    """Two identical isolated atoms give 2x the energy of a single atom.

    Places two carbon atoms 100 Angstroms apart (no edges within r_cutoff=5.0)
    and verifies that the predicted energy is exactly twice the energy of a
    single carbon atom.  This holds because there are no inter-atom interactions
    so each atom contributes independently.
    """
    # Single carbon atom — explicitly set empty edge_index so no edges exist.
    data_single = Data(
        z=torch.tensor([6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    batch_single = Batch.from_data_list([data_single])

    # Two carbon atoms 100 Å apart — no edges within r_cutoff=5.0.
    data_two = Data(
        z=torch.tensor([6, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    batch_two = Batch.from_data_list([data_two])

    model = HDNNPModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        energy_single = model(batch_single)["energy"].item()
        energy_two = model(batch_two)["energy"].item()

    ratio = energy_two / energy_single
    assert abs(ratio - 2.0) < 1e-5, (
        f"Extensivity violated: energy_two / energy_single = {ratio:.6f} (expected 2.0)"
    )


def test_no_inplace_ops_mps() -> None:
    """Forward pass runs without error on the best available device.

    Verifies no in-place tensor operations cause errors on MPS (or CUDA / CPU
    as a fallback).  Asserts that all outputs are free of NaN values.
    """
    device = get_device()

    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)
    batch = batch.to(device)

    model = HDNNPModel(TINY_CFG).to(device)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert not torch.isnan(out["energy"]).any(), "NaN in energy"
    assert not torch.isnan(out["dipole"]).any(), "NaN in dipole"
    assert not torch.isnan(out["charges"]).any(), "NaN in charges"


def test_gradients() -> None:
    """loss.backward() produces non-None, non-NaN gradients on all parameters.

    Verifies that the full computational graph is connected — every trainable
    parameter receives a valid gradient after a backward pass through both
    the energy and dipole predictions.
    """
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = HDNNPModel(TINY_CFG)
    model.train()

    out = model(batch)
    loss = out["energy"].sum() + out["dipole"].sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter '{name}' has None gradient"
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradient"
            )
