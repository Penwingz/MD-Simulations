"""
Tests for PaiNNModel.

Uses mock batches (5 atoms, 2 molecules) — no real QM9 data.
Run: python -m pytest tests/test_painn_model.py -v
"""

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch

from src.model import PaiNNModel
from src.utils import RadiusGraphTransform, get_device


TINY_CFG = OmegaConf.create({
    "model": {
        "type": "painn",
        "n_features": 32,
        "n_interactions": 2,
        "n_rbf": 20,
        "r_cutoff": 5.0,
        "max_atomic_num": 10,
    }
})


def make_batch(z_list: list, pos_list: list, r_cutoff: float = 5.0) -> Batch:
    """Build a PyG Batch from lists of z-arrays and position arrays."""
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


def test_forward_pass_shapes() -> None:
    """Forward pass returns correct output shapes and dtypes, no NaN."""
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = PaiNNModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert out["energy"].shape == (2,), f"Expected (2,), got {out['energy'].shape}"
    assert out["dipole"].shape == (2,), f"Expected (2,), got {out['dipole'].shape}"
    assert out["charges"].shape == (5,), f"Expected (5,), got {out['charges'].shape}"

    assert out["energy"].dtype == torch.float32
    assert out["dipole"].dtype == torch.float32

    assert not torch.isnan(out["energy"]).any(), "NaN in energy"
    assert not torch.isnan(out["dipole"]).any(), "NaN in dipole"
    assert (out["dipole"] >= 0).all(), "Dipole norm must be non-negative"


def test_equivariance_dipole() -> None:
    """Dipole norm is invariant under 3D rotation of all atomic positions.

    The PaiNN vector path accumulates equivariant features that rotate with
    the molecule.  The dipole is the L2 norm of the summed vectors, so
    rotating positions must not change the predicted dipole norm.
    """
    z_list = [[6, 1, 1, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ]
    batch_orig = make_batch(z_list, pos_list)

    # Build a random orthogonal rotation matrix via QR decomposition
    torch.manual_seed(99)
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(R) < 0:
        R = R * torch.tensor([-1.0, 1.0, 1.0])  # ensure proper rotation

    # Rotate all atom positions
    pos_rotated = batch_orig.pos @ R.t()
    batch_rot = Batch.from_data_list([
        RadiusGraphTransform(r=5.0)(
            Data(
                z=batch_orig.z.clone(),
                pos=pos_rotated.clone(),
            )
        )
    ])

    model = PaiNNModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        dipole_orig = model(batch_orig)["dipole"]
        dipole_rot = model(batch_rot)["dipole"]

    assert torch.allclose(dipole_orig, dipole_rot, atol=1e-4), (
        f"Dipole not rotationally invariant: orig={dipole_orig.item():.6f}, "
        f"rotated={dipole_rot.item():.6f}, diff={abs(dipole_orig - dipole_rot).item():.2e}"
    )


def test_vector_channel_nonzero_after_forward() -> None:
    """Vector features v are zero at init but become non-zero after one forward pass.

    Confirms that r_hat direction vectors are actually reaching and populating
    the equivariant vector channel via the message blocks.
    """
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = PaiNNModel(TINY_CFG)
    model.eval()

    # Intercept v after the first message+update block
    v_after_first_block: list = []

    original_upd_forward = model.update_blocks[0].forward

    def hook_upd_forward(s, v):
        result = original_upd_forward(s, v)
        v_after_first_block.append(result[1].detach().clone())
        return result

    model.update_blocks[0].forward = hook_upd_forward  # type: ignore[method-assign]

    with torch.no_grad():
        model(batch)

    assert len(v_after_first_block) == 1
    v = v_after_first_block[0]
    assert v.shape[2] == 3, "Vector channel should have spatial dim=3"
    assert v.abs().max().item() > 1e-6, (
        "Vector features still zero after one forward pass — "
        "r_hat is not reaching the message block"
    )


def test_energy_extensivity() -> None:
    """Two isolated identical atoms give 2x the energy of one atom.

    With no edges (atoms 100 Å apart, below r_cutoff), each atom contributes
    independently so extensivity must hold exactly.
    """
    data_single = Data(
        z=torch.tensor([6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )
    data_two = Data(
        z=torch.tensor([6, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=torch.float32),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
    )

    model = PaiNNModel(TINY_CFG)
    model.eval()

    with torch.no_grad():
        e_single = model(Batch.from_data_list([data_single]))["energy"].item()
        e_two = model(Batch.from_data_list([data_two]))["energy"].item()

    ratio = e_two / e_single
    assert abs(ratio - 2.0) < 1e-5, (
        f"Extensivity violated: e_two / e_single = {ratio:.6f} (expected 2.0)"
    )


def test_no_inplace_ops_mps() -> None:
    """Forward pass completes without error on the best available device (MPS/CUDA/CPU)."""
    device = get_device()

    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list).to(device)

    model = PaiNNModel(TINY_CFG).to(device)
    model.eval()

    with torch.no_grad():
        out = model(batch)

    assert not torch.isnan(out["energy"]).any(), "NaN in energy"
    assert not torch.isnan(out["dipole"]).any(), "NaN in dipole"


def test_gradients() -> None:
    """loss.backward() produces non-None, non-NaN gradients on all parameters.

    Verifies the full graph is connected through both the energy (scalar) and
    dipole (equivariant vector) readout paths.
    """
    z_list = [[6, 1, 1], [8, 1]]
    pos_list = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    ]
    batch = make_batch(z_list, pos_list)

    model = PaiNNModel(TINY_CFG)
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
