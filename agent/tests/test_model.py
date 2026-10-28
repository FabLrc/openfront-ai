"""Réseau, pertes d'imitation et PolicyAgent sur des données synthétiques (sans serveur)."""

import numpy as np
import torch

from openfront_ai.agents import PolicyAgent
from openfront_ai.env import ATTACK, BOAT, BUILD, NOOP, N_CHANNELS, N_PLAYER_FEATURES, N_SCALARS
from openfront_ai.model import PolicyNet, load_checkpoint, save_checkpoint

B, K1, A = 6, 17, 512


def fake_batch(rng):
    masks = {
        "type": np.zeros((B, 7), np.uint8),
        "target": rng.integers(0, 2, (B, 7, K1), dtype=np.uint8),
        "cell": rng.integers(0, 2, (B, 7, A), dtype=np.uint8),
        "amount": np.ones((B, 4), np.uint8),
        "unit": rng.integers(0, 2, (B, 6), dtype=np.uint8),
        "unit_cell": rng.integers(0, 2, (B, 6, A), dtype=np.uint8),
    }
    masks["type"][:, [NOOP, ATTACK, BOAT, BUILD]] = 1
    masks["unit"][:, 0] = 1
    masks["target"][:, :, 1] = 1
    masks["cell"][:, :, 5] = 1
    masks["unit_cell"][:, :, 5] = 1
    return {
        "spatial": rng.integers(0, 256, (B, N_CHANNELS, 64, 128), dtype=np.uint8),
        "players": rng.random((B, K1, N_PLAYER_FEATURES), dtype=np.float32),
        "scalars": rng.random((B, N_SCALARS), dtype=np.float32),
        "masks": masks,
    }


def test_formes_et_taille():
    model = PolicyNet()
    obs = fake_batch(np.random.default_rng(0))
    from openfront_ai.model import obs_to_torch

    out = model(obs_to_torch(obs, torch.device("cpu")))
    assert out["type"].shape == (B, 7)
    assert out["target"].shape == (B, 7, K1)
    assert out["cell"].shape == (B, 7, A)
    assert out["amount"].shape == (B, 7, 4)
    assert out["unit"].shape == (B, 6)
    assert out["value"].shape == (B,)
    n = sum(p.numel() for p in model.parameters())
    assert 1e6 <= n <= 5e6


def test_pertes_imitation():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train_bc import losses, to_device

    rng = np.random.default_rng(1)
    obs = fake_batch(rng)
    flat = {k: v for k, v in obs.items() if k != "masks"}
    flat.update({f"mask_{k}": v for k, v in obs["masks"].items()})
    flat["action"] = np.array([[ATTACK, 1, 0, 2, 0], [BOAT, 0, 5, 1, 0], [BUILD, 0, 5, 0, 0], [NOOP, 0, 0, 0, 0]] * 2, np.int16)[:B]
    flat["weight"] = np.ones(B, np.float32)
    res = losses(PolicyNet(), to_device(flat, torch.device("cpu")), torch.ones(7))
    for head in ("type", "target", "cell", "amount", "unit"):
        assert torch.isfinite(res[f"loss_{head}"]), head
        assert res[f"n_{head}"] > 0


def test_policy_agent_respecte_les_masques(tmp_path):
    save_checkpoint(tmp_path / "m.pt", PolicyNet())
    model, _ = load_checkpoint(tmp_path / "m.pt")
    assert isinstance(model, PolicyNet)
    agent = PolicyAgent(str(tmp_path / "m.pt"), seed=0)
    rng = np.random.default_rng(2)
    for _ in range(5):
        obs = fake_batch(rng)
        actions = agent.act_batch(obs)
        m = obs["masks"]
        for b, (t, target, cell, amount, unit) in enumerate(actions):
            assert m["type"][b, t]
            if t == ATTACK:
                assert m["target"][b, t, target] and m["amount"][b, amount]
            if t == BOAT:
                assert m["cell"][b, t, cell]
            if t == BUILD:
                assert m["unit"][b, unit] and m["unit_cell"][b, unit, cell]
    assert abs(sum(agent.last_probs.values()) - 1) < 1e-3
