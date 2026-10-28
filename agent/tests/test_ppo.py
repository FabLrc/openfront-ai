"""PPO : log-probabilités factorisées, KL vers la référence et GAE sur un petit lot factice."""

import numpy as np
import torch

from openfront_ai.env import ATTACK, BOAT, BUILD, NOOP
from openfront_ai.model import PolicyNet, obs_to_torch
from openfront_ai.ppo import gae, kl_to_reference, log_prob_entropy, sample
from test_model import fake_batch


def test_sample_et_log_prob_coherents():
    torch.manual_seed(0)
    model = PolicyNet().eval()
    obs = fake_batch(np.random.default_rng(3))
    x = obs_to_torch(obs, torch.device("cpu"))
    with torch.no_grad():
        out = model(x)
        actions, logp = sample(out, x)
        again, ent = log_prob_entropy(out, x, actions)
    assert torch.allclose(logp, again, atol=1e-5)
    assert (ent > 0).all() and torch.isfinite(ent).all()
    # Seules les têtes utilisées comptent : changer la case d'une attaque ne change rien.
    a1 = torch.tensor([[ATTACK, 1, 0, 2, 0]] * 3 + [[NOOP, 0, 0, 0, 0]] * 3)
    a2 = torch.tensor([[ATTACK, 1, 5, 2, 3]] * 3 + [[NOOP, 0, 7, 1, 2]] * 3)
    lp1, _ = log_prob_entropy(out, x, a1)
    lp2, _ = log_prob_entropy(out, x, a2)
    assert torch.allclose(lp1, lp2)


def test_kl_nulle_contre_soi_meme_et_positive_sinon():
    model = PolicyNet().eval()
    other = PolicyNet().eval()
    obs = fake_batch(np.random.default_rng(4))
    x = obs_to_torch(obs, torch.device("cpu"))
    a = torch.tensor([[BOAT, 0, 5, 1, 0], [BUILD, 0, 5, 0, 0], [ATTACK, 1, 0, 0, 0]] * 2)
    with torch.no_grad():
        out = model(x)
        assert kl_to_reference(out, out, x, a).abs().max() < 1e-5
        assert (kl_to_reference(out, other(x), x, a) > 0).all()


def test_gae():
    rewards = np.array([[1.0], [0.0], [2.0]], np.float32)
    values = np.array([[0.5], [0.5], [0.5]], np.float32)
    dones = np.array([[0.0], [1.0], [0.0]], np.float32)
    adv, ret = gae(rewards, values, dones, last_value=np.array([1.0], np.float32), gamma=0.9, lam=1.0)
    # Pas 2 : 2 + 0,9·1 − 0,5 ; pas 1 (fin d'épisode) : 0 − 0,5 ; pas 0 : 1 + 0,9·0,5 − 0,5 + 0,9·(−0,5)
    assert np.allclose(adv[:, 0], [0.5, -0.5, 2.4])
    assert np.allclose(ret, adv + values)
