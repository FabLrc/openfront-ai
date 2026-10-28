"""Briques du PPO sur l'action factorisée [type, cible, case, quantité, unité].

La probabilité d'une action est le produit des têtes utilisées par son type :
  type, puis cible (spawn, attaque, alliance, rupture), quantité (attaque,
  bateau), unité (construction), case (bateau ; construction, selon l'unité).
Toutes les têtes sont masquées.
"""

from __future__ import annotations

import numpy as np
import torch

from .env import ALLIANCE, ATTACK, BOAT, BREAK, BUILD, SPAWN
from .model import masked

TARGET_TYPES = (SPAWN, ATTACK, ALLIANCE, BREAK)
AMOUNT_TYPES = (ATTACK, BOAT)
CELL_TYPES = (BOAT, BUILD)


def _is(typ: torch.Tensor, types: tuple[int, ...]) -> torch.Tensor:
    out = torch.zeros_like(typ, dtype=torch.bool)
    for t in types:
        out |= typ == t
    return out


def head_logits(out: dict[str, torch.Tensor], x: dict[str, torch.Tensor], typ: torch.Tensor, unit: torch.Tensor):
    """Logits masqués de chaque tête, conditionnés par le type (et l'unité pour la case)."""
    b = typ.shape[0]
    idx = torch.arange(b, device=typ.device)
    cell_mask = torch.where(
        (typ == BUILD)[:, None], x["mask_unit_cell"][idx, unit.clamp(0, 5)], x["mask_cell"][idx, typ]
    )
    return {
        "type": masked(out["type"], x["mask_type"]),
        "target": masked(out["target"][idx, typ], x["mask_target"][idx, typ]),
        "amount": masked(out["amount"][idx, typ], x["mask_amount"]),
        "unit": masked(out["unit"], x["mask_unit"]),
        "cell": masked(out["cell"][idx, typ], cell_mask),
    }


def _uses(typ: torch.Tensor) -> dict[str, torch.Tensor]:
    """Têtes utilisées par chaque action (1.0 / 0.0)."""
    return {
        "type": torch.ones_like(typ, dtype=torch.float32),
        "target": _is(typ, TARGET_TYPES).float(),
        "amount": _is(typ, AMOUNT_TYPES).float(),
        "unit": (typ == BUILD).float(),
        "cell": _is(typ, CELL_TYPES).float(),
    }


@torch.no_grad()
def sample(out, x, generator: torch.Generator | None = None, greedy: bool = False):
    """Actions [B, 5] et log-probabilités [B]."""

    def draw(lg):
        if greedy:
            return lg.argmax(-1)
        return torch.multinomial(torch.softmax(lg, -1), 1, generator=generator).squeeze(-1)

    typ = draw(masked(out["type"], x["mask_type"]))
    unit = draw(masked(out["unit"], x["mask_unit"]))
    lg = head_logits(out, x, typ, unit)
    target, amount, cell = draw(lg["target"]), draw(lg["amount"]), draw(lg["cell"])
    actions = torch.stack([typ, target, cell, amount, unit], 1)
    logp, _ = log_prob_entropy(out, x, actions)
    return actions, logp


def log_prob_entropy(out, x, actions: torch.Tensor):
    """Log-probabilité de l'action et entropie des têtes utilisées par son type."""
    typ, target, cell, amount, unit = actions.long().unbind(1)
    lg = head_logits(out, x, typ, unit)
    uses = _uses(typ)
    labels = {"type": typ, "target": target, "amount": amount, "unit": unit, "cell": cell}
    logp = torch.zeros(typ.shape[0], device=typ.device)
    ent = torch.zeros_like(logp)
    for h, logits in lg.items():
        lsm = torch.log_softmax(logits, -1)
        logp = logp + uses[h] * lsm.gather(1, labels[h][:, None]).squeeze(1)
        p = lsm.exp()
        ent = ent + uses[h] * -(p * lsm.clamp(min=-30)).sum(-1)
    return logp, ent


def kl_to_reference(out, ref_out, x, actions: torch.Tensor) -> torch.Tensor:
    """KL(π || π_ref) : exacte sur le type, et sur les têtes utilisées par l'action jouée."""
    typ, _, _, _, unit = actions.long().unbind(1)
    lg = head_logits(out, x, typ, unit)
    lr = head_logits(ref_out, x, typ, unit)
    uses = _uses(typ)
    kl = torch.zeros(typ.shape[0], device=typ.device)
    for h in lg:
        lp = torch.log_softmax(lg[h], -1)
        lq = torch.log_softmax(lr[h], -1)
        kl = kl + uses[h] * (lp.exp() * (lp - lq).clamp(-30, 30)).sum(-1)
    return kl


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: np.ndarray, gamma: float, lam: float):
    """Avantages et retours (GAE). dones[t] = 1 si l'épisode s'arrête après le pas t."""
    t_max = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_value)
    for t in reversed(range(t_max)):
        next_value = last_value if t == t_max - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values
