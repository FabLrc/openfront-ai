"""Agents de référence : aléatoire (dans les masques) et à règles simples."""

from __future__ import annotations

from typing import Any

import numpy as np

from .env import ATTACK, BUILD, NOOP, SPAWN

# Colonnes de la table des joueurs et des scalaires (cf. env/src/seat.ts).
P_TROOPS = 4
S_TROOPS_RATIO = 2
CITY = 0


def _pick(rng: np.random.Generator, mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask)
    return int(rng.choice(valid)) if len(valid) else -1


class RandomAgent:
    """Tire uniformément un type d'action autorisé, puis ses paramètres autorisés."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict[str, Any]) -> list[int]:
        m = obs["masks"]
        t = _pick(self.rng, m["type"])
        if t <= NOOP:
            return [NOOP, 0, 0, 0, 0]
        amount = _pick(self.rng, m["amount"])
        if t == BUILD:
            unit = _pick(self.rng, m["unit"])
            return [t, 0, _pick(self.rng, m["unit_cell"][unit]), 0, unit]
        cell = _pick(self.rng, m["cell"][t])
        target = _pick(self.rng, m["target"][t])
        return [t, max(target, 0), max(cell, 0), amount, 0]


class RuleAgent:
    """Spawn sur le premier candidat valide ; construit des villes dès que
    possible ; quand ses troupes dépassent ``threshold`` de son maximum, attaque
    les terres neutres, sinon le voisin attaquable le plus faible (s'il est
    plus faible que lui)."""

    def __init__(self, seed: int | None = None, threshold: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.threshold = threshold

    def act(self, obs: dict[str, Any]) -> list[int]:
        m = obs["masks"]
        if m["type"][SPAWN]:
            return [SPAWN, int(np.flatnonzero(m["target"][SPAWN])[0]), 0, 0, 0]
        if m["type"][BUILD] and m["unit"][CITY]:
            return [BUILD, 0, _pick(self.rng, m["unit_cell"][CITY]), 0, CITY]
        if m["type"][ATTACK] and obs["scalars"][S_TROOPS_RATIO] > self.threshold:
            targets = m["target"][ATTACK]
            if targets[0]:
                return [ATTACK, 0, 0, 1, 0]  # terres neutres, 25 %
            rows = np.flatnonzero(targets)
            weakest = int(rows[np.argmin(obs["players"][rows, P_TROOPS])])
            if obs["players"][weakest, 6] < 0:  # moins de troupes que moi
                return [ATTACK, weakest, 0, 2, 0]  # 50 %
        return [NOOP, 0, 0, 0, 0]


class PolicyAgent:
    """Politique apprise, chargée depuis un checkpoint (voir train_bc.py).

    Décode les têtes l'une après l'autre en respectant les masques : type,
    puis cible / quantité / unité / case selon le type. ``greedy`` prend
    l'action la plus probable de chaque tête, sinon on échantillonne.
    ``last_probs`` garde les probabilités des types de la dernière décision
    (affichées sur /live).
    """

    def __init__(self, checkpoint: str, greedy: bool = False, seed: int | None = None, device: str = "cpu"):
        import torch

        from .model import load_checkpoint

        self.torch = torch
        self.device = torch.device(device)
        self.model, _ = load_checkpoint(checkpoint, self.device)
        self.model.eval()
        self.greedy = greedy
        self.gen = torch.Generator(device=self.device).manual_seed(seed if seed is not None else 0)
        self.last_probs: dict[str, float] = {}

    def act(self, obs: dict[str, Any]) -> list[int]:
        batch = {k: v[None] for k, v in obs.items() if k != "masks"}
        batch["masks"] = {k: v[None] for k, v in obs["masks"].items()}
        return self.act_batch(batch)[0].tolist()

    def act_batch(self, obs: dict[str, Any]) -> np.ndarray:
        """Actions [B, 5] pour un lot d'observations (champs empilés)."""
        torch = self.torch
        from .env import ACTION_TYPES
        from .model import masked, obs_to_torch

        with torch.no_grad():
            x = obs_to_torch(obs, self.device)
            out = self.model(x)
            b = out["type"].shape[0]
            idx = torch.arange(b)

            def choose(logits, mask):
                lg = masked(logits, mask)
                if self.greedy:
                    return lg.argmax(-1)
                return torch.multinomial(torch.softmax(lg, -1), 1, generator=self.gen).squeeze(-1)

            typ = choose(out["type"], x["mask_type"])
            # Une tête sans choix valide (masque vide) tire au hasard : sa valeur
            # n'est pas utilisée pour ce type d'action.
            target = choose(out["target"][idx, typ], x["mask_target"][idx, typ])
            amount = choose(out["amount"][idx, typ], x["mask_amount"])
            unit = choose(out["unit"], x["mask_unit"])
            cell_mask = torch.where((typ == BUILD)[:, None], x["mask_unit_cell"][idx, unit], x["mask_cell"][idx, typ])
            cell = choose(out["cell"][idx, typ], cell_mask)
            probs = torch.softmax(masked(out["type"], x["mask_type"]), -1)[0]
            self.last_probs = {t: round(float(p), 3) for t, p in zip(ACTION_TYPES, probs)}
        return torch.stack([typ, target, cell, amount, unit], 1).cpu().numpy().astype(np.int64)
