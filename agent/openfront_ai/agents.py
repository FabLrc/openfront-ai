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
