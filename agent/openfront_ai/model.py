"""Réseau de la politique (PyTorch).

Entrées (une observation de siège, voir env/src/seat.ts) :
  spatial  uint8 [13, H, W]  → CNN ; la carte de caractéristiques finale est à
                              la résolution de la grille d'actions (H/4, W/4)
  players  float [K+1, F]    → petit transformer (une ligne par joueur)
  scalars  float [S]         → MLP

Têtes factorisées, toutes masquées :
  type    [T]                      quel type d'action
  target  [T, K+1]                 pointeur sur la table des joueurs (un par type)
  cell    [T, A]                   case de la grille d'actions (bateau, construction)
  amount  [T, 4]                   part des troupes (attaque, bateau)
  unit    [U]                      structure à construire
  value   []                       estimation de la valeur (pour PPO)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .env import ACTION_TYPES, AMOUNTS, BUILD_UNITS, N_CHANNELS, N_PLAYER_FEATURES, N_SCALARS

NEG = -1e9
T, NA, U = len(ACTION_TYPES), len(AMOUNTS), len(BUILD_UNITS)


class PolicyNet(nn.Module):
    def __init__(self, k: int = 16, width: int = 128, trunk: int = 512):
        super().__init__()
        self.config = {"k": k, "width": width, "trunk": trunk}
        c = width
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, c // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
        )
        self.players_in = nn.Linear(N_PLAYER_FEATURES, c)
        layer = nn.TransformerEncoderLayer(c, nhead=4, dim_feedforward=2 * c, dropout=0.0, batch_first=True, norm_first=True)
        self.players = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.scalars = nn.Sequential(nn.Linear(N_SCALARS, c), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(4 * c, trunk), nn.GELU(), nn.Linear(trunk, trunk), nn.GELU())
        self.type_head = nn.Linear(trunk, T)
        self.target_query = nn.Linear(trunk, T * c)
        self.cell_film = nn.Linear(trunk, c)
        self.cell_head = nn.Conv2d(c, T, 1)
        self.amount_head = nn.Linear(trunk, T * NA)
        self.unit_head = nn.Linear(trunk, U)
        self.value_head = nn.Linear(trunk, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = obs["spatial"].float() / 255.0
        fmap = self.cnn(x)  # [B, c, H/4, W/4]
        b, c = fmap.shape[:2]
        pooled = torch.cat([fmap.mean((2, 3)), fmap.amax((2, 3))], 1)
        rows = self.players(self.players_in(obs["players"].float()))  # [B, K+1, c]
        present = obs["players"][..., 0:1].float()
        players_pool = (rows * present).sum(1) / present.sum(1).clamp(min=1)
        h = self.trunk(torch.cat([pooled, players_pool, self.scalars(obs["scalars"].float())], 1))

        target = torch.einsum("btc,bkc->btk", self.target_query(h).view(b, T, c), rows) / c**0.5
        cell = self.cell_head(fmap + self.cell_film(h)[:, :, None, None]).flatten(2)  # [B, T, A]
        return {
            "type": self.type_head(h),
            "target": target,
            "cell": cell,
            "amount": self.amount_head(h).view(b, T, NA),
            "unit": self.unit_head(h),
            "value": self.value_head(h).squeeze(1),
        }


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(mask == 0, NEG)


def obs_to_torch(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Observation (ou lot d'observations) numpy → tenseurs, masques compris."""
    out = {f: torch.as_tensor(np.asarray(obs[f]), device=device) for f in ("spatial", "players", "scalars")}
    masks = obs.get("masks")
    if masks is not None:
        for f, v in masks.items():
            out[f"mask_{f}"] = torch.as_tensor(np.asarray(v), device=device)
    else:  # données d'imitation : masques à plat (mask_type…)
        for f in ("type", "target", "cell", "amount", "unit", "unit_cell"):
            out[f"mask_{f}"] = torch.as_tensor(np.asarray(obs[f"mask_{f}"]), device=device)
    if out["spatial"].dim() == 3:
        out = {k: v.unsqueeze(0) for k, v in out.items()}
    return out


def save_checkpoint(path, model: PolicyNet, **extra: Any) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), **extra}, path)


def load_checkpoint(path, device: torch.device | str = "cpu") -> tuple[PolicyNet, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = PolicyNet(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state"])
    return model, ckpt


def best_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
