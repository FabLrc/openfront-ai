"""Apprentissage par imitation des Nations (behavior cloning) sur data/bc/.

    agent/.venv/bin/python agent/train_bc.py --series bc1 --epochs 4
    agent/.venv/bin/python agent/train_bc.py --series bc1 --resume      # reprend après une interruption

Pertes : entropie croisée masquée par tête (type, cible, case, quantité,
unité), chaque tête n'étant comptée que sur les échantillons où elle sert.
Les types sont repondérés en 1/√fréquence pour que « rien » ne domine pas.
Un shard sur dix sert à la validation. Les pertes et précisions vont dans
runs/<série>/train.jsonl (tableau de bord), les checkpoints dans runs/<série>/ckpt/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from openfront_ai.env import ALLIANCE, ATTACK, BOAT, BREAK, BUILD
from openfront_ai.model import PolicyNet, best_device, load_checkpoint, masked, obs_to_torch, save_checkpoint
from openfront_ai.vec import ROOT, RUNS_DIR, log_train

DATA_DIR = ROOT / "data" / "bc"
TARGET_TYPES = (ATTACK, ALLIANCE, BREAK)
AMOUNT_TYPES = (ATTACK, BOAT)
HEADS = ("type", "target", "cell", "amount", "unit")


def shards() -> tuple[list[Path], list[Path]]:
    files = sorted(DATA_DIR.glob("*.npz"))
    val = files[9::10] or files[-1:]
    train = [f for f in files if f not in val]
    return train, val


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def type_weights(files: list[Path]) -> np.ndarray:
    """Poids des types en 1/√fréquence (fréquences réelles, « rien » compris), moyenne 1, plafonnés à 10."""
    freq = np.zeros(7)
    for f in files:
        with np.load(f) as z:
            np.add.at(freq, z["action"][:, 0], z["weight"])
    freq = freq / freq.sum()
    w = np.where(freq > 0, 1 / np.sqrt(np.maximum(freq, 1e-12)), 0.0)
    w = w / (w * freq).sum()
    # Plafond : un type très rare (rompre une alliance) ne doit pas dominer.
    return np.minimum(w, 10.0)


def losses(model: PolicyNet, batch: dict[str, torch.Tensor], tw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pertes et précisions par tête (sommes et effectifs, pour moyenner ensuite)."""
    out = model(batch)
    a = batch["action"].long()
    typ = a[:, 0]
    n = typ.shape[0]
    idx = torch.arange(n, device=typ.device)
    res: dict[str, torch.Tensor] = {}

    def head(name: str, logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor, sel: torch.Tensor, weight=None):
        if sel.sum() == 0:
            return
        lg = masked(logits[sel], mask[sel])
        ce = F.cross_entropy(lg, labels[sel], reduction="none")
        w = weight[sel] if weight is not None else torch.ones_like(ce)
        res[f"loss_{name}"] = (ce * w).sum()
        res[f"w_{name}"] = w.sum()
        res[f"acc_{name}"] = (lg.argmax(1) == labels[sel]).float().sum()
        res[f"n_{name}"] = sel.sum().float()

    all_ = torch.ones_like(typ, dtype=torch.bool)
    head("type", out["type"], batch["mask_type"], typ, all_, weight=batch["weight"].float() * tw[typ])
    is_target = (typ == ATTACK) | (typ == ALLIANCE) | (typ == BREAK)
    head("target", out["target"][idx, typ], batch["mask_target"][idx, typ], a[:, 1], is_target)
    is_amount = (typ == ATTACK) | (typ == BOAT)
    head("amount", out["amount"][idx, typ], batch["mask_amount"], a[:, 3], is_amount)
    # Case : bateau → masque des bateaux ; construction → masque de l'unité construite.
    cell_mask = torch.where(
        (typ == BUILD)[:, None], batch["mask_unit_cell"][idx, a[:, 4].clamp(0, 5)], batch["mask_cell"][idx, typ]
    )
    head("cell", out["cell"][idx, typ], cell_mask, a[:, 2], (typ == BOAT) | (typ == BUILD))
    head("unit", out["unit"], batch["mask_unit"], a[:, 4], typ == BUILD)
    return res


def batches(data: dict[str, np.ndarray], size: int, rng: np.random.Generator | None):
    n = len(data["action"])
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        sel = order[i : i + size]
        yield {k: v[sel] for k, v in data.items()}


def to_device(b: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    t = obs_to_torch(b, device)
    t["action"] = torch.as_tensor(b["action"], device=device)
    t["weight"] = torch.as_tensor(b["weight"], device=device)
    return t


def summarize(acc: dict[str, float], prefix: str = "") -> dict[str, float]:
    out = {}
    for h in HEADS:
        if acc.get(f"n_{h}", 0) > 0:
            out[f"{prefix}loss_{h}"] = acc[f"loss_{h}"] / max(acc[f"w_{h}"], 1e-9)
            out[f"{prefix}acc_{h}"] = acc[f"acc_{h}"] / acc[f"n_{h}"]
    out[f"{prefix}loss"] = sum(v for k, v in out.items() if k.startswith(f"{prefix}loss_"))
    return out


@torch.no_grad()
def evaluate(model, files, tw, device, batch_size) -> dict[str, float]:
    model.eval()
    acc: dict[str, float] = {}
    for f in files:
        for b in batches(load(f), batch_size, None):
            for k, v in losses(model, to_device(b, device), tw).items():
                acc[k] = acc.get(k, 0.0) + float(v)
    model.train()
    return summarize(acc, "val_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minutes", type=float, default=0, help="arrêt propre (checkpoint) après cette durée ; 0 : jusqu'au bout")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = best_device()
    ckpt_dir = RUNS_DIR / args.series / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_files, val_files = shards()
    if not train_files:
        raise SystemExit("aucun shard dans data/bc/ : lancer d'abord agent/collect_bc.py")
    tw_np = type_weights(train_files)
    tw = torch.as_tensor(tw_np, dtype=torch.float32, device=device)

    state = {"epoch": 0, "shard": 0, "step": 0, "best": float("inf"), "seconds": 0.0}
    if args.resume and (ckpt_dir / "last.pt").exists():
        model, ckpt = load_checkpoint(ckpt_dir / "last.pt", device)
        state.update(ckpt["progress"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        opt.load_state_dict(ckpt["optimizer"])
        print(f"Reprise : époque {state['epoch']}, shard {state['shard']}, pas {state['step']}")
    else:
        model = PolicyNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    params = sum(p.numel() for p in model.parameters())
    print(f"{params / 1e6:.2f} M paramètres, {device}, {len(train_files)} shards d'entraînement, {len(val_files)} de validation")
    print("poids des types :", dict(zip(("noop", "spawn", "attack", "boat", "build", "alliance", "break"), np.round(tw_np, 2))))

    def checkpoint(name: str) -> None:
        save_checkpoint(ckpt_dir / name, model, optimizer=opt.state_dict(), progress=state, type_weights=tw_np.tolist())

    start = time.monotonic()
    t0_total = state["seconds"]
    acc: dict[str, float] = {}
    stop = False
    while state["epoch"] < args.epochs and not stop:
        order = np.random.default_rng(state["epoch"]).permutation(len(train_files))
        while state["shard"] < len(order):
            data = load(train_files[order[state["shard"]]])
            rng = np.random.default_rng(state["step"])
            for b in batches(data, args.batch, rng):
                res = losses(model, to_device(b, device), tw)
                loss = sum(res[f"loss_{h}"] / res[f"w_{h}"].clamp(min=1e-9) for h in HEADS if f"loss_{h}" in res)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                state["step"] += 1
                for k, v in res.items():
                    acc[k] = acc.get(k, 0.0) + float(v.detach())
                if state["step"] % args.log_every == 0:
                    log_train(args.series, step=state["step"], epoch=state["epoch"], **summarize(acc))
                    acc = {}
            state["shard"] += 1
            state["seconds"] = t0_total + time.monotonic() - start
            checkpoint("last.pt")
            if args.minutes and time.monotonic() - start > args.minutes * 60:
                stop = True
                break
        if stop:
            break
        val = evaluate(model, val_files, tw, device, args.batch)
        log_train(args.series, step=state["step"], epoch=state["epoch"], **val)
        print(f"époque {state['epoch']} : " + ", ".join(f"{k} {v:.3f}" for k, v in val.items()), flush=True)
        if val["val_loss"] < state["best"]:
            state["best"] = val["val_loss"]
            checkpoint("best.pt")
        state["epoch"] += 1
        state["shard"] = 0
        checkpoint("last.pt")
    state["seconds"] = t0_total + time.monotonic() - start
    checkpoint("last.pt")
    (RUNS_DIR / args.series / "bc.json").write_text(json.dumps({**state, "params": params, "device": str(device)}, indent=1))
    print(f"{'Arrêt (reprendre avec --resume)' if stop else 'Terminé'} : pas {state['step']}, {state['seconds'] / 60:.1f} min d'entraînement")


if __name__ == "__main__":
    main()
