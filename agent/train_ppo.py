"""PPO à partir de la politique imitée, contre des Nations (1 siège + 7 Nations).

    caffeinate -i agent/.venv/bin/python agent/train_ppo.py --series ppo1 --minutes 30
    (relancer la même commande reprend au dernier checkpoint ; voir scripts/train-ppo.sh)

- Initialisation depuis --init (par défaut runs/bc1/ckpt/best.pt), copie gelée
  de cette politique comme référence : pénalité KL(π‖π_imitée) dont le
  coefficient décroît jusqu'à 0, pour ne pas tout oublier au début.
- Premières mises à jour : tête de valeur seule (elle n'a pas été entraînée).
- GAE, objectif PPO tronqué, bonus d'entropie, avantages normalisés.
- Curriculum : Medium → Hard → Impossible quand le taux de victoire glissant
  sur la difficulté courante dépasse --promote.
- Récompense de l'environnement : victoire, rang final, et un shaping sur la
  part de territoire dont le coefficient décroît.
- Métriques dans runs/<série>/train.jsonl (tableau de bord), épisodes dans
  metrics.jsonl, une partie sur N enregistrée, checkpoints dans runs/<série>/ckpt/.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque

import numpy as np
import torch

from openfront_ai.model import best_device, load_checkpoint, obs_to_torch, save_checkpoint
from openfront_ai.ppo import gae, kl_to_reference, log_prob_entropy, sample
from openfront_ai.vec import RUNS_DIR, ServerPool, VecOpenFrontEnv, log_train, performance_cores

DIFFICULTIES = ["Medium", "Hard", "Impossible"]


def schedule(start: float, update: int, horizon: int) -> float:
    """Décroissance linéaire de start à 0 en `horizon` mises à jour."""
    return start * max(0.0, 1.0 - update / max(1, horizon))


class Rollout:
    """Tampon [T, B] d'observations, d'actions et de résultats."""

    def __init__(self, t: int, b: int, obs: dict):
        self.obs = {k: np.zeros((t, *v.shape), v.dtype) for k, v in obs.items() if k != "masks"}
        self.masks = {k: np.zeros((t, *v.shape), v.dtype) for k, v in obs["masks"].items()}
        self.actions = np.zeros((t, b, 5), np.int64)
        self.logp = np.zeros((t, b), np.float32)
        self.values = np.zeros((t, b), np.float32)
        self.rewards = np.zeros((t, b), np.float32)
        self.dones = np.zeros((t, b), np.float32)
        self.valid = np.zeros((t, b), bool)

    def store_obs(self, t: int, obs: dict) -> None:
        for k in self.obs:
            self.obs[k][t] = obs[k]
        for k in self.masks:
            self.masks[k][t] = obs["masks"][k]

    def flat(self, sel: np.ndarray) -> dict:
        """Échantillons valides (sel : indices dans [T*B])."""
        out = {k: v.reshape(-1, *v.shape[2:])[sel] for k, v in self.obs.items()}
        for k, v in self.masks.items():
            out[f"mask_{k}"] = v.reshape(-1, *v.shape[2:])[sel]
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", required=True)
    p.add_argument("--init", default=str(RUNS_DIR / "bc1" / "ckpt" / "best.pt"))
    p.add_argument("--minutes", type=float, default=30, help="durée de ce segment (checkpoint à chaque mise à jour)")
    p.add_argument("--fresh", action="store_true", help="ignorer un checkpoint existant de la série")
    p.add_argument("--procs", type=int, default=performance_cores())
    p.add_argument("--games-per-proc", type=int, default=4)
    p.add_argument("--steps", type=int, default=128, help="pas par rollout")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--kl", type=float, default=0.5, help="coefficient initial de la pénalité KL vers la politique imitée")
    p.add_argument("--kl-decay", type=int, default=300, help="mises à jour pour l'annuler")
    p.add_argument("--shaping", type=float, default=0.5, help="coefficient initial du shaping de territoire")
    p.add_argument("--shaping-decay", type=int, default=600)
    p.add_argument("--value-warmup", type=int, default=5, help="mises à jour de la tête de valeur seule")
    p.add_argument("--promote", type=float, default=0.5, help="taux de victoire glissant pour passer à la difficulté suivante")
    p.add_argument("--window", type=int, default=50, help="épisodes de la fenêtre glissante")
    p.add_argument("--max-ticks", type=int, default=30000)
    p.add_argument("--record-every", type=int, default=50, help="enregistre une partie sur N (0 : aucune)")
    p.add_argument("--snapshot-every", type=int, default=25, help="copie du checkpoint toutes les N mises à jour")
    p.add_argument("--base-port", type=int, default=6400)
    p.add_argument("--freeze-convs", type=int, default=None,
                   help="premières convolutions gelées ; défaut : 2 sur le GPU du Mac (rétropropagation lente), 0 ailleurs")
    args = p.parse_args()

    device = best_device()
    ckpt_dir = RUNS_DIR / args.series / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume = (ckpt_dir / "last.pt").exists() and not args.fresh
    model, ckpt = load_checkpoint(ckpt_dir / "last.pt" if resume else args.init, device)
    reference, _ = load_checkpoint(args.init, device)
    reference.eval()
    for q in reference.parameters():
        q.requires_grad_(False)
    # Les premières convolutions (apprises par imitation) restent gelées : sur
    # le GPU du Mac, leur rétropropagation coûtait ~2/3 de la mise à jour.
    if args.freeze_convs is None:
        args.freeze_convs = 2 if device.type == "mps" else 0
    convs = [m for m in model.cnn if isinstance(m, torch.nn.Conv2d)]
    for conv in convs[: args.freeze_convs]:
        conv.requires_grad_(False)
    trainable = [q for q in model.parameters() if q.requires_grad]
    value_params = {id(q) for q in model.value_head.parameters()}
    opt = torch.optim.Adam(trainable, lr=args.lr, eps=1e-5)
    state = {"update": 0, "decisions": 0, "difficulty": 0, "seconds": 0.0, "episodes": 0}
    window: deque = deque(maxlen=args.window)
    if resume:
        try:
            opt.load_state_dict(ckpt["optimizer"])
        except ValueError:
            # Autres couches gelées qu'à la sauvegarde (par ex. reprise sur CUDA
            # d'un run du Mac) : on garde les poids, l'optimiseur repart de zéro.
            print("Optimiseur réinitialisé : les couches entraînées diffèrent de celles du checkpoint.")
        state.update(ckpt["progress"])
        window.extend(ckpt.get("window", []))
        print(f"Reprise : mise à jour {state['update']}, {state['decisions']} décisions, difficulté {DIFFICULTIES[state['difficulty']]}")
    else:
        print(f"Départ depuis {args.init}")

    def checkpoint(name: str) -> None:
        save_checkpoint(ckpt_dir / name, model, optimizer=opt.state_dict(), progress=state, window=list(window), args=vars(args))

    start = time.monotonic()
    seconds0 = state["seconds"]
    with ServerPool(args.procs, base_port=args.base_port, dashboard=True, label=args.series) as pool:
        env = VecOpenFrontEnv(
            pool.ports,
            games_per_server=args.games_per_proc,
            agents=1,
            nations=7,
            difficulty=DIFFICULTIES[state["difficulty"]],
            max_ticks=args.max_ticks,
            seed=f"{args.series}-{state['update']}",
            series=args.series,
            record=args.series if args.record_every > 0 else None,
            record_every=max(1, args.record_every),
            shaping=schedule(args.shaping, state["update"], args.shaping_decay),
        )
        obs, _ = env.reset()
        B, T = env.num_envs, args.steps
        buf = Rollout(T, B, obs)
        ep_return = np.zeros(B, np.float32)
        returns: deque = deque(maxlen=args.window)
        try:
            while time.monotonic() - start < args.minutes * 60:
                u = state["update"]
                kl_coef = schedule(args.kl, u, args.kl_decay)
                env.config["shaping"] = schedule(args.shaping, u, args.shaping_decay)
                env.config["difficulty"] = DIFFICULTIES[state["difficulty"]]
                t0 = time.monotonic()

                # ── Collecte ──
                model.eval()
                for t in range(T):
                    active = env.active.copy()
                    buf.store_obs(t, obs)
                    with torch.no_grad():
                        x = obs_to_torch(obs, device)
                        out = model(x)
                        actions, logp = sample(out, x)
                    a = actions.cpu().numpy()
                    buf.actions[t], buf.logp[t] = a, logp.cpu().numpy()
                    buf.values[t] = np.where(active, out["value"].cpu().numpy(), 0.0)
                    obs, rew, term, trunc, infos = env.step(a)
                    done = term | trunc
                    buf.valid[t] = active
                    buf.rewards[t] = np.where(active, rew, 0.0)
                    # Un siège inactif (partie en réinitialisation) coupe aussi la chaîne GAE.
                    buf.dones[t] = np.where(active, done, 1.0)
                    state["decisions"] += int(active.sum())
                    ep_return += buf.rewards[t]
                    for b in np.flatnonzero(done & active):
                        info = infos[b]
                        window.append({"won": bool(info.get("won")), "rank": info.get("rank"), "difficulty": state["difficulty"]})
                        returns.append(float(ep_return[b]))
                        ep_return[b] = 0.0
                        state["episodes"] += 1
                with torch.no_grad():
                    last_value = model(obs_to_torch(obs, device))["value"].cpu().numpy() * env.active
                adv, ret = gae(buf.rewards, buf.values, buf.dones, last_value, args.gamma, args.lam)
                collect_s = time.monotonic() - t0

                # ── Mise à jour ──
                model.train()
                sel = np.flatnonzero(buf.valid.reshape(-1))
                data = buf.flat(sel)
                actions_all = buf.actions.reshape(-1, 5)[sel]
                old_logp = buf.logp.reshape(-1)[sel]
                adv_all = adv.reshape(-1)[sel]
                ret_all = ret.reshape(-1)[sel]
                adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)
                # Sorties de la politique imitée, une fois par rollout (pas à chaque époque).
                ref_store: dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    for i in range(0, len(sel), args.minibatch):
                        part = reference(obs_to_torch({k: v[i : i + args.minibatch] for k, v in data.items()}, device))
                        for k, v in part.items():
                            ref_store.setdefault(k, []).append(v)
                ref_store = {k: torch.cat(v) for k, v in ref_store.items()}
                warmup = u < args.value_warmup
                stats: dict[str, list[float]] = {k: [] for k in ("loss_pi", "loss_v", "entropy", "kl_bc", "approx_kl", "clipfrac")}
                for _ in range(args.epochs):
                    order = np.random.permutation(len(sel))
                    for i in range(0, len(sel), args.minibatch):
                        mb = order[i : i + args.minibatch]
                        x = obs_to_torch({k: v[mb] for k, v in data.items()}, device)
                        act = torch.as_tensor(actions_all[mb], device=device)
                        out = model(x)
                        logp, ent = log_prob_entropy(out, x, act)
                        ratio = (logp - torch.as_tensor(old_logp[mb], device=device)).exp()
                        a_t = torch.as_tensor(adv_all[mb], device=device)
                        pg = torch.max(-a_t * ratio, -a_t * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                        v_loss = 0.5 * (out["value"] - torch.as_tensor(ret_all[mb], device=device)).pow(2).mean()
                        mb_t = torch.as_tensor(mb, device=device)
                        kl_bc = kl_to_reference(out, {k: v[mb_t] for k, v in ref_store.items()}, x, act).mean()
                        if warmup:
                            loss = args.vf * v_loss
                        else:
                            loss = pg + args.vf * v_loss - args.ent * ent.mean() + kl_coef * kl_bc
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        if warmup:
                            # Échauffement : seule la tête de valeur bouge (le tronc est partagé
                            # avec la politique, qui doit rester celle de l'imitation).
                            for q in trainable:
                                if id(q) not in value_params:
                                    q.grad = None
                        torch.nn.utils.clip_grad_norm_(trainable, 0.5)
                        opt.step()
                        with torch.no_grad():
                            stats["loss_pi"].append(float(pg))
                            stats["loss_v"].append(float(v_loss))
                            stats["entropy"].append(float(ent.mean()))
                            stats["kl_bc"].append(float(kl_bc))
                            stats["approx_kl"].append(float((ratio - 1 - ratio.log()).mean()))
                            stats["clipfrac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
                update_s = time.monotonic() - t0 - collect_s

                # ── Curriculum, journal, checkpoint ──
                state["update"] = u + 1
                state["seconds"] = seconds0 + time.monotonic() - start
                current = [w for w in window if w["difficulty"] == state["difficulty"]]
                win_rate = float(np.mean([w["won"] for w in current])) if current else 0.0
                promoted = False
                if len(current) >= args.window and win_rate >= args.promote and state["difficulty"] < len(DIFFICULTIES) - 1:
                    state["difficulty"] += 1
                    promoted = True
                var = np.var(ret_all)
                explained = float(1 - np.var(ret_all - buf.values.reshape(-1)[sel]) / var) if var > 0 else 0.0
                log_train(
                    args.series,
                    step=state["decisions"],
                    update=state["update"],
                    **{k: float(np.mean(v)) for k, v in stats.items()},
                    episode_return=float(np.mean(returns)) if returns else 0.0,
                    win_rate=win_rate,
                    rank=float(np.mean([w["rank"] for w in current if w["rank"] is not None])) if current else 0.0,
                    difficulty=state["difficulty"],
                    kl_coef=kl_coef,
                    shaping=env.config["shaping"],
                    explained_variance=explained,
                    decisions_per_s=float(buf.valid.sum() / (collect_s + update_s)),
                    collect_s=collect_s,
                    update_s=update_s,
                )
                print(
                    f"maj {state['update']:4d}  décisions {state['decisions']:8d}  épisodes {state['episodes']:5d}  "
                    f"{DIFFICULTIES[state['difficulty']]:10s} victoires {100 * win_rate:4.1f} %  "
                    f"retour {np.mean(returns) if returns else 0:+.3f}  kl_bc {np.mean(stats['kl_bc']):.3f}  "
                    f"{buf.valid.sum() / (collect_s + update_s):.0f} déc/s" + ("  → difficulté suivante" if promoted else "")
                    + ("  (valeur seule)" if warmup else ""),
                    flush=True,
                )
                checkpoint("last.pt")
                if state["update"] % args.snapshot_every == 0:
                    checkpoint(f"ppo_{state['update']:05d}.pt")
        finally:
            env.close()
    checkpoint("last.pt")
    (RUNS_DIR / args.series / "ppo.json").write_text(json.dumps(state, indent=1))
    print(f"Segment terminé : mise à jour {state['update']}, {state['seconds'] / 60:.1f} min d'entraînement au total.")


if __name__ == "__main__":
    main()
