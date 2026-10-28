"""Collecte des données d'imitation : des Nations jouent entre elles, on
enregistre pour quelques-unes l'observation telle qu'un agent la verrait et
leur décision traduite dans notre espace d'actions.

    agent/.venv/bin/python agent/collect_bc.py --minutes 5
    caffeinate -i agent/.venv/bin/python agent/collect_bc.py --minutes 30

Chaque lancement ajoute des shards data/bc/<run>_NNNN.npz (indépendants : une
collecte interrompue garde ses shards écrits) et met à jour data/bc/stats.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from openfront_ai.env import EnvClient, MASK_FIELDS, OBS_FIELDS
from openfront_ai.vec import ROOT, ServerPool, performance_cores

DATA_DIR = ROOT / "data" / "bc"


class ShardWriter:
    def __init__(self, run: str, size: int):
        self.run = run
        self.size = size
        self.buffer: list[dict[str, np.ndarray]] = []
        self.index = 0
        self.written = 0
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, sample: dict[str, np.ndarray]) -> None:
        self.buffer.append(sample)
        if len(self.buffer) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        arrays = {k: np.stack([s[k] for s in self.buffer]) for k in self.buffer[0]}
        path = DATA_DIR / f"{self.run}_{self.index:04d}.npz"
        np.savez_compressed(path, **arrays)
        self.written += len(self.buffer)
        self.index += 1
        self.buffer = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=5)
    parser.add_argument("--procs", type=int, default=performance_cores())
    parser.add_argument("--games-per-proc", type=int, default=2)
    parser.add_argument("--observe", type=int, default=4, help="Nations observées par partie")
    parser.add_argument("--nations", type=int, default=8)
    parser.add_argument("--difficulties", default="Hard,Impossible,Hard,Impossible,Medium", help="tirées à tour de rôle")
    parser.add_argument("--keep-noop", type=float, default=0.1, help="part des pas « rien » gardés (poids 1/p)")
    parser.add_argument("--max-ticks", type=int, default=25000)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--base-port", type=int, default=6200)
    args = parser.parse_args()

    run = time.strftime("%Y%m%d-%H%M%S")
    difficulties = args.difficulties.split(",")
    rng = np.random.default_rng(int(time.time()))
    writer = ShardWriter(run, args.shard_size)
    stats: Counter[str] = Counter()
    start = time.monotonic()

    with ServerPool(args.procs, base_port=args.base_port, dashboard=False) as pool:
        clients = [EnvClient(port=pool.ports[i % pool.n], timeout=None) for i in range(pool.n * args.games_per_proc)]
        episode = [0] * len(clients)
        diff = [""] * len(clients)
        prev: list[dict[str, dict[str, np.ndarray]]] = [{} for _ in clients]

        def reset(i: int) -> None:
            episode[i] += 1
            diff[i] = difficulties[(i + episode[i]) % len(difficulties)]
            clients[i].send({"cmd": "reset", "config": {
                "agents": 0, "nations": args.nations, "difficulty": diff[i], "observeNations": args.observe,
                "maxTicks": args.max_ticks, "seed": f"bc-{run}-{i}-{episode[i]}",
            }})

        def take(i: int, header: dict, buffers: dict) -> None:
            """Range les observations reçues comme « précédentes » de la partie i."""
            prev[i] = {}
            for name in header.get("observed") or []:
                obs = {f: buffers[f"{name}/{f}"] for f in OBS_FIELDS}
                obs.update({f"mask_{f}": buffers[f"{name}/mask_{f}"] for f in MASK_FIELDS})
                prev[i][name] = obs

        for i in range(len(clients)):
            reset(i)
        for i, c in enumerate(clients):
            take(i, *c.receive())

        last_print = start
        while time.monotonic() - start < args.minutes * 60:
            for c in clients:
                c.send({"cmd": "step", "ticks": 10})
            for i, c in enumerate(clients):
                header, buffers = c.receive()
                for name, label in (header.get("labels") or {}).items():
                    obs = prev[i].get(name)
                    if obs is None:
                        continue
                    stats["pas"] += 1
                    stats[f"type:{label['kind'].split(':')[0]}"] += 1
                    for k in label["all"]:
                        stats[f"décision:{k}"] += 1
                    if label["mapped"]:
                        stats["traduites"] += 1
                        stats[f"traduites:{label['kind']}"] += 1
                        if label["valid"]:
                            stats[f"valides:{label['kind']}"] += 1
                    if not label["valid"]:
                        continue  # hors de nos masques (ou décision non traduite) : inutilisable
                    weight = 1.0
                    if label["kind"] == "noop":
                        if rng.random() >= args.keep_noop:
                            continue
                        weight = 1.0 / args.keep_noop
                    stats["échantillons"] += 1
                    writer.add({
                        **obs,
                        "action": np.asarray(label["action"], dtype=np.int16),
                        "weight": np.float32(weight),
                        "difficulty": np.int8(["Easy", "Medium", "Hard", "Impossible"].index(diff[i])),
                    })
                take(i, header, buffers)
                if header["state"]["done"] or header["state"]["tick"] >= args.max_ticks or not header.get("observed"):
                    stats["parties"] += 1
                    reset(i)
                    take(i, *c.receive())
            now = time.monotonic()
            if now - last_print > 30:
                last_print = now
                print(f"  {(now - start) / 60:4.1f} min  parties {stats['parties']}  pas {stats['pas']}  "
                      f"échantillons {stats['échantillons']}  traduites {stats['traduites']}", flush=True)
        for c in clients:
            c.close()
    writer.flush()

    # Statistiques cumulées sur toutes les collectes.
    stats_file = DATA_DIR / "stats.json"
    total = Counter(json.loads(stats_file.read_text())) if stats_file.exists() else Counter()
    total.update(stats)
    stats_file.write_text(json.dumps(dict(sorted(total.items())), ensure_ascii=False, indent=1))
    decisions = stats["pas"] - stats["type:noop"]
    print(f"Collecte {run} : {writer.written} échantillons en {writer.index} shard(s), {stats['parties']} parties.")
    if decisions:
        valid = sum(v for k, v in stats.items() if k.startswith("valides:"))
        print(f"  pas avec décision {100 * decisions / stats['pas']:.1f} %, traduites {100 * stats['traduites'] / decisions:.1f} %, "
              f"valides selon nos masques {100 * valid / max(1, stats['traduites']):.1f} %")


if __name__ == "__main__":
    main()
