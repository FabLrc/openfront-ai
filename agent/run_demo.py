"""Joue une partie FFA Compact à 8 Nations jusqu'à la fin et affiche le vainqueur.

Le serveur doit tourner : (cd env && npm start). Exemple :
    agent/.venv/bin/python agent/run_demo.py --seed demo
"""

from __future__ import annotations

import argparse
import time

from openfront_ai import OpenFrontEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default="world")
    parser.add_argument("--size", default="Compact", choices=["Compact", "Normal"])
    parser.add_argument("--nations", type=int, default=8)
    parser.add_argument("--difficulty", default="Medium", choices=["Easy", "Medium", "Hard", "Impossible"])
    parser.add_argument("--seed", default="openfront-ai")
    parser.add_argument("--ticks", type=int, default=10, help="ticks par step")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    start = time.perf_counter()
    with OpenFrontEnv(port=args.port, timeout=None) as env:
        state = env.reset(
            map=args.map, size=args.size, nations=args.nations, difficulty=args.difficulty, seed=args.seed
        )
        print(f"Partie lancée : {args.map} {args.size}, {state['alivePlayers']} joueurs, tick {state['tick']}")
        last_report = 0
        while not state["done"]:
            state = env.step(args.ticks)
            if state["tick"] - last_report >= 2000:
                last_report = state["tick"]
                leader = max(state["players"], key=lambda p: p["tiles"])
                print(
                    f"  tick {state['tick']:>6}  vivants {state['alivePlayers']}  "
                    f"en tête {leader['name']} ({leader['tiles']} tuiles)"
                )

    elapsed = time.perf_counter() - start
    print(f"Vainqueur : {state['winner']} au tick {state['tick']} ({elapsed:.1f} s, hash {state['hash']})")


if __name__ == "__main__":
    main()
