"""Joue une partie FFA Compact jusqu'à la fin et affiche le vainqueur.

Par défaut, 8 Nations sans agent. Avec --agent, un siège est joué par un
agent de référence contre 7 Nations. Le serveur doit tourner : (cd env && npm start).
    agent/.venv/bin/python agent/run_demo.py --seed demo
    agent/.venv/bin/python agent/run_demo.py --agent rule --seed demo
"""

from __future__ import annotations

import argparse
import time

from openfront_ai import EnvClient, OpenFrontEnv
from openfront_ai.agents import RandomAgent, RuleAgent
from openfront_ai.env import ACTION_TYPES


def run_nations(args: argparse.Namespace) -> None:
    with EnvClient(port=args.port, timeout=None) as env:
        header, _ = env.reset(map=args.map, size=args.size, nations=args.nations, difficulty=args.difficulty, seed=args.seed)
        state = header["state"]
        print(f"Partie lancée : {args.map} {args.size}, {state['alivePlayers']} joueurs, tick {state['tick']}")
        last_report = 0
        while not state["done"]:
            header, _ = env.step(ticks=args.ticks)
            state = header["state"]
            if state["tick"] - last_report >= 2000:
                last_report = state["tick"]
                leader = max(state["players"], key=lambda p: p["tiles"])
                print(f"  tick {state['tick']:>6}  vivants {state['alivePlayers']}  en tête {leader['name']} ({leader['tiles']} tuiles)")
    print(f"Vainqueur : {state['winner']} au tick {state['tick']} (hash {state['hash']})")


def run_agent(args: argparse.Namespace) -> None:
    agent = (RuleAgent if args.agent == "rule" else RandomAgent)(seed=0)
    env = OpenFrontEnv(agents=1, nations=args.nations - 1, map=args.map, size=args.size, difficulty=args.difficulty, ticks_per_step=args.ticks, port=args.port)
    try:
        obs, _ = env.reset(seed=args.seed)
        print(f"Partie lancée : {args.map} {args.size}, agent « {args.agent} » contre {args.nations - 1} Nations")
        info: dict = {}
        while env.agents:
            a = env.agents[0]
            action = agent.act(obs[a])
            # Pour /live : nombre d'actions valides par type (les vraies probabilités viendront de la politique).
            m = obs[a]["masks"]["type"].astype(float)
            probs = dict(zip(ACTION_TYPES, (m / m.sum()).round(2).tolist()))
            obs, _, terms, truncs, infos = env.step({a: action}, annotations={a: {"probs": probs}})
            info = infos[a]
            if info["tick"] % 2000 < args.ticks:
                print(f"  tick {info['tick']:>6}  tuiles {info['tiles']:>6}  troupes {info['troops']:>8}  dernière action : {info['last_action']}")
        verdict = "VICTOIRE" if info.get("won") else f"rang {info.get('rank')}/{info.get('players')}"
        print(f"Fin pour l'agent au tick {info['tick']} : {verdict} (vainqueur de la partie : {env.state()['winner']})")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", default="world")
    parser.add_argument("--size", default="Compact", choices=["Compact", "Normal"])
    parser.add_argument("--nations", type=int, default=8, help="nombre total de joueurs (agent compris)")
    parser.add_argument("--difficulty", default="Medium", choices=["Easy", "Medium", "Hard", "Impossible"])
    parser.add_argument("--seed", default="openfront-ai")
    parser.add_argument("--ticks", type=int, default=10, help="ticks par pas")
    parser.add_argument("--agent", choices=["random", "rule"], help="un siège joué par cet agent")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()
    start = time.perf_counter()
    (run_agent if args.agent else run_nations)(args)
    print(f"Durée : {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
