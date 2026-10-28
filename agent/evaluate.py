"""Évalue les agents de référence : M parties FFA Compact (1 siège agent + 7 Nations Medium).

Le serveur doit tourner : (cd env && npm start). Exemple :
    agent/.venv/bin/python agent/evaluate.py --games 10 --agent both
"""

from __future__ import annotations

import argparse
import statistics
import time

from openfront_ai import OpenFrontEnv
from openfront_ai.agents import RandomAgent, RuleAgent

AGENTS = {"random": RandomAgent, "rule": RuleAgent}


def play(env: OpenFrontEnv, policy_cls, seed: str, policy_seed: int) -> dict:
    policies = {a: policy_cls(seed=policy_seed + i) for i, a in enumerate(env.possible_agents)}
    t_reset = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    start = time.perf_counter()
    reset_seconds = start - t_reset
    steps = 0
    results = {}
    while env.agents:
        actions = {a: policies[a].act(obs[a]) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1
        for a, info in infos.items():
            if terms[a] or truncs[a]:
                results[a] = info
    elapsed = time.perf_counter() - start
    first = results[env.possible_agents[0]]
    return {
        "won": first.get("won", False),
        "rank": first.get("rank"),
        "players": first.get("players"),
        "agent_tick": first["tick"],
        "end_tick": env.state()["tick"],
        "winner": env.state()["winner"],
        "steps": steps,
        "seconds": elapsed,
        "reset_seconds": reset_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--agent", choices=["random", "rule", "both"], default="both")
    parser.add_argument("--agents", type=int, default=1, help="nombre de sièges (tous joués par le même agent)")
    parser.add_argument("--nations", type=int, default=7)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--map", default="world")
    parser.add_argument("--seed", default="eval", help="préfixe des graines de partie")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    names = ["random", "rule"] if args.agent == "both" else [args.agent]
    env = OpenFrontEnv(agents=args.agents, nations=args.nations, difficulty=args.difficulty, map=args.map, port=args.port)
    try:
        for name in names:
            games = []
            for g in range(args.games):
                r = play(env, AGENTS[name], f"{args.seed}-{g}", policy_seed=g)
                games.append(r)
                outcome = "VICTOIRE" if r["won"] else f"rang {r['rank']}/{r['players']}"
                print(
                    f"[{name}] partie {g + 1}/{args.games} : {outcome}"
                    f", fin du siège au tick {r['agent_tick']}, vainqueur {r['winner']}, "
                    f"{r['steps']} pas en {r['seconds']:.1f} s",
                    flush=True,
                )
            steps = sum(r["steps"] for r in games)
            secs = sum(r["seconds"] for r in games)
            # Ticks joués pendant les pas (après la phase de spawn, jouée au premier pas).
            ticks = sum(r["end_tick"] for r in games)
            print(
                f"\n== {name} ({args.games} parties, {args.agents} siège(s), {args.nations} Nations {args.difficulty}) ==\n"
                f"  taux de victoire : {100 * sum(r['won'] for r in games) / len(games):.0f} %\n"
                f"  rang moyen       : {statistics.mean(r['rank'] for r in games):.2f} / {games[0]['players']}\n"
                f"  durée moyenne    : {statistics.mean(r['agent_tick'] for r in games):.0f} ticks de vie du siège, "
                f"{statistics.mean(r['seconds'] for r in games):.1f} s par partie\n"
                f"  débit            : {steps / secs:.0f} pas/s, {ticks / secs:.0f} ticks/s "
                f"(hors reset : {statistics.mean(r['reset_seconds'] for r in games):.2f} s par partie)\n",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
