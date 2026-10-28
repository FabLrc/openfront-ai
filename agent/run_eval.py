"""Série d'évaluation en parallèle : plusieurs processus serveur, plusieurs
parties chacun, un agent de référence dans chaque siège. Les métriques vont
dans runs/<série>/ et s'affichent sur http://localhost:8080/.

    agent/.venv/bin/python agent/run_eval.py --series regles --minutes 5
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from openfront_ai.agents import PolicyAgent, RandomAgent, RuleAgent
from openfront_ai.vec import RUNS_DIR, ServerPool, VecOpenFrontEnv, performance_cores

AGENTS = {"random": RandomAgent, "rule": RuleAgent}


def seat_obs(obs: dict, b: int) -> dict:
    out = {k: v[b] for k, v in obs.items() if k != "masks"}
    out["masks"] = {k: v[b] for k, v in obs["masks"].items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", required=True, help="nom de la série (dossier runs/SERIE)")
    parser.add_argument("--agent", choices=[*sorted(AGENTS), "policy"], default="rule")
    parser.add_argument("--checkpoint", help="checkpoint de la politique (--agent policy)")
    parser.add_argument("--greedy", action="store_true", help="politique : action la plus probable au lieu d'échantillonner")
    parser.add_argument("--procs", type=int, default=performance_cores(), help="processus serveur (défaut : cœurs performance)")
    parser.add_argument("--games-per-proc", type=int, default=2)
    parser.add_argument("--agents", type=int, default=1, help="sièges d'agents par partie")
    parser.add_argument("--nations", type=int, default=7)
    parser.add_argument("--difficulty", default="Medium")
    parser.add_argument("--max-ticks", type=int, default=30000, help="limite de durée d'un épisode (fin par classement)")
    parser.add_argument("--minutes", type=float, default=0, help="durée de la série (0 : selon --episodes)")
    parser.add_argument("--episodes", type=int, default=0, help="nombre d'épisodes de sièges (0 : selon --minutes)")
    parser.add_argument("--record-every", type=int, default=0, help="enregistre une partie sur N (0 : aucune)")
    parser.add_argument("--base-port", type=int, default=6000)
    parser.add_argument("--seed", help="préfixe des graines (même valeur = mêmes parties d'un agent à l'autre)")
    parser.add_argument("--no-dashboard", action="store_true", help="ne sert pas le tableau de bord sur :8080")
    args = parser.parse_args()
    if args.minutes <= 0 and args.episodes <= 0:
        args.minutes = 5

    print(f"Série « {args.series} » : {args.procs} processus × {args.games_per_proc} parties, agent {args.agent}", flush=True)
    with ServerPool(args.procs, base_port=args.base_port, dashboard=not args.no_dashboard, label=args.series) as pool:
        env = VecOpenFrontEnv(
            pool.ports,
            games_per_server=args.games_per_proc,
            agents=args.agents,
            nations=args.nations,
            difficulty=args.difficulty,
            max_ticks=args.max_ticks,
            seed=args.seed or f"{args.series}-{int(time.time())}",
            series=args.series,
            record=args.series if args.record_every > 0 else None,
            record_every=max(1, args.record_every),
        )
        if args.agent == "policy":
            if not args.checkpoint:
                raise SystemExit("--agent policy demande --checkpoint")
            policy = PolicyAgent(args.checkpoint, greedy=args.greedy, seed=0)
        else:
            policies = [AGENTS[args.agent](seed=i) for i in range(env.num_envs)]
        start = time.monotonic()
        obs, _ = env.reset()
        done_episodes = 0
        results: list[dict] = []
        last_print = start
        try:
            while True:
                actions = np.zeros((env.num_envs, 5), dtype=np.int64)
                active = np.flatnonzero(env.active)
                if args.agent == "policy":
                    if len(active):
                        # Toutes les décisions du lot en un seul passage du réseau.
                        sub = {k: v[active] for k, v in obs.items() if k != "masks"}
                        sub["masks"] = {k: v[active] for k, v in obs["masks"].items()}
                        actions[active] = policy.act_batch(sub)
                else:
                    for b in active:
                        actions[b] = policies[b].act(seat_obs(obs, b))
                obs, _, terms, truncs, infos = env.step(actions)
                for b in np.flatnonzero(terms | truncs):
                    done_episodes += 1
                    results.append(infos[b])
                now = time.monotonic()
                if now - last_print > 15:
                    last_print = now
                    recent = results[-50:]
                    tp = getattr(env, "last_throughput", None)
                    print(
                        f"  {(now - start) / 60:5.1f} min  épisodes {done_episodes:4d}  "
                        f"victoires {100 * np.mean([r.get('won', False) for r in recent]) if recent else 0:4.1f} %  "
                        f"rang moyen {np.mean([r.get('rank', np.nan) for r in recent]) if recent else float('nan'):4.2f}  "
                        + (f"{tp['decisions_per_s']:.0f} décisions/s, {tp['ticks_per_s']:.0f} ticks/s" if tp else ""),
                        flush=True,
                    )
                if args.episodes and done_episodes >= args.episodes:
                    break
                if args.minutes and now - start >= args.minutes * 60:
                    break
        finally:
            env.close()
    print(f"Terminé : {done_episodes} épisodes ; métriques dans {RUNS_DIR / args.series}/metrics.jsonl")
    if results:
        print(json.dumps({
            "episodes": len(results),
            "victoires_%": round(100 * float(np.mean([r.get("won", False) for r in results])), 1),
            "rang_moyen": round(float(np.mean([r.get("rank", np.nan) for r in results])), 2),
        }, ensure_ascii=False))
        print(f"  (détail par épisode, dont la part de territoire finale : {RUNS_DIR / args.series}/metrics.jsonl)")


if __name__ == "__main__":
    main()
