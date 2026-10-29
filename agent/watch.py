"""Une partie solo du modèle contre 7 Nations, à vitesse réelle, à suivre sur /live.

Lancé par scripts/watch.sh (qui démarre le serveur au besoin) :
    agent/.venv/bin/python agent/watch.py [checkpoint] [--difficulty Hard] [--speed 2] [--greedy]

Sans checkpoint, prend le meilleur disponible (voir CHECKPOINTS). La partie
est enregistrée dans runs/watch/ jusqu'au vainqueur : la commande pour la
revoir dans le lecteur officiel est affichée à la fin.
"""

from __future__ import annotations

import argparse
import random
import string
import sys
import time
import webbrowser
from pathlib import Path

from openfront_ai import OpenFrontEnv
from openfront_ai.agents import PolicyAgent

ROOT = Path(__file__).resolve().parent.parent
# Du meilleur au moins bon : PPO évalué (PC), dernier PPO, imitation.
CHECKPOINTS = ["runs/ppo2/ckpt/best_eval.pt", "runs/ppo1/ckpt/last.pt", "runs/bc1/ckpt/best.pt"]


def default_checkpoint() -> Path:
    for c in CHECKPOINTS:
        if (ROOT / c).exists():
            return ROOT / c
    sys.exit(f"Aucun checkpoint trouvé ({', '.join(CHECKPOINTS)}) : indiquez-en un.")


def random_seed() -> str:
    # 10 caractères alphanumériques : la graine sert telle quelle d'identifiant de partie.
    return "w" + "".join(random.choices(string.ascii_lowercase + string.digits, k=9))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", nargs="?", help="checkpoint de la politique (défaut : le meilleur disponible)")
    parser.add_argument("--difficulty", default="Medium", choices=["Easy", "Medium", "Hard", "Impossible"])
    parser.add_argument("--nations", type=int, default=7)
    parser.add_argument("--map", default="world")
    parser.add_argument("--size", default="Compact", choices=["Compact", "Normal"])
    parser.add_argument("--speed", default="1", choices=["1", "2", "4", "10", "max"], help="multiple du temps réel (1 tick = 100 ms)")
    parser.add_argument("--greedy", action="store_true", help="action la plus probable (sinon échantillonnage)")
    parser.add_argument("--seed", default=None, help="graine (défaut : au hasard)")
    parser.add_argument("--record", default="watch", metavar="SERIE", help="série où enregistrer la partie (runs/SERIE/)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--live-url", default="http://localhost:8080/live")
    parser.add_argument("--no-open", action="store_true", help="ne pas ouvrir le navigateur")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint()
    seed = args.seed or random_seed()
    agent = PolicyAgent(str(checkpoint), greedy=args.greedy, seed=0)
    env = OpenFrontEnv(
        agents=1,
        nations=args.nations,
        map=args.map,
        size=args.size,
        difficulty=args.difficulty,
        port=args.port,
        record=args.record,
        record_to_end=True,
    )
    mode = "greedy" if args.greedy else "échantillonnage"
    print(f"Modèle : {checkpoint.relative_to(ROOT) if checkpoint.is_relative_to(ROOT) else checkpoint} ({mode})")
    print(f"Partie : {args.map} {args.size}, contre {args.nations} Nations {args.difficulty}, graine {seed}, vitesse x{args.speed}")
    start = time.perf_counter()
    try:
        # liveSpeed règle la vitesse de /live (modifiable ensuite avec les boutons de la page).
        obs, _ = env.reset(seed=seed, options={"liveSpeed": args.speed})
        if not args.no_open:
            webbrowser.open(args.live_url)
        print(f"En direct : {args.live_url}")
        info: dict = {}
        while env.agents:
            a = env.agents[0]
            action = agent.act(obs[a])
            obs, _, _, _, infos = env.step({a: action}, annotations={a: {"probs": agent.last_probs}})
            info = infos[a]
            if info["tick"] % 1000 < env.ticks_per_step:
                print(f"  tick {info['tick']:>6}  tuiles {info['tiles']:>6}  troupes {info['troops']:>8}  {info['last_action']}")
        verdict = "VICTOIRE" if info.get("won") else f"rang {info.get('rank')}/{info.get('players')}"
        print(f"Fin pour le modèle au tick {info.get('tick')} : {verdict} (vainqueur : {env.state()['winner']})")
    except KeyboardInterrupt:
        print("\nInterrompu : la partie est enregistrée telle quelle.")
    finally:
        env.close()
    print(f"Durée : {time.perf_counter() - start:.0f} s")
    print(f"Revoir la partie : scripts/replay-viewer.sh {seed}")


if __name__ == "__main__":
    main()
