"""Garde le checkpoint de la meilleure évaluation d'une série PPO.

    agent/.venv/bin/python agent/best_eval.py --series ppo2 --step 123456

Lit les évaluations de ce pas dans runs/<série>/train.jsonl (eval_hard_*,
eval_medium_*, écrites par run_eval.py --log-to). Si elles battent la
meilleure jusqu'ici (victoires en Hard, puis victoires en Medium, puis rang
moyen en Hard), copie ckpt/last.pt vers ckpt/best_eval.pt et note le score
dans ckpt/best_eval.json.
"""

from __future__ import annotations

import argparse
import json
import shutil

from openfront_ai.vec import RUNS_DIR


def evals_at(lines: list[dict], step: int) -> dict:
    """Champs eval_* des lignes du pas `step` (la dernière l'emporte)."""
    out: dict = {}
    for line in lines:
        if line.get("step") == step:
            out.update({k: v for k, v in line.items() if k.startswith("eval_")})
    return out


def score(evals: dict) -> tuple[float, float, float]:
    """Ordre de comparaison : plus grand = meilleur."""
    return (evals.get("eval_hard_win_rate", 0.0), evals.get("eval_medium_win_rate", 0.0), -evals.get("eval_hard_rank", 8.0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--series", required=True)
    p.add_argument("--step", type=int, required=True)
    args = p.parse_args()

    run = RUNS_DIR / args.series
    lines = [json.loads(line) for line in (run / "train.jsonl").read_text().splitlines() if line.strip()]
    evals = evals_at(lines, args.step)
    if "eval_hard_win_rate" not in evals or "eval_medium_win_rate" not in evals:
        print(f"best_eval : évaluations incomplètes au pas {args.step}, rien à faire")
        return
    best_path = run / "ckpt" / "best_eval.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or score(evals) > score(best["evals"]):
        shutil.copyfile(run / "ckpt" / "last.pt", run / "ckpt" / "best_eval.pt")
        best_path.write_text(json.dumps({"step": args.step, "evals": evals}, indent=1))
        print(f"best_eval : nouveau meilleur checkpoint au pas {args.step}")
    else:
        print(f"best_eval : le meilleur reste celui du pas {best['step']}")


if __name__ == "__main__":
    main()
