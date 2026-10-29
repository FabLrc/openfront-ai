"""best_eval.py : garde le checkpoint de la meilleure évaluation (Hard, puis Medium)."""

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("best_eval", Path(__file__).resolve().parents[1] / "best_eval.py")
best_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(best_eval)


def write_evals(run: Path, step: int, hard: float, medium: float, hard_rank: float = 5.0) -> None:
    with open(run / "train.jsonl", "a") as f:
        f.write(json.dumps({"step": step, "update": 1, "loss_pi": 0.1}) + "\n")
        f.write(json.dumps({"step": step, "eval_medium_win_rate": medium, "eval_medium_rank": 3.0}) + "\n")
        f.write(json.dumps({"step": step, "eval_hard_win_rate": hard, "eval_hard_rank": hard_rank}) + "\n")


def run_main(monkeypatch, tmp_path, step: int) -> None:
    monkeypatch.setattr(best_eval, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["best_eval.py", "--series", "s", "--step", str(step)])
    best_eval.main()


def test_score_hard_puis_medium():
    assert best_eval.score({"eval_hard_win_rate": 0.1, "eval_medium_win_rate": 0.2}) > best_eval.score(
        {"eval_hard_win_rate": 0.05, "eval_medium_win_rate": 0.9}
    )
    assert best_eval.score({"eval_hard_win_rate": 0.1, "eval_medium_win_rate": 0.5}) > best_eval.score(
        {"eval_hard_win_rate": 0.1, "eval_medium_win_rate": 0.4}
    )


def test_garde_le_meilleur(monkeypatch, tmp_path):
    run = tmp_path / "s"
    (run / "ckpt").mkdir(parents=True)
    last = run / "ckpt" / "last.pt"

    last.write_bytes(b"pas-100")
    write_evals(run, 100, hard=0.05, medium=0.5)
    run_main(monkeypatch, tmp_path, 100)
    assert (run / "ckpt" / "best_eval.pt").read_bytes() == b"pas-100"

    # Moins bon en Hard malgré un meilleur Medium : on garde le pas 100.
    last.write_bytes(b"pas-200")
    write_evals(run, 200, hard=0.025, medium=0.9)
    run_main(monkeypatch, tmp_path, 200)
    assert (run / "ckpt" / "best_eval.pt").read_bytes() == b"pas-100"

    # Égalité en Hard, meilleur en Medium : nouveau meilleur.
    last.write_bytes(b"pas-300")
    write_evals(run, 300, hard=0.05, medium=0.6)
    run_main(monkeypatch, tmp_path, 300)
    assert (run / "ckpt" / "best_eval.pt").read_bytes() == b"pas-300"
    assert json.loads((run / "ckpt" / "best_eval.json").read_text())["step"] == 300


def test_evaluations_incompletes(monkeypatch, tmp_path):
    run = tmp_path / "s"
    (run / "ckpt").mkdir(parents=True)
    (run / "ckpt" / "last.pt").write_bytes(b"x")
    (run / "train.jsonl").write_text(json.dumps({"step": 1, "eval_medium_win_rate": 0.5}) + "\n")
    run_main(monkeypatch, tmp_path, 1)
    assert not (run / "ckpt" / "best_eval.pt").exists()
