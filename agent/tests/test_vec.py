"""VecOpenFrontEnv : lance ses propres serveurs (node requis).

    agent/.venv/bin/python -m pytest agent/tests
"""

import json

import numpy as np
import pytest

from openfront_ai import vec
from openfront_ai.agents import RandomAgent


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    # Les serveurs lancés héritent de la variable : leurs métriques vont aussi là.
    monkeypatch.setenv("OPENFRONT_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(vec, "RUNS_DIR", tmp_path)
    return tmp_path


def seat_obs(obs, b):
    out = {k: v[b] for k, v in obs.items() if k != "masks"}
    out["masks"] = {k: v[b] for k, v in obs["masks"].items()}
    return out


def test_batch_reset_auto_et_metriques(runs_dir):
    with vec.ServerPool(2, base_port=6300, dashboard=False) as pool:
        # Épisodes très courts (600 ticks) pour voir la réinitialisation automatique.
        env = vec.VecOpenFrontEnv(
            pool.ports, games_per_server=2, agents=2, nations=2, max_ticks=600, series="test", throughput_interval=0.5
        )
        try:
            obs, _ = env.reset()
            assert env.num_envs == 8
            assert obs["spatial"].shape[0] == 8 and obs["spatial"].dtype == np.uint8
            assert obs["masks"]["type"].shape == (8, 7)
            assert env.active.all()
            policies = [RandomAgent(seed=b) for b in range(env.num_envs)]
            ended = 0
            for _ in range(120):
                actions = np.zeros((env.num_envs, 5), dtype=np.int64)
                for b in np.flatnonzero(env.active):
                    actions[b] = policies[b].act(seat_obs(obs, b))
                obs, rewards, terms, truncs, infos = env.step(actions)
                assert rewards.shape == (8,)
                ended += int((terms | truncs).sum())
                for b in np.flatnonzero(terms | truncs):
                    assert "rank" in infos[b] and "reset" in infos[b]
            assert env.episodes > 0 and ended > 0
        finally:
            env.close()

    lines = [json.loads(line) for line in (runs_dir / "test" / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) >= ended
    for key in ("won", "rank", "share", "ticks", "reward", "actions", "refused", "truncated"):
        assert key in lines[0]
    # Toutes les parties s'arrêtent au plus tard à la limite de durée.
    assert max(line["ticks"] for line in lines) <= 600 + 10
    assert (runs_dir / "test" / "throughput.jsonl").exists()


def test_log_train(runs_dir):
    vec.log_train("entrainement", step=1, loss=0.5, entropy=1.2)
    line = json.loads((runs_dir / "entrainement" / "train.jsonl").read_text())
    assert line["loss"] == 0.5 and line["step"] == 1
