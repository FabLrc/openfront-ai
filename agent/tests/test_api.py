"""Conformité PettingZoo, avec un serveur lancé pour le test sur un port libre (node requis)."""

import socket

from pettingzoo.test import parallel_api_test

from openfront_ai import OpenFrontEnv
from openfront_ai import vec
from openfront_ai.vec import ServerPool


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_parallel_api(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENFRONT_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(vec, "RUNS_DIR", tmp_path)
    with ServerPool(1, base_port=free_port(), dashboard=False, live_base_port=free_port()) as pool:
        for agents, nations in ((1, 7), (3, 2)):
            env = OpenFrontEnv(agents=agents, nations=nations, port=pool.ports[0], max_ticks=3000)
            try:
                parallel_api_test(env, num_cycles=200)
            finally:
                env.close()
