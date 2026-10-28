"""Conformité PettingZoo (serveur requis : cd env && npm start).

    agent/.venv/bin/python -m unittest agent/tests/test_api.py
"""

import socket
import unittest

from pettingzoo.test import parallel_api_test

from openfront_ai import OpenFrontEnv


def server_up(port: int = 5555) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(server_up(), "serveur d'environnement absent")
class ApiTest(unittest.TestCase):
    def test_parallel_api(self):
        for agents, nations in ((1, 7), (3, 2)):
            env = OpenFrontEnv(agents=agents, nations=nations)
            try:
                parallel_api_test(env, num_cycles=200)
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
