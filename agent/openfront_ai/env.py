"""Client du serveur d'environnement (env/, socket TCP local).

Chaque message est un entier non signé 32 bits big-endian (longueur) suivi
d'un JSON UTF-8. Voir env/src/server.ts pour le détail des commandes.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

_HEADER = struct.Struct(">I")


class EnvError(RuntimeError):
    """Erreur renvoyée par le serveur d'environnement."""


class OpenFrontEnv:
    """Une partie OpenFront headless, pilotée par ticks.

    >>> with OpenFrontEnv() as env:
    ...     state = env.reset(map="world", size="Compact", nations=8, seed="s")
    ...     while not state["done"]:
    ...         state = env.step(10)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float | None = 60.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reset(
        self,
        map: str = "world",
        size: str = "Compact",
        nations: int = 8,
        difficulty: str = "Medium",
        seed: str = "openfront-ai",
    ) -> dict[str, Any]:
        """Crée une nouvelle partie (phase de spawn déjà jouée) et renvoie son état."""
        config = {"map": map, "size": size, "nations": nations, "difficulty": difficulty, "seed": seed}
        return self._request({"cmd": "reset", "config": config})

    def step(self, n_ticks: int = 10) -> dict[str, Any]:
        """Avance de n_ticks (moins si la partie se termine) et renvoie l'état."""
        return self._request({"cmd": "step", "ticks": n_ticks})

    def close(self) -> None:
        try:
            self._send({"cmd": "close"})
        except OSError:
            pass
        self._sock.close()

    def __enter__(self) -> OpenFrontEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._send(message)
        reply = self._recv()
        if not reply.get("ok"):
            raise EnvError(reply.get("error", "erreur inconnue"))
        return reply["state"]

    def _send(self, message: dict[str, Any]) -> None:
        body = json.dumps(message).encode("utf-8")
        self._sock.sendall(_HEADER.pack(len(body)) + body)

    def _recv(self) -> dict[str, Any]:
        (length,) = _HEADER.unpack(self._recv_exact(_HEADER.size))
        return json.loads(self._recv_exact(length))

    def _recv_exact(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self._sock.recv(n - len(chunks))
            if not chunk:
                raise EnvError("connexion fermée par le serveur")
            chunks += chunk
        return bytes(chunks)
