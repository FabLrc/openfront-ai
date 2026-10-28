"""Client du serveur d'environnement (env/, socket TCP local).

- ``EnvClient`` : le protocole brut (reset/step, en-tête JSON + tableaux numpy).
- ``OpenFrontEnv`` : environnement multi-agents au format PettingZoo ParallelEnv.
"""

from __future__ import annotations

import socket
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .protocol import _U32, decode_body, encode_frame

# Doivent correspondre à env/src/seat.ts (vérifié au reset).
ACTION_TYPES = ("noop", "spawn", "attack", "boat", "build", "alliance", "break")
NOOP, SPAWN, ATTACK, BOAT, BUILD, ALLIANCE, BREAK = range(len(ACTION_TYPES))
AMOUNTS = (0.1, 0.25, 0.5, 0.75)
BUILD_UNITS = ("City", "Port", "Factory", "Defense Post", "Missile Silo", "SAM Launcher")
N_CHANNELS = 13
N_PLAYER_FEATURES = 16
N_SCALARS = 14

OBS_FIELDS = ("spatial", "players", "scalars", "spawn")
MASK_FIELDS = ("type", "target", "cell", "amount", "unit", "unit_cell")


class EnvError(RuntimeError):
    """Erreur renvoyée par le serveur d'environnement."""


class EnvClient:
    """Connexion au serveur : une partie par connexion."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float | None = 60.0):
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def reset(self, **config: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Crée une partie. Clés : map, size, nations, difficulty, seed, agents,
        obsWidth, obsHeight, actionWidth, actionHeight, k, shaping, autoSpawn, maxTicks,
        record ({"run": ..., "every": N, "playToEnd": bool})."""
        return self.request({"cmd": "reset", "config": config})

    def step(
        self,
        actions: dict[str, list[int]] | None = None,
        ticks: int = 10,
        annotations: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        message: dict[str, Any] = {"cmd": "step", "ticks": ticks, "actions": actions or {}}
        if annotations:
            message["annotations"] = annotations
        return self.request(message)

    def request(self, message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        self.send(message)
        return self.receive()

    def send(self, message: dict[str, Any]) -> None:
        """Envoie une requête sans attendre la réponse (voir ``receive``) : on
        peut ainsi faire avancer plusieurs parties, sur plusieurs processus, en
        même temps."""
        self._sock.sendall(encode_frame(message))

    def receive(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        (length,) = _U32.unpack(self._recv_exact(4))
        header, buffers = decode_body(self._recv_exact(length))
        if not header.get("ok"):
            raise EnvError(header.get("error", "erreur inconnue"))
        return header, buffers

    def close(self) -> None:
        try:
            self._sock.sendall(encode_frame({"cmd": "close"}))
        except OSError:
            pass
        self._sock.close()

    def __enter__(self) -> EnvClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _recv_exact(self, n: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < n:
            chunk = self._sock.recv(min(n - len(chunks), 1 << 20))
            if not chunk:
                raise EnvError("connexion fermée par le serveur")
            chunks += chunk
        return bytes(chunks)


class OpenFrontEnv(ParallelEnv):
    """Partie FFA OpenFront : ``agents`` sièges contre ``nations`` Nations.

    Observation (dict) : ``spatial`` uint8 [C, H, W], ``players`` float32
    [K+1, F] (ligne 0 = terres neutres), ``scalars`` float32 [S], ``spawn``
    float32 [K+1, 3] (candidats de spawn) et ``masks`` (dict de masques
    uint8 : type [T], target [T, K+1], cell [T, A], amount [4], unit [U],
    unit_cell [U, A]).

    Action : MultiDiscrete [type, cible, case, quantité, unité].
    """

    metadata = {"name": "openfront_v0", "render_modes": []}

    def __init__(
        self,
        agents: int = 1,
        nations: int = 7,
        map: str = "world",
        size: str = "Compact",
        difficulty: str = "Medium",
        ticks_per_step: int = 10,
        obs_width: int = 128,
        obs_height: int = 64,
        action_width: int = 32,
        action_height: int = 16,
        k: int = 16,
        shaping: float = 1.0,
        max_ticks: int = 0,
        record: str | None = None,
        record_every: int = 1,
        record_to_end: bool = False,
        host: str = "127.0.0.1",
        port: int = 5555,
    ):
        self.config = {
            "agents": agents,
            "nations": nations,
            "map": map,
            "size": size,
            "difficulty": difficulty,
            "obsWidth": obs_width,
            "obsHeight": obs_height,
            "actionWidth": action_width,
            "actionHeight": action_height,
            "k": k,
            "shaping": shaping,
            "maxTicks": max_ticks,
        }
        # Enregistrement des parties dans runs/<record>/ (voir env/src/records.ts).
        if record is not None:
            self.config["record"] = {"run": record, "every": record_every, "playToEnd": record_to_end}
        self.ticks_per_step = ticks_per_step
        self.possible_agents = [f"agent_{i}" for i in range(agents)]
        self.agents: list[str] = []
        self.state_info: dict[str, Any] = {}
        self.spec_info: dict[str, Any] = {}
        self._client = EnvClient(host, port, timeout=None)

        rows, cells, T, U = k + 1, action_width * action_height, len(ACTION_TYPES), len(BUILD_UNITS)
        mask = lambda *shape: spaces.Box(0, 1, shape, np.uint8)  # noqa: E731
        self._obs_space = spaces.Dict(
            {
                "spatial": spaces.Box(0, 255, (N_CHANNELS, obs_height, obs_width), np.uint8),
                "players": spaces.Box(-1.0, np.inf, (rows, N_PLAYER_FEATURES), np.float32),
                "scalars": spaces.Box(0.0, np.inf, (N_SCALARS,), np.float32),
                "spawn": spaces.Box(0.0, 1.0, (rows, 3), np.float32),
                "masks": spaces.Dict(
                    {
                        "type": mask(T),
                        "target": mask(T, rows),
                        "cell": mask(T, cells),
                        "amount": mask(len(AMOUNTS)),
                        "unit": mask(U),
                        "unit_cell": mask(U, cells),
                    }
                ),
            }
        )
        self._action_space = spaces.MultiDiscrete([T, rows, cells, len(AMOUNTS), U])

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_space

    def reset(self, seed: int | str | None = None, options: dict[str, Any] | None = None):
        config = dict(self.config)
        if seed is not None:
            config["seed"] = str(seed)
        config.update(options or {})
        header, buffers = self._client.reset(**config)
        self.spec_info = header["spec"]
        self._check_spec()
        self.agents = list(header["agents"])
        self.state_info = header["state"]
        return self._observations(buffers, self.agents), header["infos"]

    def step(self, actions: dict[str, Any], annotations: dict[str, Any] | None = None):
        sent = {a: [int(x) for x in act] for a, act in actions.items() if a in self.agents}
        header, buffers = self._client.step(sent, self.ticks_per_step, annotations)
        observed = list(header["rewards"].keys())
        self.agents = list(header["agents"])
        self.state_info = header["state"]
        return (
            self._observations(buffers, observed),
            header["rewards"],
            header["terminations"],
            header["truncations"],
            header["infos"],
        )

    def state(self) -> dict[str, Any]:
        return self.state_info

    def close(self) -> None:
        self._client.close()

    def _observations(self, buffers: dict[str, np.ndarray], agents: list[str]) -> dict[str, Any]:
        out = {}
        for agent in agents:
            obs: dict[str, Any] = {f: buffers[f"{agent}/{f}"] for f in OBS_FIELDS}
            obs["masks"] = {f: buffers[f"{agent}/mask_{f}"] for f in MASK_FIELDS}
            out[agent] = obs
        return out

    def _check_spec(self) -> None:
        s = self.spec_info
        expected = (list(ACTION_TYPES), list(AMOUNTS), list(BUILD_UNITS), N_CHANNELS, N_PLAYER_FEATURES, N_SCALARS)
        got = (s["action_types"], s["amounts"], s["build_units"], len(s["channels"]), len(s["player_features"]), len(s["scalars"]))
        if expected != got:
            raise EnvError(f"client et serveur désaccordés : {got} != {expected}")
