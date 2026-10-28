"""Plusieurs parties en parallèle, regroupées en un seul batch de sièges.

- ``ServerPool`` lance N processus serveur d'environnement (un par cœur
  performance par défaut). Le premier sert aussi le tableau de bord
  (http://localhost:8080/) et l'API des replays si ces ports sont libres.
- ``VecOpenFrontEnv`` répartit G parties sur ces processus (une connexion par
  partie). À chaque pas, il envoie les actions de tous les sièges, puis lit
  toutes les réponses : les processus calculent en même temps. Les
  observations de tous les sièges arrivent empilées (numpy, [B, ...]). Une
  partie dont tous les sièges ont fini est réinitialisée en arrière-plan :
  ses sièges sautent un pas (inactifs) au lieu de bloquer tout le lot.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .env import ACTION_TYPES, AMOUNTS, BUILD_UNITS, MASK_FIELDS, NOOP, OBS_FIELDS, EnvClient, EnvError

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("OPENFRONT_RUNS_DIR", ROOT / "runs"))


def performance_cores() -> int:
    """Nombre de cœurs performance (macOS), sinon la moitié des cœurs."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
            return max(1, int(out.stdout.strip()))
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return max(1, (os.cpu_count() or 2) // 2)


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _port_open(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
        return True
    except OSError:
        return False


class ServerPool:
    """Processus serveur d'environnement (env/src/main.ts) lancés localement."""

    def __init__(
        self,
        n: int | None = None,
        base_port: int = 6000,
        dashboard: bool = True,
        verbose: bool = False,
        label: str = "série",
        live_base_port: int = 7100,  # 7000 : récepteur AirPlay de macOS
    ):
        """``label`` nomme ces processus dans le sélecteur de /live : chacun sert
        sa propre page /live sur ``live_base_port + i`` (si le port est libre),
        enregistrée dans runs/live-sources.json pour le serveur principal."""
        self.n = n or performance_cores()
        self.ports = [base_port + i for i in range(self.n)]
        self.label = label
        self.live: list[dict[str, Any]] = []
        env_dir = ROOT / "env"
        tsx = env_dir / "node_modules" / ".bin" / "tsx"
        path = os.environ.get("PATH", "")
        # Node 24 : NODE_BIN si fourni, sinon l'emplacement Homebrew habituel s'il existe.
        node_bin = os.environ.get("NODE_BIN") or ("/opt/homebrew/opt/node@24/bin" if Path("/opt/homebrew/opt/node@24/bin").exists() else "")
        env = {**os.environ, "PATH": f"{node_bin}:{path}" if node_bin else path}
        self.procs: list[subprocess.Popen] = []
        for i, port in enumerate(self.ports):
            if not _port_free(port):
                self.close()
                raise EnvError(f"port {port} déjà utilisé")
            first = i == 0 and dashboard
            http = 8080 if first and _port_free(8080) else live_base_port + i if _port_free(live_base_port + i) else 0
            api = 8787 if first and _port_free(8787) else 0
            if http:
                self.live.append({"label": f"{label} · processus {i + 1}", "http_port": http, "env_port": port})
            cmd = [str(tsx), "src/main.ts", "--port", str(port), "--http-port", str(http), "--api-port", str(api)]
            out = None if verbose else subprocess.DEVNULL
            self.procs.append(subprocess.Popen(cmd, cwd=env_dir, env=env, stdout=out, stderr=out))
        deadline = time.monotonic() + 60
        for port, proc in zip(self.ports, self.procs):
            while not _port_open(port):
                if proc.poll() is not None or time.monotonic() > deadline:
                    self.close()
                    raise EnvError(f"le serveur du port {port} n'a pas démarré")
                time.sleep(0.1)
        for entry, proc in zip(self.live, self.procs):
            entry["pid"] = proc.pid
        self._register(add=True)

    def _register(self, add: bool) -> None:
        """Ajoute ou retire nos pages /live dans runs/live-sources.json."""
        path = RUNS_DIR / "live-sources.json"
        try:
            entries = json.loads(path.read_text()) if path.exists() else []
        except (OSError, json.JSONDecodeError):
            entries = []
        mine = {e["http_port"] for e in self.live}
        entries = [e for e in entries if e.get("http_port") not in mine]
        if add:
            entries += self.live
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, ensure_ascii=False, indent=1))

    def close(self) -> None:
        if self.procs:
            self._register(add=False)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []

    def __enter__(self) -> ServerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class _Game:
    client: EnvClient
    index: int
    episode: int = 0
    tick: int = 0
    agents: list[str] = field(default_factory=list)
    # Réinitialisation envoyée, réponse pas encore lue.
    resetting: bool = False


class VecOpenFrontEnv:
    """B = parties × sièges. ``reset()`` puis ``step(actions)`` avec
    ``actions`` de forme [B, 5] ; seuls les sièges ``active`` sont joués (les
    autres attendent la réinitialisation de leur partie, leurs actions sont
    ignorées). Chaque épisode est borné à ``max_ticks`` (fin propre : classement
    par territoire)."""

    def __init__(
        self,
        ports: list[int],
        games_per_server: int = 1,
        agents: int = 1,
        nations: int = 7,
        map: str = "world",
        size: str = "Compact",
        difficulty: str = "Medium",
        ticks_per_step: int = 10,
        max_ticks: int = 30000,
        seed: str = "vec",
        series: str | None = None,
        record: str | None = None,
        record_every: int = 1,
        throughput_interval: float = 10.0,
        **options: Any,
    ):
        self.config: dict[str, Any] = {
            "agents": agents,
            "nations": nations,
            "map": map,
            "size": size,
            "difficulty": difficulty,
            "maxTicks": max_ticks,
            **options,
        }
        if series:
            self.config["series"] = series
        if record:
            self.config["record"] = {"run": record, "every": record_every}
        self.series = series
        self.seed = seed
        self.ticks_per_step = ticks_per_step
        self.n_agents = agents
        self.servers = len(ports)
        self.games = [
            _Game(EnvClient(port=ports[i % len(ports)], timeout=None), i) for i in range(len(ports) * games_per_server)
        ]
        self.num_envs = len(self.games) * agents
        self.names = [f"agent_{i}" for i in range(agents)]
        self.active = np.zeros(self.num_envs, dtype=bool)
        self.obs: dict[str, Any] | None = None
        self.spec: dict[str, Any] = {}
        self.throughput_interval = throughput_interval
        self._t_mark = time.monotonic()
        self._decisions = 0
        self._ticks = 0
        self._server_ms = 0.0
        self._server_steps = 0
        self.episodes = 0

    # ── API ──

    def reset(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for g in self.games:
            self._send_reset(g)
        for g in self.games:
            self._receive_reset(g, infos)
        return self.obs, infos

    def step(self, actions: np.ndarray, annotations: dict[int, Any] | None = None):
        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs, 5)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        # Parties en cours de réinitialisation : leurs sièges étaient inactifs
        # quand les actions ont été choisies ; elles sautent ce pas.
        resetting = [g for g in self.games if g.resetting]
        stepping = [g for g in self.games if not g.resetting]
        for g in stepping:
            acts = {}
            notes = {}
            for j, name in enumerate(self.names):
                b = g.index * self.n_agents + j
                if name in g.agents:
                    acts[name] = actions[b].tolist()
                    self._decisions += 1
                    if annotations and b in annotations:
                        notes[name] = annotations[b]
            g.client.send({"cmd": "step", "ticks": self.ticks_per_step, "actions": acts, **({"annotations": notes} if notes else {})})
        for g in resetting:
            self._receive_reset(g, infos)
        for g in stepping:
            header, buffers = g.client.receive()
            self._server_ms += header.get("server_ms", 0.0)
            self._server_steps += 1
            self._ticks += header["state"]["tick"] - g.tick
            g.tick = header["state"]["tick"]
            for j, name in enumerate(self.names):
                b = g.index * self.n_agents + j
                if name not in header["rewards"]:
                    continue
                rewards[b] = header["rewards"][name]
                terminated[b] = header["terminations"][name]
                truncated[b] = header["truncations"][name]
                infos[b] = header["infos"][name]
                self._fill(b, name, buffers)
            g.agents = list(header["agents"])
            for j, name in enumerate(self.names):
                self.active[g.index * self.n_agents + j] = name in g.agents
            if not g.agents or header["state"]["done"]:
                # Réinitialisation automatique, sans attendre la réponse : elle
                # est lue au pas suivant, pendant que les autres parties jouent.
                # D'ici là, ces sièges sont inactifs ; infos garde le bilan de
                # l'épisode fini, et infos[b]["reset"] arrivera au pas suivant.
                self.episodes += 1
                self._send_reset(g)
                for j in range(self.n_agents):
                    self.active[g.index * self.n_agents + j] = False
        self._log_throughput()
        return self.obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        for g in self.games:
            g.client.close()

    def __enter__(self) -> VecOpenFrontEnv:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ── interne ──

    def _send_reset(self, g: _Game) -> None:
        g.resetting = True
        g.episode += 1
        g.tick = 0
        g.client.send({"cmd": "reset", "config": {**self.config, "seed": f"{self.seed}-{g.index}-{g.episode}"}})

    def _receive_reset(self, g: _Game, infos: list[dict[str, Any]]) -> None:
        header, buffers = g.client.receive()
        g.resetting = False
        if self.obs is None:
            self._allocate(header["spec"], buffers)
        g.agents = list(header["agents"])
        g.tick = header["state"]["tick"]
        for j, name in enumerate(self.names):
            b = g.index * self.n_agents + j
            self._fill(b, name, buffers)
            self.active[b] = name in g.agents
            infos[b]["reset"] = header["infos"].get(name, {})

    def _allocate(self, spec: dict[str, Any], buffers: dict[str, np.ndarray]) -> None:
        self.spec = spec
        name = self.names[0]
        self.obs = {f: np.zeros((self.num_envs, *buffers[f"{name}/{f}"].shape), buffers[f"{name}/{f}"].dtype) for f in OBS_FIELDS}
        self.obs["masks"] = {
            f: np.zeros((self.num_envs, *buffers[f"{name}/mask_{f}"].shape), np.uint8) for f in MASK_FIELDS
        }

    def _fill(self, b: int, name: str, buffers: dict[str, np.ndarray]) -> None:
        if f"{name}/spatial" not in buffers:
            return
        for f in OBS_FIELDS:
            self.obs[f][b] = buffers[f"{name}/{f}"]
        for f in MASK_FIELDS:
            self.obs["masks"][f][b] = buffers[f"{name}/mask_{f}"]

    def _log_throughput(self) -> None:
        now = time.monotonic()
        dt = now - self._t_mark
        if dt < self.throughput_interval:
            return
        line = {
            "date": datetime.now(timezone.utc).isoformat(),
            "decisions_per_s": self._decisions / dt,
            "ticks_per_s": self._ticks / dt,
            "servers": self.servers,
            "games": len(self.games),
            "seats": self.num_envs,
            "episodes": self.episodes,
            # Calcul moyen d'un pas de partie côté serveur ; le reste du temps
            # est l'attente de la partie la plus lente du lot, les sockets et Python.
            "server_ms_per_step": self._server_ms / max(1, self._server_steps),
        }
        self.last_throughput = line
        if self.series:
            path = RUNS_DIR / self.series
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "throughput.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
        self._t_mark = now
        self._decisions = 0
        self._ticks = 0
        self._server_ms = 0.0
        self._server_steps = 0


def log_train(series: str, **values: Any) -> None:
    """Ajoute une ligne de métriques d'entraînement (perte, entropie… : champs
    libres, ``step`` en abscisse) à runs/<série>/train.jsonl ; le tableau de
    bord en fait une courbe par champ numérique."""
    path = RUNS_DIR / series
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "train.jsonl", "a") as f:
        f.write(json.dumps({"date": datetime.now(timezone.utc).isoformat(), **values}) + "\n")


def noop_actions(n: int) -> np.ndarray:
    a = np.zeros((n, 5), dtype=np.int64)
    a[:, 0] = NOOP
    return a


__all__ = ["ServerPool", "VecOpenFrontEnv", "performance_cores", "noop_actions", "log_train", "ACTION_TYPES", "AMOUNTS", "BUILD_UNITS"]
