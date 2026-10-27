/**
 * Serveur d'environnement : une partie par connexion TCP.
 *
 * Requêtes (une trame JSON chacune, réponse dans le même ordre) :
 *   {"cmd": "reset", "config": {map, size, nations, difficulty, seed}}
 *   {"cmd": "step", "ticks": 10}
 *   {"cmd": "close"}
 * Réponses : {"ok": true, "state": GameState} ou {"ok": false, "error": "..."}.
 */
import net from "node:net";
import { GameState, OpenFrontGame, ResetConfig } from "./game";
import { FrameReader, sendFrame } from "./protocol";
import { LiveHub } from "./live";

type Request =
  | { cmd: "reset"; config?: ResetConfig }
  | { cmd: "step"; ticks?: number }
  | { cmd: "close" };

const MS_PER_TICK = 100; // durée d'un tick en temps réel (EngineConfig.msPerTick)

export function startEnvServer(port: number, hub: LiveHub): net.Server {
  const server = net.createServer((socket) => {
    socket.setNoDelay(true);
    const reader = new FrameReader();
    let game: OpenFrontGame | null = null;
    // Les requêtes sont traitées une par une, même si plusieurs arrivent d'un coup.
    let queue = Promise.resolve();

    const handle = async (req: Request): Promise<GameState | null> => {
      switch (req.cmd) {
        case "reset":
          if (game !== null) hub.detach(game);
          game = await OpenFrontGame.create(req.config ?? {});
          hub.attach(game);
          return game.state();
        case "step": {
          if (game === null) throw new Error("step avant reset");
          const ticks = Math.max(1, Math.floor(req.ticks ?? 10));
          const start = performance.now();
          const before = game.game.ticks();
          const state = game.step(ticks);
          hub.recordTicks(state.tick - before);
          // Vitesse réglée depuis la page /live : on ralentit la réponse.
          const speed = hub.speed();
          if (speed !== null) {
            const target = ((state.tick - before) * MS_PER_TICK) / speed;
            const wait = target - (performance.now() - start);
            if (wait > 0) await new Promise((r) => setTimeout(r, wait));
          }
          return state;
        }
        case "close":
          socket.end();
          return null;
        default:
          throw new Error(`commande inconnue : ${JSON.stringify(req)}`);
      }
    };

    socket.on("data", (chunk) => {
      let requests: unknown[];
      try {
        requests = reader.push(chunk);
      } catch (err) {
        sendFrame(socket, { ok: false, error: String(err) });
        socket.destroy();
        return;
      }
      for (const req of requests) {
        queue = queue.then(async () => {
          try {
            const state = await handle(req as Request);
            if (!socket.destroyed && state !== null) sendFrame(socket, { ok: true, state });
          } catch (err) {
            if (!socket.destroyed) {
              sendFrame(socket, { ok: false, error: err instanceof Error ? err.message : String(err) });
            }
          }
        });
      }
    });
    socket.on("close", () => {
      if (game !== null) hub.detach(game);
    });
    socket.on("error", () => {});
  });
  server.listen(port, "127.0.0.1");
  return server;
}
