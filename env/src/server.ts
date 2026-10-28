/**
 * Serveur d'environnement : une partie par connexion TCP (voir protocol.ts).
 *
 * Requêtes (une trame chacune, réponse dans le même ordre) :
 *   {"cmd": "reset", "config": EnvConfig}
 *   {"cmd": "step", "ticks": 10, "actions": {"agent_0": [type, cible, case, quantité, unité]},
 *    "annotations": {"agent_0": {...}}}   // facultatif, affiché sur /live
 *   {"cmd": "close"}
 * Réponse : {"ok": true, "state", "agents", "rewards", "terminations",
 * "truncations", "infos"} (+ "spec" au reset) et les observations en tableaux
 * binaires nommés "<agent>/<champ>" ; ou {"ok": false, "error": "..."}.
 */
import net from "node:net";
import { AgentEnv, EnvConfig } from "./agentEnv";
import { LiveHub } from "./live";
import { Frame, FrameReader, sendFrame } from "./protocol";
import { Action } from "./seat";

const MS_PER_TICK = 100; // durée d'un tick en temps réel (EngineConfig.msPerTick)

export function startEnvServer(port: number, hub: LiveHub): net.Server {
  const server = net.createServer((socket) => {
    socket.setNoDelay(true);
    const reader = new FrameReader();
    let env: AgentEnv | null = null;
    // Les requêtes sont traitées une par une, même si plusieurs arrivent d'un coup.
    let queue = Promise.resolve();

    const handle = async ({ header: req }: Frame): Promise<void> => {
      switch (req.cmd) {
        case "reset": {
          if (env !== null) {
            env.finish();
            hub.detach(env.game);
          }
          env = await AgentEnv.create((req.config ?? {}) as EnvConfig);
          const current = env;
          hub.attach(env.game, () => current.displays());
          const { buffers, ...rest } = env.reset();
          sendFrame(socket, { ok: true, spec: env.spec(), ...rest }, buffers);
          return;
        }
        case "step": {
          if (env === null) throw new Error("step avant reset");
          const ticks = Math.max(1, Math.floor((req.ticks as number | undefined) ?? 10));
          const start = performance.now();
          const before = env.game.game.ticks();
          const { buffers, ...rest } = env.step(
            (req.actions ?? {}) as Record<string, Action>,
            ticks,
            (req.annotations ?? {}) as Record<string, unknown>,
          );
          const computeMs = performance.now() - start;
          hub.recordTicks(rest.state.tick - before);
          // Vitesse réglée depuis la page /live : on ralentit la réponse.
          const speed = hub.speed();
          if (speed !== null) {
            const target = ((rest.state.tick - before) * MS_PER_TICK) / speed;
            const wait = target - (performance.now() - start);
            if (wait > 0) await new Promise((r) => setTimeout(r, wait));
          }
          // Temps de calcul du pas (hors attente pour la vitesse de /live).
          if (!socket.destroyed) sendFrame(socket, { ok: true, server_ms: computeMs, ...rest }, buffers);
          return;
        }
        case "close":
          socket.end();
          return;
        default:
          throw new Error(`commande inconnue : ${JSON.stringify(req)}`);
      }
    };

    socket.on("data", (chunk) => {
      let frames: Frame[];
      try {
        frames = reader.push(chunk);
      } catch (err) {
        sendFrame(socket, { ok: false, error: String(err) });
        socket.destroy();
        return;
      }
      for (const frame of frames) {
        queue = queue.then(async () => {
          try {
            await handle(frame);
          } catch (err) {
            if (!socket.destroyed) {
              sendFrame(socket, { ok: false, error: err instanceof Error ? err.message : String(err) });
            }
          }
        });
      }
    });
    socket.on("close", () => {
      if (env === null) return;
      // Partie abandonnée en cours : on l'enregistre telle quelle.
      queue = queue.then(() => {
        env?.finish();
      });
      hub.detach(env.game);
    });
    socket.on("error", () => {});
  });
  server.listen(port, "127.0.0.1");
  return server;
}
