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
          // liveSpeed : vitesse de /live imposée par le client (scripts/watch.sh).
          const { liveSpeed, ...config } = (req.config ?? {}) as EnvConfig & { liveSpeed?: string };
          if (liveSpeed !== undefined) hub.setSpeed(String(liveSpeed));
          env = await AgentEnv.create(config);
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
          const actions = (req.actions ?? {}) as Record<string, Action>;
          const annotations = (req.annotations ?? {}) as Record<string, unknown>;
          // Vitesse réglée depuis la page /live : on joue tick par tick en
          // attendant entre deux, pour que la page voie chaque tick.
          const speed = hub.speed();
          let waited = 0; // temps passé à attendre, exclu de server_ms
          let done = 0;
          const pause = async () => {
            done++;
            const s = hub.speed();
            if (s === null) return;
            const wait = start + (done * MS_PER_TICK) / s - performance.now();
            if (wait <= 0) return;
            await new Promise((r) => setTimeout(r, wait));
            waited += wait;
          };
          const { buffers, ...rest } = speed === null
            ? env.step(actions, ticks, annotations)
            : await env.stepPaced(actions, ticks, annotations, pause);
          const computeMs = performance.now() - start - waited;
          hub.recordTicks(rest.state.tick - before);
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
