/**
 * Point d'entrée : serveur d'environnement (TCP), pages /live et /games
 * (HTTP/WebSocket) et API des replays pour le lecteur officiel.
 *
 *   npx tsx src/main.ts [--port 5555] [--http-port 8080] [--api-port 8787]
 *                       [--viewer-url http://localhost:9000]
 */
import { parseArgs } from "node:util";
import { renderGamesPage } from "./gamesPage";
import { LiveHub } from "./live";
import { DEFAULT_API_PORT, startReplayApi } from "./replayApi";
import { startEnvServer } from "./server";

const { values } = parseArgs({
  options: {
    port: { type: "string", default: "5555" },
    "http-port": { type: "string", default: "8080" },
    "api-port": { type: "string", default: String(DEFAULT_API_PORT) },
    "viewer-url": { type: "string", default: "http://localhost:9000" },
    verbose: { type: "boolean", default: false },
  },
});

const log = console.log;
if (!values.verbose) {
  // Le moteur journalise beaucoup (constructeur, victoires, nations…).
  console.log = console.debug = console.info = () => {};
}

const port = parseInt(values.port, 10);
const httpPort = parseInt(values["http-port"], 10);
const hub = new LiveHub();
hub.listen(httpPort, { "/games": () => renderGamesPage(values["viewer-url"]) });
const apiPort = parseInt(values["api-port"], 10);
startReplayApi(apiPort, values.verbose ? (m) => log(`[api] ${m}`) : undefined);
startEnvServer(port, hub);
log(`Environnement : tcp://127.0.0.1:${port}`);
log(`Partie en direct : http://localhost:${httpPort}/live`);
log(`Parties enregistrées : http://localhost:${httpPort}/games`);
log(`API des replays : http://localhost:${apiPort}/game/<id>`);
