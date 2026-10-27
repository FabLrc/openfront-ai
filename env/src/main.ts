/**
 * Point d'entrée : serveur d'environnement (TCP) + page /live (HTTP/WebSocket).
 *
 *   npx tsx src/main.ts [--port 5555] [--http-port 8080]
 */
import { parseArgs } from "node:util";
import { LiveHub } from "./live";
import { startEnvServer } from "./server";

const { values } = parseArgs({
  options: {
    port: { type: "string", default: "5555" },
    "http-port": { type: "string", default: "8080" },
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
hub.listen(httpPort);
startEnvServer(port, hub);
log(`Environnement : tcp://127.0.0.1:${port}`);
log(`Partie en direct : http://localhost:${httpPort}/live`);
