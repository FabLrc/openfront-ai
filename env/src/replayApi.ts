/**
 * API minimale pour le lecteur de replays officiel d'OpenFront.
 *
 * En développement, le client lit `${getApiBase()}/game/<id>` ; sur localhost,
 * sans API_DOMAIN, getApiBase() vaut http://localhost:8787. On y répond avec
 * nos records (runs/), sans toucher au sous-module. Toute autre route de
 * l'API (compte, boutique…) répond 404 : le client s'en passe.
 */
import fs from "node:fs";
import http from "node:http";
import { findRecordFile } from "./records";

export const DEFAULT_API_PORT = 8787;

export function startReplayApi(port: number, log: (msg: string) => void = () => {}): http.Server {
  const server = http.createServer((req, res) => {
    // Le client tourne sur un autre port (vite, 9000) : requêtes cross-origin.
    res.setHeader("Access-Control-Allow-Origin", req.headers.origin ?? "*");
    res.setHeader("Access-Control-Allow-Credentials", "true");
    res.setHeader("Access-Control-Allow-Headers", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
    if (req.method === "OPTIONS") {
      res.writeHead(204).end();
      return;
    }
    const url = new URL(req.url ?? "/", "http://localhost");
    const match = /^\/game\/([^/]+)$/.exec(url.pathname);
    if (req.method === "GET" && match) {
      const file = findRecordFile(decodeURIComponent(match[1]));
      if (file === null) {
        log(`404 ${url.pathname}`);
        res.writeHead(404, { "content-type": "application/json" }).end(JSON.stringify({ error: "not found" }));
        return;
      }
      log(`200 ${url.pathname}`);
      const headers: http.OutgoingHttpHeaders = { "content-type": "application/json" };
      // Les records sont stockés compressés : le navigateur décompresse.
      if (file.endsWith(".gz")) headers["content-encoding"] = "gzip";
      res.writeHead(200, headers);
      fs.createReadStream(file).pipe(res);
      return;
    }
    log(`404 ${req.method} ${url.pathname}`);
    res.writeHead(404, { "content-type": "application/json" }).end(JSON.stringify({ error: "not found" }));
  });
  server.listen(port, "127.0.0.1");
  return server;
}
