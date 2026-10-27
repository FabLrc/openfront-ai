/**
 * Page « partie en direct » : http://localhost:<port>/live.
 *
 * Le serveur pousse via WebSocket (/ws), environ 5 fois par seconde, le
 * propriétaire de chaque tuile d'une carte sous-échantillonnée (Uint16Array
 * compressé en deflate puis base64) et la liste des joueurs. La page renvoie
 * le réglage de vitesse ({"type": "speed", "value": "1" | "10" | "max"}).
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { WebSocket, WebSocketServer } from "ws";
import { OpenFrontGame } from "./game";

const VIEWER_HTML = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../viewer/index.html",
);
const BROADCAST_MS = 200;
const MAX_VIEW_DIM = 1000; // plus grande dimension de la carte envoyée
const OWNER_MASK = 0x0fff; // bits 0-11 de l'état d'une tuile

type Speed = "1" | "10" | "max";

/** Couleur stable par smallID (angle d'or sur la teinte). */
export function playerColor(id: number): string {
  const h = (id * 137.508) % 360;
  const s = 55 + (id % 3) * 12;
  const l = 48 + (id % 2) * 10;
  return `hsl(${h.toFixed(0)} ${s}% ${l}%)`;
}

export class LiveHub {
  private current: OpenFrontGame | null = null;
  private ended = false;
  private speedSetting: Speed = "max";
  private clients = new Set<WebSocket>();
  private mapMessage: string | null = null;
  private lastTick = -1;
  private tickLog: { t: number; n: number }[] = [];
  private stride = 1;
  private viewW = 0;
  private viewH = 0;

  /** Vitesse en multiple du temps réel, null pour « max ». */
  speed(): number | null {
    return this.speedSetting === "max" ? null : Number(this.speedSetting);
  }

  attach(game: OpenFrontGame): void {
    this.current = game;
    this.ended = false;
    this.lastTick = -1;
    const map = game.game.map();
    const w = map.width();
    const h = map.height();
    this.stride = Math.max(1, Math.ceil(Math.max(w, h) / MAX_VIEW_DIM));
    this.viewW = Math.ceil(w / this.stride);
    this.viewH = Math.ceil(h / this.stride);
    const land = new Uint8Array(this.viewW * this.viewH);
    for (let y = 0; y < this.viewH; y++) {
      for (let x = 0; x < this.viewW; x++) {
        land[y * this.viewW + x] = map.isLand(map.ref(x * this.stride, y * this.stride)) ? 1 : 0;
      }
    }
    this.mapMessage = JSON.stringify({
      type: "map",
      map: game.config.map,
      size: game.config.size,
      width: this.viewW,
      height: this.viewH,
      stride: this.stride,
      land: zlib.deflateSync(land).toString("base64"),
    });
    for (const ws of this.clients) ws.send(this.mapMessage);
  }

  detach(game: OpenFrontGame): void {
    // On garde la dernière image affichée.
    if (this.current === game) this.ended = true;
  }

  recordTicks(n: number): void {
    this.tickLog.push({ t: performance.now(), n });
  }

  private ticksPerSecond(): number {
    const now = performance.now();
    this.tickLog = this.tickLog.filter((e) => now - e.t < 1000);
    return this.tickLog.reduce((a, e) => a + e.n, 0);
  }

  private frameMessage(): string {
    const game = this.current!;
    const map = game.game.map();
    const buf = map.tileStateBuffer();
    const w = map.width();
    const owners = new Uint16Array(this.viewW * this.viewH);
    for (let y = 0; y < this.viewH; y++) {
      const row = y * this.stride * w;
      for (let x = 0; x < this.viewW; x++) {
        owners[y * this.viewW + x] = buf[row + x * this.stride] & OWNER_MASK;
      }
    }
    const state = game.state();
    return JSON.stringify({
      type: "frame",
      tick: state.tick,
      ticksPerSecond: this.ticksPerSecond(),
      speed: this.speedSetting,
      ended: this.ended,
      winner: state.winner,
      players: state.players.map((p) => ({ ...p, color: playerColor(p.id) })),
      owners: zlib
        .deflateSync(Buffer.from(owners.buffer, owners.byteOffset, owners.byteLength))
        .toString("base64"),
    });
  }

  private broadcast(force: WebSocket[] = []): void {
    if (this.current === null) return;
    const tick = this.current.game.ticks();
    const targets = tick !== this.lastTick ? [...this.clients] : force;
    if (targets.length === 0) return;
    this.lastTick = tick;
    const msg = this.frameMessage();
    for (const ws of targets) ws.send(msg);
  }

  listen(port: number): http.Server {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://localhost");
      if (url.pathname === "/" || url.pathname === "/live") {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(fs.readFileSync(VIEWER_HTML));
        return;
      }
      res.writeHead(404).end("introuvable");
    });
    const wss = new WebSocketServer({ server, path: "/ws" });
    wss.on("connection", (ws) => {
      this.clients.add(ws);
      if (this.mapMessage !== null) ws.send(this.mapMessage);
      this.broadcast([ws]);
      ws.on("message", (data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === "speed" && ["1", "10", "max"].includes(msg.value)) {
            this.speedSetting = msg.value;
            this.broadcast([...this.clients]);
          }
        } catch {
          // message ignoré
        }
      });
      ws.on("close", () => this.clients.delete(ws));
    });
    setInterval(() => this.broadcast(), BROADCAST_MS).unref();
    server.listen(port, "127.0.0.1");
    return server;
  }
}
