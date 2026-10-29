/**
 * Page « partie en direct » : http://localhost:<port>/live.
 *
 * Le serveur pousse via WebSocket (/ws), environ 5 fois par seconde, le
 * propriétaire de chaque tuile d'une carte sous-échantillonnée (Uint16Array
 * compressé en deflate puis base64) et la liste des joueurs. La page renvoie
 * le réglage de vitesse ({"type": "speed", "value": "1" | "2" | "4" | "10" | "max"}).
 */
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";
import { WebSocket, WebSocketServer } from "ws";
import { UnitType } from "@openfront/engine-api/game/GameTypes";
import { OpenFrontGame } from "./game";
import { SeatDisplay } from "./seat";

const VIEWER_HTML = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../viewer/index.html",
);
const BROADCAST_MS = 200;
const MAX_VIEW_DIM = 1000; // plus grande dimension de la carte envoyée
const OWNER_MASK = 0x0fff; // bits 0-11 de l'état d'une tuile
const FALLOUT_BIT = 1 << 13; // retombées nucléaires
/** Bit ajouté au propriétaire envoyé à la page pour une tuile irradiée. */
const VIEW_FALLOUT = 0x8000;

/**
 * Unités affichées sur /live, dans cet ordre (même liste dans viewer/index.html).
 * Chaque unité est envoyée comme [id, type, propriétaire, x, y, cibleX, cibleY,
 * en construction], coordonnées dans la carte réduite.
 */
export const VIEW_UNITS = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.TransportShip,
  UnitType.Warship,
  UnitType.TradeShip,
  UnitType.Train,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
  UnitType.SAMMissile,
];
type ViewUnit = number[];

const SPEEDS = ["1", "2", "4", "10", "max"] as const;
type Speed = (typeof SPEEDS)[number];

/** Couleur stable par smallID (angle d'or sur la teinte). */
export function playerColor(id: number): string {
  const h = (id * 137.508) % 360;
  const s = 55 + (id % 3) * 12;
  const l = 48 + (id % 2) * 10;
  return `hsl(${h.toFixed(0)} ${s}% ${l}%)`;
}

export class LiveHub {
  private current: OpenFrontGame | null = null;
  private displays: () => SeatDisplay[] = () => [];
  private ended = false;
  private speedSetting: Speed = "max";
  private clients = new Set<WebSocket>();
  /** Clients qui n'ont pas encore reçu la liste complète des unités. */
  private needFull = new Set<WebSocket>();
  /** Unités du dernier envoi (id → données), pour n'envoyer ensuite que les changements. */
  private lastUnits = new Map<number, string>();
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

  /** Règle la vitesse (valeur inconnue ignorée). */
  setSpeed(value: string): void {
    if (!(SPEEDS as readonly string[]).includes(value)) return;
    this.speedSetting = value as Speed;
    this.broadcast([...this.clients]);
  }

  attach(game: OpenFrontGame, displays: () => SeatDisplay[] = () => []): void {
    this.current = game;
    this.lastUnits.clear();
    for (const ws of this.clients) this.needFull.add(ws);
    this.displays = displays;
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
    const now = performance.now();
    this.tickLog.push({ t: now, n });
    // Purge ici aussi : sans spectateur, ticksPerSecond() n'est jamais appelé
    // et le journal grossissait sans fin.
    if (this.tickLog.length > 1000) this.dropOld(now);
  }

  private dropOld(now: number): void {
    let i = 0;
    while (i < this.tickLog.length && now - this.tickLog[i].t >= 1000) i++;
    if (i > 0) this.tickLog.splice(0, i);
  }

  private ticksPerSecond(): number {
    this.dropOld(performance.now());
    return this.tickLog.reduce((a, e) => a + e.n, 0);
  }

  private viewUnits(): Map<number, ViewUnit> {
    const map = this.current!.game.map();
    const w = map.width();
    const st = this.stride;
    const out = new Map<number, ViewUnit>();
    for (const u of this.current!.game.units(VIEW_UNITS)) {
      if (!u.isActive()) continue;
      const t = u.tile();
      const target = u.targetTile();
      out.set(u.id(), [
        u.id(),
        VIEW_UNITS.indexOf(u.type()),
        u.owner().smallID(),
        Math.round((10 * (t % w)) / st) / 10,
        Math.round((10 * Math.floor(t / w)) / st) / 10,
        target === undefined ? -1 : Math.round((10 * (target % w)) / st) / 10,
        target === undefined ? -1 : Math.round((10 * Math.floor(target / w)) / st) / 10,
        u.isUnderConstruction() ? 1 : 0,
      ]);
    }
    return out;
  }

  /**
   * Image de la partie. Les unités partent en différences depuis l'envoi
   * précédent (ajoutées ou modifiées, retirées), sauf `full` : liste complète,
   * pour un nouveau client ou une nouvelle partie.
   */
  private frameMessage(units: Map<number, ViewUnit>, full: boolean): string {
    const game = this.current!;
    const map = game.game.map();
    const buf = map.tileStateBuffer();
    const w = map.width();
    const owners = new Uint16Array(this.viewW * this.viewH);
    for (let y = 0; y < this.viewH; y++) {
      const row = y * this.stride * w;
      for (let x = 0; x < this.viewW; x++) {
        const s = buf[row + x * this.stride];
        owners[y * this.viewW + x] = (s & OWNER_MASK) | (s & FALLOUT_BIT ? VIEW_FALLOUT : 0);
      }
    }
    const changed: ViewUnit[] = [];
    const removed: number[] = [];
    if (full) {
      changed.push(...units.values());
    } else {
      for (const [id, u] of units) if (this.lastUnits.get(id) !== JSON.stringify(u)) changed.push(u);
      for (const id of this.lastUnits.keys()) if (!units.has(id)) removed.push(id);
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
      // Sièges d'agents : mis en évidence sur la carte, avec leur dernière action.
      agents: this.displays(),
      owners: zlib
        .deflateSync(Buffer.from(owners.buffer, owners.byteOffset, owners.byteLength))
        .toString("base64"),
      units: { full, changed, removed },
    });
  }


  private broadcast(force: WebSocket[] = []): void {
    if (this.current === null) return;
    const tick = this.current.game.ticks();
    const targets = tick !== this.lastTick ? [...this.clients] : force;
    if (targets.length === 0) return;
    this.lastTick = tick;
    const units = this.viewUnits();
    const fresh = targets.filter((ws) => this.needFull.has(ws));
    const others = targets.filter((ws) => !this.needFull.has(ws));
    if (fresh.length > 0) {
      const msg = this.frameMessage(units, true);
      for (const ws of fresh) {
        ws.send(msg);
        this.needFull.delete(ws);
      }
    }
    if (others.length > 0) {
      const msg = this.frameMessage(units, false);
      for (const ws of others) ws.send(msg);
    }
    // Référence des prochaines différences. Les unités ne changent qu'avec le
    // tick, et tous les clients sont servis quand il change.
    this.lastUnits = new Map([...units].map(([id, u]) => [id, JSON.stringify(u)]));
  }


  /** `pages` : chemin → HTML (chaîne) ou données renvoyées en JSON ({ json }). */
  listen(port: number, pages: Record<string, (url: URL) => string | { json: unknown }> = {}): http.Server {
    const server = http.createServer((req, res) => {
      const url = new URL(req.url ?? "/", "http://localhost");
      const page = pages[url.pathname];
      if (page !== undefined) {
        const out = page(url);
        if (typeof out === "string") {
          res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
          res.end(out);
        } else {
          res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
          res.end(JSON.stringify(out.json));
        }
        return;
      }
      if (url.pathname === "/live") {
        res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        res.end(fs.readFileSync(VIEWER_HTML));
        return;
      }
      res.writeHead(404).end("introuvable");
    });
    const wss = new WebSocketServer({ server, path: "/ws" });
    wss.on("connection", (ws) => {
      this.clients.add(ws);
      this.needFull.add(ws);
      if (this.mapMessage !== null) ws.send(this.mapMessage);
      this.broadcast([ws]);
      ws.on("message", (data) => {
        try {
          const msg = JSON.parse(data.toString());
          if (msg.type === "speed") this.setSpeed(String(msg.value));
        } catch {
          // message ignoré
        }
      });
      ws.on("close", () => {
        this.clients.delete(ws);
        this.needFull.delete(ws);
      });
    });
    setInterval(() => this.broadcast(), BROADCAST_MS).unref();
    server.listen(port, "127.0.0.1");
    return server;
  }
}
