/**
 * Siège d'agent : observation binaire, masques d'actions valides et
 * traduction des actions factorisées en intentions du moteur.
 *
 * Action = [type, cible, case, quantité, unité] (MultiDiscrete) :
 *   type     : ACTION_TYPES
 *   cible    : 0 = terres neutres / 1..K = ligne de la table des joueurs ;
 *              pendant la phase de spawn, index du candidat de spawn
 *   case     : case de la grille d'actions (grossière), pour bateau et construction
 *   quantité : part des troupes envoyée (AMOUNTS)
 *   unité    : BUILD_UNITS
 *
 * Les masques reproduisent les contrôles que font les exécutions du moteur
 * (AttackExecution, TransportShipExecution, ConstructionExecution,
 * AllianceRequestExecution, BreakAllianceExecution, SpawnExecution) au
 * moment où l'intention est exécutée, c'est-à-dire au tick suivant.
 */
import { TileRef } from "@openfront/engine-api/game/GameMap";
import { PlayerType, Structures, UnitType } from "@openfront/engine-api/game/GameTypes";
import { Intent, StampedIntent } from "@openfront/engine-api/Schemas";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { getSpawnTiles } from "@openfront/engine/execution/Util";
import { Execution, Game, Player } from "@openfront/engine/game/Game";
import { targetTransportTile } from "@openfront/engine/game/TransportShipUtils";
import { Buffers } from "./protocol";

export const ACTION_TYPES = ["noop", "spawn", "attack", "boat", "build", "alliance", "break"] as const;
export const T_NOOP = 0;
export const T_SPAWN = 1;
export const T_ATTACK = 2;
export const T_BOAT = 3;
export const T_BUILD = 4;
export const T_ALLIANCE = 5;
export const T_BREAK = 6;
export const AMOUNTS = [0.1, 0.25, 0.5, 0.75];
export const BUILD_UNITS = [
  UnitType.City,
  UnitType.Port,
  UnitType.Factory,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
] as const;
export const CHANNELS = [
  "me",
  "allies",
  "enemies",
  "neutral",
  "water",
  "front",
  "fallout",
  ...BUILD_UNITS.map((u) => `structure:${u}`),
];
export const PLAYER_FEATURES = [
  "present",
  "neutral",
  "alive",
  "tile_share",
  "troops",
  "gold",
  "troops_vs_me",
  "relation",
  "allied",
  "border",
  "nearby",
  "incoming_attack",
  "outgoing_attack",
  "alliance_request",
  "nation",
  "human",
];
export const SCALARS = [
  "troops",
  "max_troops",
  "troops_ratio",
  "gold",
  "tile_share",
  "tick",
  "win_threshold",
  "alive_share",
  "spawn_phase",
  "boats",
  "cities",
  "outgoing_attacks",
  "incoming_attacks",
  "alive",
];

/** En dessous, attaques et bateaux sont masqués (quantité nulle). */
const MIN_TROOPS = 10;
/** Contacts minimaux (tuile frontalière × voisine) pour autoriser une attaque. */
const MIN_FRONT = 4;
const FALLOUT_BIT = 1 << 13;
const OWNER_MASK = 0x0fff;

export interface SeatOptions {
  obsWidth: number;
  obsHeight: number;
  actionWidth: number;
  actionHeight: number;
  /** Nombre de joueurs dans la table (hors ligne « terres neutres »). */
  k: number;
  /** Nombre maximal de cases proposées pour les bateaux (les plus proches). */
  boatCandidates: number;
  /** Tests exacts (recherche de chemin sur l'eau) au plus par pas ; les autres viennent du cache. */
  boatChecksPerStep: number;
  /** Durée de vie, en pas, d'un résultat de test exact en cache. */
  boatCacheSteps: number;
  /** Cases de construction recalculées au plus par pas, et durée de vie du cache. */
  buildChecksPerStep: number;
  buildCacheSteps: number;
}

export const DEFAULT_SEAT_OPTIONS: SeatOptions = {
  obsWidth: 128,
  obsHeight: 64,
  actionWidth: 32,
  actionHeight: 16,
  k: 16,
  boatCandidates: 8,
  // Chaque test exact coûte ~0,1 ms (recherches de chemin sur l'eau).
  boatChecksPerStep: 3,
  boatCacheSteps: 5,
  buildChecksPerStep: 24,
  buildCacheSteps: 5,
};

const logNorm = (x: number, scale: number) => Math.log1p(Math.max(0, x)) / Math.log1p(scale);
const clip = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const toByte = (x: number) => Math.min(255, Math.round(x * 255));

/** Correspondances tuile → case, calculées une fois par partie. */
export class MapIndex {
  readonly width: number;
  readonly height: number;
  readonly obsCells: number;
  readonly actCells: number;
  readonly tileObsCell: Int32Array;
  readonly tileActCell: Int32Array;
  readonly tilesPerObsCell: Uint16Array;
  readonly waterPerObsCell: Uint16Array;
  readonly totalLand: number;
  /** Tuiles de côte de chaque case d'action, de la plus centrale à la plus excentrée. */
  readonly shoreByActCell: TileRef[][];
  /** Terres de chaque case d'action, de la plus centrale à la plus excentrée (case c : landOffsets[c]..landOffsets[c+1]). */
  readonly landByActCell: Int32Array;
  readonly landOffsets: Int32Array;
  readonly actCenterX: Float32Array;
  readonly actCenterY: Float32Array;

  constructor(
    readonly game: Game,
    readonly opts: SeatOptions,
  ) {
    const map = game.map();
    const w = (this.width = map.width());
    const h = (this.height = map.height());
    this.obsCells = opts.obsWidth * opts.obsHeight;
    this.actCells = opts.actionWidth * opts.actionHeight;
    this.tileObsCell = new Int32Array(w * h);
    this.tileActCell = new Int32Array(w * h);
    this.tilesPerObsCell = new Uint16Array(this.obsCells);
    this.waterPerObsCell = new Uint16Array(this.obsCells);
    this.actCenterX = new Float32Array(this.actCells);
    this.actCenterY = new Float32Array(this.actCells);
    for (let c = 0; c < this.actCells; c++) {
      this.actCenterX[c] = ((c % opts.actionWidth) + 0.5) * (w / opts.actionWidth);
      this.actCenterY[c] = (Math.floor(c / opts.actionWidth) + 0.5) * (h / opts.actionHeight);
    }
    const shore: TileRef[][] = Array.from({ length: this.actCells }, () => []);
    let land = 0;
    for (let y = 0; y < h; y++) {
      const oy = Math.floor((y * opts.obsHeight) / h) * opts.obsWidth;
      const ay = Math.floor((y * opts.actionHeight) / h) * opts.actionWidth;
      for (let x = 0; x < w; x++) {
        const t = y * w + x;
        const oc = oy + Math.floor((x * opts.obsWidth) / w);
        const ac = ay + Math.floor((x * opts.actionWidth) / w);
        this.tileObsCell[t] = oc;
        this.tileActCell[t] = ac;
        this.tilesPerObsCell[oc]++;
        if (map.isLand(t)) {
          land++;
          if (map.isShore(t) && !map.isImpassable(t)) shore[ac].push(t);
        } else {
          this.waterPerObsCell[oc]++;
        }
      }
    }
    this.totalLand = land;
    const byCell: TileRef[][] = Array.from({ length: this.actCells }, () => []);
    for (let t = 0; t < w * h; t++) if (map.isLand(t) && !map.isImpassable(t)) byCell[this.tileActCell[t]].push(t);
    this.landOffsets = new Int32Array(this.actCells + 1);
    this.landByActCell = new Int32Array(byCell.reduce((n, l) => n + l.length, 0));
    for (let c = 0; c < this.actCells; c++) {
      const cx = this.actCenterX[c];
      const cy = this.actCenterY[c];
      const d = (t: TileRef) => ((t % w) - cx) ** 2 + (Math.floor(t / w) - cy) ** 2;
      shore[c].sort((a, b) => d(a) - d(b));
      byCell[c].sort((a, b) => d(a) - d(b));
      this.landByActCell.set(byCell[c], this.landOffsets[c]);
      this.landOffsets[c + 1] = this.landOffsets[c] + byCell[c].length;
    }
    this.shoreByActCell = shore;
  }
}

/** Comptes par case partagés par tous les sièges pour un même tick. */
export class SharedFrame {
  readonly stride: number;
  readonly ownerCounts: Uint16Array;
  /** Tuiles de chaque joueur par case d'action (pour sauter les cases vides). */
  readonly actOwnerCounts: Uint16Array;
  readonly falloutCounts: Uint16Array;
  readonly structures: { cell: number; unit: number; owner: number }[] = [];

  constructor(index: MapIndex) {
    const game = index.game;
    let maxID = 0;
    for (const p of game.allPlayers()) maxID = Math.max(maxID, p.smallID());
    this.stride = maxID + 1;
    this.ownerCounts = new Uint16Array(index.obsCells * this.stride);
    this.actOwnerCounts = new Uint16Array(index.actCells * this.stride);
    this.falloutCounts = new Uint16Array(index.obsCells);
    const buf = game.map().tileStateBuffer();
    const cells = index.tileObsCell;
    const actCells = index.tileActCell;
    const stride = this.stride;
    for (let t = 0; t < buf.length; t++) {
      const s = buf[t];
      const c = cells[t];
      const o = s & OWNER_MASK;
      this.ownerCounts[c * stride + o]++;
      if (o !== 0) this.actOwnerCounts[actCells[t] * stride + o]++;
      if (s & FALLOUT_BIT) this.falloutCounts[c]++;
    }
    for (const u of game.units(BUILD_UNITS)) {
      this.structures.push({
        cell: cells[u.tile()],
        unit: BUILD_UNITS.indexOf(u.type() as (typeof BUILD_UNITS)[number]),
        owner: u.owner().smallID(),
      });
    }
  }
}

export type Action = number[];

interface Pending {
  type: number;
  exec?: Execution;
  accepted?: boolean;
  reason?: string;
  target?: Player | null;
  tile?: TileRef;
  description: string;
}

/** Ce que la page /live affiche pour un siège. */
export interface SeatDisplay {
  id: number;
  name: string;
  lastAction: string;
  accepted: boolean | null;
  validCounts: Record<string, number>;
  annotations?: unknown;
}

export class AgentSeat {
  readonly k: number;
  /** Ligne i de la table : null = vide, "neutral" = terres neutres (ligne 0). */
  table: (Player | "neutral" | null)[];
  spawnCandidates: TileRef[] = [];
  readonly typeMask = new Uint8Array(ACTION_TYPES.length);
  readonly targetMask: Uint8Array;
  readonly cellMask: Uint8Array;
  readonly unitMask = new Uint8Array(BUILD_UNITS.length);
  readonly unitCellMask: Uint8Array;
  readonly amountMask = new Uint8Array(AMOUNTS.length);
  private boatTile: Int32Array;
  private landBuildTile: Int32Array;
  private portBuildTile: Int32Array;
  private pending: Pending | null = null;
  private courted = new Set<Player>();
  /** Pas courant (incrémenté à chaque calcul des masques). */
  private maskStep = 0;
  /** Résultat du test exact de bateau par case : tuile visée, verdict, pas du test. */
  private boatCache = new Map<number, { tile: TileRef; ok: boolean; at: number }>();
  /** Candidats et verdicts de construction par case. */
  private buildCache = new Map<
    number,
    { land: TileRef; landOk: boolean; port: TileRef; portOk: boolean; at: number }
  >();
  /** Vrai si l'action précédente a été annulée car son masque (en cache) était périmé. */
  lastStale = false;
  lastAction = "—";
  lastAccepted: boolean | null = null;
  lastRejectReason: string | null = null;
  annotations: unknown = undefined;

  constructor(
    readonly index: number,
    readonly name: string,
    readonly player: Player,
    readonly map: MapIndex,
    seed: string,
  ) {
    this.k = map.opts.k;
    const rows = this.k + 1;
    this.table = new Array(rows).fill(null);
    this.targetMask = new Uint8Array(ACTION_TYPES.length * rows);
    this.cellMask = new Uint8Array(ACTION_TYPES.length * map.actCells);
    this.unitCellMask = new Uint8Array(BUILD_UNITS.length * map.actCells);
    this.boatTile = new Int32Array(map.actCells).fill(-1);
    this.landBuildTile = new Int32Array(map.actCells).fill(-1);
    this.portBuildTile = new Int32Array(map.actCells).fill(-1);
    this.pickSpawnCandidates(new PseudoRandom(simpleHash(seed) + 7919 * (index + 1)));
  }

  private get game(): Game {
    return this.map.game;
  }

  /** K+1 centres de spawn valides (même critère que SpawnExecution), bien répartis. */
  private pickSpawnCandidates(random: PseudoRandom): void {
    const game = this.game;
    const gm = game.map();
    const w = this.map.width;
    const h = this.map.height;
    let minDist = Math.floor((w + h) / 12);
    for (let tries = 0; this.spawnCandidates.length < this.k + 1 && tries < 20000; tries++) {
      if (tries % 4000 === 3999) minDist = Math.floor(minDist / 2);
      const t = gm.ref(random.nextInt(0, w), random.nextInt(0, h));
      if (!gm.isLand(t) || gm.isImpassable(t) || gm.hasOwner(t) || gm.isBorder(t)) continue;
      if (this.spawnCandidates.some((c) => gm.manhattanDist(c, t) < minDist)) continue;
      if (getSpawnTiles(gm, t, true) === null) continue;
      this.spawnCandidates.push(t);
    }
  }

  alive(): boolean {
    return this.player.isAlive();
  }

  // ── Masques ──

  /** `frame` n'est nécessaire qu'après la phase de spawn (masque de construction). */
  computeMasks(frame?: SharedFrame): void {
    this.maskStep++;
    const game = this.game;
    const p = this.player;
    const rows = this.k + 1;
    const A = this.map.actCells;
    this.typeMask.fill(0);
    this.targetMask.fill(0);
    this.cellMask.fill(0);
    this.unitMask.fill(0);
    this.unitCellMask.fill(0);
    this.amountMask.fill(1);
    this.typeMask[T_NOOP] = 1;

    if (game.inSpawnPhase()) {
      this.table.fill(null);
      const gm = game.map();
      this.spawnCandidates.forEach((t, i) => {
        if (getSpawnTiles(gm, t, false).length > 0) this.targetMask[T_SPAWN * rows + i] = 1;
      });
      this.typeMask[T_SPAWN] = this.targetMask.subarray(T_SPAWN * rows, (T_SPAWN + 1) * rows).some((v) => v)
        ? 1
        : 0;
      return;
    }
    if (!p.isAlive() || game.getWinner() !== null) {
      this.table.fill(null);
      return;
    }

    const attackable = this.attackableOwners();
    this.fillTable(attackable);
    const troopsOk = p.troops() >= MIN_TROOPS;
    // Un joueur à qui j'ai proposé une alliance peut l'accepter pendant le
    // tick qui précède l'exécution de mon intention : il deviendrait allié,
    // donc inattaquable. On ne l'attaque pas tant que la demande est en cours.
    this.courted = new Set(p.outgoingAllianceRequests().map((r) => r.recipient()));

    for (let r = 0; r < rows; r++) {
      const o = this.table[r];
      if (o === null) continue;
      if (o === "neutral") {
        if (troopsOk && attackable.has(0)) this.targetMask[T_ATTACK * rows] = 1;
        continue;
      }
      if (!o.isAlive()) continue;
      if (troopsOk && attackable.has(o.smallID()) && p.canAttackPlayer(o) && !this.courted.has(o)) {
        this.targetMask[T_ATTACK * rows + r] = 1;
      }
      if (p.canSendAllianceRequest(o) && !this.incomingExpiresSoon(o)) this.targetMask[T_ALLIANCE * rows + r] = 1;
      if (p.isAlliedWith(o)) this.targetMask[T_BREAK * rows + r] = 1;
    }

    if (troopsOk && p.unitCount(UnitType.TransportShip) < game.config().boatMaxNumber()) {
      this.computeBoatMask();
    }
    if (frame !== undefined) this.computeBuildMask(frame);

    for (const t of [T_ATTACK, T_ALLIANCE, T_BREAK]) {
      this.typeMask[t] = this.targetMask.subarray(t * rows, (t + 1) * rows).some((v) => v) ? 1 : 0;
    }
    for (const t of [T_BOAT, T_BUILD]) {
      this.typeMask[t] = this.cellMask.subarray(t * A, (t + 1) * A).some((v) => v) ? 1 : 0;
    }
  }

  /**
   * Propriétaires (smallID, 0 = neutre) d'au moins MIN_FRONT tuiles terrestres
   * au contact de ma frontière. Le moteur exige au moins une tuile au moment
   * où il initialise l'attaque, un tick plus tard : la marge évite qu'un front
   * d'une ou deux tuiles disparaisse entre-temps.
   */
  private attackableOwners(): Set<number> {
    const map = this.game.map();
    const me = this.player.smallID();
    const counts = new Map<number, number>();
    const nbuf: TileRef[] = [0, 0, 0, 0];
    this.player.borderTiles().forEach((tile) => {
      const n = map.neighbors4(tile, nbuf);
      for (let i = 0; i < n; i++) {
        const nb = nbuf[i];
        if (map.isWater(nb) || map.isImpassable(nb)) continue;
        const o = map.ownerID(nb);
        if (o !== me) counts.set(o, (counts.get(o) ?? 0) + 1);
      }
    });
    return new Set([...counts].filter(([, n]) => n >= MIN_FRONT).map(([o]) => o));
  }

  /** Ligne 0 : terres neutres ; puis les voisins, puis les plus gros joueurs. */
  private fillTable(attackable: Set<number>): void {
    const p = this.player;
    const nearby = new Set(
      p
        .nearby()
        .filter((o): o is Player => o.isPlayer())
        .map((o) => o.smallID()),
    );
    const others = this.game
      .players()
      .filter((o) => o !== p)
      .sort((a, b) => {
        const na = attackable.has(a.smallID()) || nearby.has(a.smallID()) ? 1 : 0;
        const nb = attackable.has(b.smallID()) || nearby.has(b.smallID()) ? 1 : 0;
        return nb - na || b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID();
      });
    // Les alliés restent dans la table pour pouvoir rompre.
    for (const ally of p.allies()) {
      if (!others.includes(ally) && ally.isAlive()) others.push(ally);
    }
    this.table.fill(null);
    this.table[0] = "neutral";
    for (let i = 0; i < this.k && i < others.length; i++) this.table[i + 1] = others[i];
  }

  private computeBoatMask(): void {
    const game = this.game;
    const p = this.player;
    const A = this.map.actCells;
    const w = this.map.width;
    this.boatTile.fill(-1);
    // Repère : centre de ma plus grande zone, sinon ma tuile de spawn.
    const box = p.largestClusterBoundingBox;
    const spawn = p.spawnTile();
    const ox = box ? (box.min.x + box.max.x) / 2 : spawn !== undefined ? spawn % w : 0;
    const oy = box ? (box.min.y + box.max.y) / 2 : spawn !== undefined ? Math.floor(spawn / w) : 0;
    const gm = game.map();
    const me = p.smallID();
    const h = this.map.height;
    // Marge : une côte à moins de 3 cases de mon territoire peut devenir
    // mienne d'ici l'exécution de l'intention (le moteur la refuserait).
    const nearMine = (t: TileRef) => {
      const x = t % w;
      const y = Math.floor(t / w);
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx >= 0 && ny >= 0 && nx < w && ny < h && gm.ownerID(ny * w + nx) === me) return true;
        }
      }
      return false;
    };
    // Mes bateaux en route peuvent débarquer d'ici là sur la côte visée.
    const landings = p
      .units(UnitType.TransportShip)
      .map((u) => u.targetTile())
      .filter((t): t is TileRef => t !== undefined && t !== null);
    const nearLanding = (t: TileRef) => landings.some((l) => gm.manhattanDist(l, t) <= 12);
    const candidates: { cell: number; tile: TileRef; d: number }[] = [];
    for (let c = 0; c < A; c++) {
      for (const t of this.map.shoreByActCell[c]) {
        if (gm.ownerID(t) === me || nearMine(t) || nearLanding(t)) continue;
        const owner = game.owner(t);
        if (owner.isPlayer() && (!p.canAttackPlayer(owner) || this.courted.has(owner))) continue;
        const d = (this.map.actCenterX[c] - ox) ** 2 + (this.map.actCenterY[c] - oy) ** 2;
        candidates.push({ cell: c, tile: t, d });
        break;
      }
    }
    candidates.sort((a, b) => a.d - b.d || a.cell - b.cell);
    // Les filtres ci-dessus sont refaits à chaque pas ; le test exact (trajet
    // par l'eau) est gardé en cache quelques pas, avec un nombre limité de
    // nouveaux tests par pas. Une case pas encore testée reste masquée.
    const { boatChecksPerStep, boatCacheSteps } = this.map.opts;
    let budget = boatChecksPerStep;
    for (const { cell, tile } of candidates.slice(0, this.map.opts.boatCandidates)) {
      const cached = this.boatCache.get(cell);
      let ok: boolean;
      if (cached !== undefined && cached.tile === tile && this.maskStep - cached.at < boatCacheSteps) {
        ok = cached.ok;
      } else if (budget > 0) {
        budget--;
        ok = this.boatReachable(tile);
        this.boatCache.set(cell, { tile, ok, at: this.maskStep });
      } else {
        continue;
      }
      if (!ok) continue;
      this.boatTile[cell] = tile;
      this.cellMask[T_BOAT * A + cell] = 1;
    }
  }

  /**
   * Une demande d'alliance reçue rend la réponse possible même pendant le délai
   * entre deux demandes ; si elle expire avant l'exécution de l'intention, le
   * moteur refuse. On ne propose pas d'y répondre dans ses 20 derniers ticks.
   */
  private incomingExpiresSoon(o: Player): boolean {
    const g = this.game;
    const req = this.player.incomingAllianceRequests().find((r) => r.requestor() === o);
    return req !== undefined && req.createdAt() + g.config().allianceRequestDuration() - g.ticks() <= 20;
  }

  /** Mêmes contrôles que TransportShipExecution.init (recherches de chemin). */
  private boatReachable(tile: TileRef): boolean {
    const dst = targetTransportTile(this.game, this.player, tile);
    return dst !== null && this.player.canBuild(UnitType.TransportShip, dst) !== false;
  }

  /**
   * Une terre étrangère à 2 cases ou moins : la tuile peut être prise avant
   * que la construction ne s'exécute (deux ticks plus tard). L'eau ne compte
   * pas, une côte reste candidate pour le port.
   */
  private exposed(t: TileRef): boolean {
    const gm = this.game.map();
    const { width: w, height: h } = this.map;
    const me = this.player.smallID();
    const x = t % w;
    const y = Math.floor(t / w);
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const n = ny * w + nx;
        if (gm.isLand(n) && gm.ownerID(n) !== me) return true;
      }
    }
    return false;
  }

  /** canBuild(type, tuile) pour une structure, hors contrôle de l'or (fait à part). */
  private placementOk(port: boolean, t: TileRef): boolean {
    const game = this.game;
    const impl = this.player as unknown as {
      landBasedStructureSpawn(t: TileRef): TileRef | false;
      portSpawn(t: TileRef, valid: TileRef[] | null): TileRef | false;
    };
    if (port) return impl.portSpawn(t, null) !== false;
    // validStructureSpawnTiles(t) contient t lui-même si aucune structure (même
    // en construction) n'est à moins de structureMinDist : une seule requête
    // spatiale suffit alors, sans le parcours de rayon 15 du moteur.
    const minDist = game.config().structureMinDist();
    const itself = !game
      .nearbyUnits(t, minDist, Structures.types, undefined, true)
      .some(({ unit }) => game.euclideanDistSquared(unit.tile(), t) < minDist * minDist);
    return itself || impl.landBasedStructureSpawn(t) !== false;
  }

  /** Construction encore possible maintenant sur cette tuile (revérification avant envoi). */
  private buildStillValid(unit: UnitType, t: TileRef): boolean {
    return (
      t >= 0 &&
      this.game.owner(t) === this.player &&
      !this.exposed(t) &&
      this.placementOk(unit === UnitType.Port, t)
    );
  }

  private computeBuildMask(frame: SharedFrame): void {
    const game = this.game;
    const p = this.player;
    const A = this.map.actCells;
    const affordable = BUILD_UNITS.map(
      (u) => !game.config().isUnitDisabled(u) && p.gold() >= game.unitInfo(u).cost(game, p),
    );
    if (!affordable.some((a) => a)) return;
    const needPort = affordable[BUILD_UNITS.indexOf(UnitType.Port)];
    const needLand = affordable.some((a, i) => a && BUILD_UNITS[i] !== UnitType.Port);

    // Pour chaque case où je possède des terres : la tuile possédée la plus
    // centrale (la côte pour le port) loin de toute terre étrangère, et le
    // verdict de placement du moteur. Ce calcul est gardé en cache quelques
    // pas, avec un nombre limité de cases recalculées par pas ; le choix
    // effectif est revérifié avant l'envoi (buildStillValid).
    const gm = game.map();
    const me = p.smallID();
    const { landByActCell, landOffsets, opts } = this.map;
    let budget = opts.buildChecksPerStep;
    this.landBuildTile.fill(-1);
    this.portBuildTile.fill(-1);
    for (let c = 0; c < A; c++) {
      if (frame.actOwnerCounts[c * frame.stride + me] === 0) {
        this.buildCache.delete(c);
        continue;
      }
      let entry = this.buildCache.get(c);
      const fresh = entry !== undefined && this.maskStep - entry.at < opts.buildCacheSteps;
      if (!fresh) {
        if (budget <= 0) continue;
        budget--;
        let land = -1;
        let port = -1;
        for (let i = landOffsets[c]; i < landOffsets[c + 1]; i++) {
          const t = landByActCell[i];
          if (gm.ownerID(t) === me && !this.exposed(t)) {
            land = t;
            break;
          }
        }
        for (const t of this.map.shoreByActCell[c]) {
          if (gm.ownerID(t) === me && !this.exposed(t)) {
            port = t;
            break;
          }
        }
        entry = {
          land,
          landOk: land >= 0 && this.placementOk(false, land),
          port,
          portOk: port >= 0 && this.placementOk(true, port),
          at: this.maskStep,
        };
        this.buildCache.set(c, entry);
      }
      this.landBuildTile[c] = entry!.land;
      this.portBuildTile[c] = entry!.port;
      BUILD_UNITS.forEach((u, i) => {
        if (!affordable[i]) return;
        const isPort = u === UnitType.Port;
        if (isPort ? needPort && entry!.portOk : needLand && entry!.landOk) {
          this.unitCellMask[i * A + c] = 1;
          this.unitMask[i] = 1;
          this.cellMask[T_BUILD * A + c] = 1;
        }
      });
    }
  }


  // ── Observation ──

  observe(frame: SharedFrame, numSpawned: number): Buffers {
    const game = this.game;
    const p = this.player;
    const { obsWidth: W, obsHeight: H } = this.map.opts;
    const N = W * H;
    const C = CHANNELS.length;
    const spatial = new Uint8Array(C * N);
    const me = p.smallID();
    const allies = new Set(p.allies().map((a) => a.smallID()));
    const stride = frame.stride;
    for (let c = 0; c < N; c++) {
      const total = this.map.tilesPerObsCell[c];
      if (total === 0) continue;
      const base = c * stride;
      const water = this.map.waterPerObsCell[c];
      let mine = 0;
      let ally = 0;
      let enemy = 0;
      for (let o = 1; o < stride; o++) {
        const n = frame.ownerCounts[base + o];
        if (n === 0) continue;
        if (o === me) mine += n;
        else if (allies.has(o)) ally += n;
        else enemy += n;
      }
      const neutral = frame.ownerCounts[base] - water;
      spatial[0 * N + c] = toByte(mine / total);
      spatial[1 * N + c] = toByte(ally / total);
      spatial[2 * N + c] = toByte(enemy / total);
      spatial[3 * N + c] = toByte(neutral / total);
      spatial[4 * N + c] = toByte(water / total);
      spatial[6 * N + c] = toByte(frame.falloutCounts[c] / total);
    }
    // Front : tuiles terrestres adverses ou neutres au contact de ma frontière.
    if (p.isAlive()) {
      const map = game.map();
      const nbuf: TileRef[] = [0, 0, 0, 0];
      const front = new Uint16Array(N);
      p.borderTiles().forEach((tile) => {
        const n = map.neighbors4(tile, nbuf);
        for (let i = 0; i < n; i++) {
          const nb = nbuf[i];
          if (!map.isWater(nb) && map.ownerID(nb) !== me) front[this.map.tileObsCell[nb]]++;
        }
      });
      for (let c = 0; c < N; c++) {
        if (front[c] > 0) spatial[5 * N + c] = toByte(Math.min(1, (4 * front[c]) / this.map.tilesPerObsCell[c]));
      }
    }
    for (const s of frame.structures) {
      const i = (7 + s.unit) * N + s.cell;
      spatial[i] = s.owner === me ? 255 : Math.max(spatial[i], 128);
    }

    const rows = this.k + 1;
    const F = PLAYER_FEATURES.length;
    const players = new Float32Array(rows * F);
    const myTroops = p.troops();
    const totalLand = this.map.totalLand;
    const outgoing = new Map<number, number>();
    for (const a of p.outgoingAttacks()) {
      const id = a.target().smallID();
      outgoing.set(id, (outgoing.get(id) ?? 0) + a.troops());
    }
    const incoming = new Map<number, number>();
    for (const a of p.incomingAttacks()) {
      const id = a.attacker().smallID();
      incoming.set(id, (incoming.get(id) ?? 0) + a.troops());
    }
    const attackRow = T_ATTACK * rows;
    for (let r = 0; r < rows; r++) {
      const o = this.table[r];
      const f = players.subarray(r * F, (r + 1) * F);
      if (o === null) continue;
      f[0] = 1;
      if (o === "neutral") {
        let owned = 0;
        for (const q of game.allPlayers()) owned += q.numTilesOwned();
        f[1] = 1;
        f[2] = 1;
        f[3] = (totalLand - owned) / totalLand;
        f[9] = this.targetMask[attackRow] ? 1 : 0;
        f[12] = logNorm(outgoing.get(0) ?? 0, 1e6);
        continue;
      }
      const id = o.smallID();
      f[2] = o.isAlive() ? 1 : 0;
      f[3] = o.numTilesOwned() / totalLand;
      f[4] = logNorm(o.troops(), 1e6);
      f[5] = logNorm(Number(o.gold()), 1e7);
      f[6] = clip(Math.log2((o.troops() + 1) / (myTroops + 1)) / 10, -1, 1);
      f[7] = o.relation(p) / 3;
      f[8] = p.isAlliedWith(o) ? 1 : 0;
      f[9] = this.targetMask[attackRow + r] ? 1 : 0;
      f[10] = p.nearby().includes(o) ? 1 : 0;
      f[11] = logNorm(incoming.get(id) ?? 0, 1e6);
      f[12] = logNorm(outgoing.get(id) ?? 0, 1e6);
      f[13] = p.incomingAllianceRequests().some((ar) => ar.requestor() === o) ? 1 : 0;
      f[14] = o.type() === PlayerType.Nation ? 1 : 0;
      f[15] = o.type() === PlayerType.Human ? 1 : 0;
    }

    const config = game.config();
    const maxTroops = p.isAlive() ? config.maxTroops(p) : 0;
    const out = [...outgoing.values()].reduce((a, b) => a + b, 0);
    const inc = [...incoming.values()].reduce((a, b) => a + b, 0);
    const scalars = new Float32Array([
      logNorm(myTroops, 1e6),
      logNorm(maxTroops, 1e6),
      maxTroops > 0 ? myTroops / maxTroops : 0,
      logNorm(Number(p.gold()), 1e7),
      p.numTilesOwned() / totalLand,
      game.ticks() / 30000,
      config.percentageTilesOwnedToWin(game.elapsedGameSeconds()) / 100,
      numSpawned > 0 ? game.players().length / numSpawned : 0,
      game.inSpawnPhase() ? 1 : 0,
      p.unitCount(UnitType.TransportShip) / config.boatMaxNumber(),
      p.unitCount(UnitType.City) / 10,
      clip(out / (myTroops + 1), 0, 4) / 4,
      clip(inc / (myTroops + 1), 0, 4) / 4,
      p.isAlive() ? 1 : 0,
    ]);

    const spawn = new Float32Array(rows * 3);
    if (game.inSpawnPhase()) {
      this.spawnCandidates.forEach((t, i) => {
        spawn[i * 3] = this.targetMask[T_SPAWN * rows + i];
        spawn[i * 3 + 1] = (t % this.map.width) / this.map.width;
        spawn[i * 3 + 2] = Math.floor(t / this.map.width) / this.map.height;
      });
    }

    const A = this.map.actCells;
    const T = ACTION_TYPES.length;
    const U = BUILD_UNITS.length;
    const prefix = this.name;
    return {
      [`${prefix}/spatial`]: { data: spatial, shape: [C, H, W] },
      [`${prefix}/players`]: { data: players, shape: [rows, F] },
      [`${prefix}/scalars`]: { data: scalars, shape: [SCALARS.length] },
      [`${prefix}/spawn`]: { data: spawn, shape: [rows, 3] },
      [`${prefix}/mask_type`]: { data: this.typeMask.slice(), shape: [T] },
      [`${prefix}/mask_target`]: { data: this.targetMask.slice(), shape: [T, rows] },
      [`${prefix}/mask_cell`]: { data: this.cellMask.slice(), shape: [T, A] },
      [`${prefix}/mask_amount`]: { data: this.amountMask.slice(), shape: [AMOUNTS.length] },
      [`${prefix}/mask_unit`]: { data: this.unitMask.slice(), shape: [U] },
      [`${prefix}/mask_unit_cell`]: { data: this.unitCellMask.slice(), shape: [U, A] },
    };
  }

  // ── Actions ──

  /** Vrai si l'action respecte les masques courants. */
  isValid(a: Action): boolean {
    const [type, target, cell, amount, unit] = a;
    const rows = this.k + 1;
    const A = this.map.actCells;
    if (!(type >= 0 && type < ACTION_TYPES.length) || !this.typeMask[type]) return false;
    switch (type) {
      case T_NOOP:
        return true;
      case T_SPAWN:
      case T_ATTACK:
      case T_ALLIANCE:
      case T_BREAK:
        return target >= 0 && target < rows && this.targetMask[type * rows + target] === 1 &&
          (type !== T_ATTACK || this.amountMask[amount] === 1);
      case T_BOAT:
        return cell >= 0 && cell < A && this.cellMask[T_BOAT * A + cell] === 1 && this.amountMask[amount] === 1;
      case T_BUILD:
        return unit >= 0 && unit < BUILD_UNITS.length && cell >= 0 && cell < A &&
          this.unitCellMask[unit * A + cell] === 1;
    }
    return false;
  }

  /** Intention correspondant à l'action (null pour « rien » ou une action masquée). */
  toIntent(a: Action): StampedIntent | null {
    this.lastStale = false;
    if (!this.isValid(a)) {
      this.pending = null;
      this.lastAction = a[0] === T_NOOP ? "rien" : `action masquée (${ACTION_TYPES[a[0]] ?? a[0]})`;
      return null;
    }
    const [type, target, cell, amount, unit] = a;
    const p = this.player;
    const troops = Math.floor(p.troops() * AMOUNTS[amount]);
    const pct = `${Math.round(AMOUNTS[amount] * 100)} %`;
    const row = this.table[target];
    const other = row !== null && row !== "neutral" ? row : null;
    const label = row === "neutral" ? "terres neutres" : (other?.displayName() ?? "?");
    let intent: Intent;
    let description: string;
    let tile: TileRef | undefined;
    switch (type) {
      case T_NOOP:
        this.pending = null;
        this.lastAction = "rien";
        return null;
      case T_SPAWN:
        tile = this.spawnCandidates[target];
        intent = { type: "spawn", tile };
        description = `spawn (candidat ${target})`;
        break;
      case T_ATTACK:
        intent = {
          type: "attack",
          targetID: other === null ? this.game.terraNullius().id() : other.id(),
          troops,
        };
        description = `attaque ${label} (${pct})`;
        break;
      case T_BOAT: {
        tile = this.boatTile[cell];
        // Verdict venu du cache : on le refait avant d'envoyer l'intention.
        if (this.boatCache.get(cell)?.at !== this.maskStep && !this.boatReachable(tile)) {
          this.pending = null;
          this.lastStale = true;
          this.lastAction = `bateau annulé (case ${cell}, masque périmé)`;
          return null;
        }
        intent = { type: "boat", troops, dst: tile };
        description = `bateau vers ${this.game.owner(tile).isPlayer() ? (this.game.owner(tile) as Player).displayName() : "terres neutres"} (case ${cell}, ${pct})`;
        break;
      }
      case T_BUILD: {
        const u = BUILD_UNITS[unit];
        tile = u === UnitType.Port ? this.portBuildTile[cell] : this.landBuildTile[cell];
        if (this.buildCache.get(cell)?.at !== this.maskStep && !this.buildStillValid(u, tile)) {
          this.pending = null;
          this.lastStale = true;
          this.lastAction = `construction annulée (case ${cell}, masque périmé)`;
          return null;
        }
        intent = { type: "build_unit", unit: u, tile };
        description = `construit ${u} (case ${cell})`;
        break;
      }
      case T_ALLIANCE:
        intent = { type: "allianceRequest", recipient: other!.id() };
        description = p.incomingAllianceRequests().some((r) => r.requestor() === other)
          ? `accepte l'alliance de ${label}`
          : `propose une alliance à ${label}`;
        break;
      case T_BREAK:
        intent = { type: "breakAlliance", recipient: other!.id() };
        description = `rompt l'alliance avec ${label}`;
        break;
      default:
        return null;
    }
    this.pending = { type, target: other, tile, description };
    this.lastAction = description;
    return { ...intent, clientID: p.clientID()! } as StampedIntent;
  }

  /**
   * Suit l'exécution créée pour l'intention en attente : le moteur décide de
   * l'accepter dans son init() (attaque, bateau, alliance) ou à son premier
   * tick() (construction, rupture, spawn). On enveloppe ces deux méthodes sur
   * l'instance pour lire son état interne (champs privés en TS) juste après,
   * sans rien changer à la simulation.
   */
  attachExecution(exec: Execution): void {
    const pend = this.pending;
    if (pend === null || pend.exec !== undefined) return;
    pend.exec = exec;
    const rec = exec as unknown as Record<string, unknown> & {
      init(mg: Game, t: number): void;
      tick(t: number): void;
    };
    const init = rec.init.bind(rec);
    const tick = rec.tick.bind(rec);
    const p = this.player;
    const decide = (ok: boolean) => {
      if (pend.accepted === undefined) pend.accepted = ok;
    };
    rec.init = (mg, t) => {
      // État juste avant que le moteur ne juge l'intention, pour expliquer un refus.
      const target = pend.type === T_BOAT ? this.game.owner(pend.tile!) : pend.target;
      const before = target?.isPlayer()
        ? !target.isAlive()
          ? "cible morte"
          : p.isFriendly(target)
            ? "cible alliée"
            : !p.canAttackPlayer(target)
              ? "cible immunisée"
              : null
        : null;
      const canSend = pend.type === T_ALLIANCE ? p.canSendAllianceRequest(pend.target!) : true;
      const incoming = pend.type === T_ALLIANCE && p.incomingAllianceRequests().some((r) => r.requestor() === pend.target);
      init(mg, t);
      if (pend.type === T_ALLIANCE) {
        pend.reason = !p.isAlive()
          ? "agent mort"
          : !canSend
            ? "demande impossible à l'exécution"
            : incoming
              ? "acceptation sans alliance"
              : "demande non créée";
      }
      if (pend.type === T_ATTACK || pend.type === T_BOAT) {
        pend.reason = !p.isAlive()
          ? "agent mort"
          : (before ?? (pend.type === T_ATTACK ? "plus de front commun" : "pas de trajet par l'eau"));
      }
      switch (pend.type) {
        case T_ATTACK: {
          const toConquer = rec.toConquer as { size(): number } | undefined;
          decide(rec.attack !== null && rec.attack !== undefined && (toConquer?.size() ?? 0) > 0);
          break;
        }
        case T_BOAT:
          decide(rec.boat !== undefined);
          break;
        case T_ALLIANCE:
          decide(rec.req !== null || p.isAlliedWith(pend.target!));
          break;
      }
    };
    rec.tick = (t) => {
      if (pend.type === T_BUILD && pend.accepted === undefined && rec.structure === null) {
        const unit = rec.constructionType as UnitType;
        const g = this.game;
        pend.reason =
          g.owner(pend.tile!) !== p
            ? "tuile perdue"
            : p.gold() < g.unitInfo(unit).cost(g, p)
              ? "or insuffisant"
              : "pas d'emplacement valide";
      }
      tick(t);
      switch (pend.type) {
        case T_BUILD:
          decide(rec.structure !== null && rec.structure !== undefined);
          break;
        case T_BREAK:
          // Si l'allié rompt dans le même Turn, le moteur ignore la seconde
          // demande : l'alliance est rompue quand même, on la compte réussie.
          decide(!p.isAlliedWith(pend.target!));
          break;
        case T_SPAWN:
          decide(p.hasSpawned() && p.spawnTile() === pend.tile);
          break;
      }
    };
  }

  /** Après le pas : le moteur a-t-il accepté l'intention ? null s'il n'y en avait pas. */
  resolvePending(): boolean | null {
    const pend = this.pending;
    this.pending = null;
    this.lastAccepted = pend === null ? null : (pend.accepted ?? false);
    this.lastRejectReason = this.lastAccepted === false ? (pend?.reason ?? null) : null;
    return this.lastAccepted;
  }

  display(): SeatDisplay {
    const rows = this.k + 1;
    const A = this.map.actCells;
    const validCounts: Record<string, number> = {};
    ACTION_TYPES.forEach((t, i) => {
      if (i === T_NOOP) return;
      const m = i === T_BOAT || i === T_BUILD
        ? this.cellMask.subarray(i * A, (i + 1) * A)
        : this.targetMask.subarray(i * rows, (i + 1) * rows);
      validCounts[t] = this.typeMask[i] ? m.reduce((s, v) => s + v, 0) : 0;
    });
    return {
      id: this.player.smallID(),
      name: this.player.displayName(),
      lastAction: this.lastAction,
      accepted: this.lastAccepted,
      validCounts,
      annotations: this.annotations,
    };
  }
}
