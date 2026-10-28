/**
 * Partie OpenFront headless : même pipeline que
 * openfront/tests/perf/fullgame/FullGamePerf.ts (EngineConfig réel, GameRunner,
 * Executor, nations du manifeste de la carte), piloté tick par tick.
 *
 * Les sièges d'agents sont des joueurs Human, comme de vrais clients : ils
 * agissent par des intentions placées dans le Turn du tick suivant.
 */
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "@openfront/engine-api/game/GameTypes";
import {
  GameUpdateType,
  HashUpdate,
} from "@openfront/engine-api/game/GameUpdates";
import {
  ClientID,
  GAME_ID_REGEX,
  GameConfig,
  GameStartInfo,
  StampedIntent,
  Turn,
  Winner,
} from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import { Execution, Game, Player } from "@openfront/engine/game/Game";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { createPartialGameRecord } from "@openfront/shared/SharedUtil";
import { GameRecord, PlayerRecord } from "@openfront/shared/WireSchemas";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeGameMapLoader } from "./NodeGameMapLoader";

const MAPS_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../openfront/resources/maps",
);
const MAX_SPAWN_TURNS = 1000;
/** Commit d'OpenFrontIO du sous-module, inscrit dans les records (le lecteur officiel le vérifie hors build DEV). */
export const OPENFRONT_COMMIT = "d1f236242864f506a040f2d56cc5ddd41bf27612";

/**
 * Identifiant de partie dérivé de la graine. Le lecteur de replays n'accepte
 * que 8 à 10 caractères alphanumériques (GAME_ID_REGEX) ; une graine qui n'en
 * est pas un est remplacée par les 10 premiers caractères de son SHA-1.
 * L'identifiant initialise les tirages du moteur : il fait partie de la partie.
 */
export function gameIDFromSeed(seed: string): string {
  return GAME_ID_REGEX.test(seed) ? seed : createHash("sha1").update(seed).digest("hex").slice(0, 10);
}

export interface ResetConfig {
  /** Nom de carte insensible à la casse, ex. "world", "europe". */
  map?: string;
  size?: "Compact" | "Normal";
  /** Nombre de Nations (0 : aucune). */
  nations?: number;
  difficulty?: "Easy" | "Medium" | "Hard" | "Impossible";
  seed?: string;
  /** Nombre de sièges d'agents (joueurs Human). */
  agents?: number;
}

export interface PlayerState {
  id: number;
  name: string;
  type: string;
  alive: boolean;
  tiles: number;
  troops: number;
  gold: number;
}

export interface GameState {
  tick: number;
  inSpawnPhase: boolean;
  alivePlayers: number;
  players: PlayerState[];
  /** Nom du vainqueur, null tant que la partie n'est pas finie. */
  winner: string | null;
  done: boolean;
  /** Dernier hash d'état publié par le moteur (déterminisme). */
  hash: number | null;
}

export function resolveMap(name: string): GameMapType {
  const key = Object.keys(GameMapType).find(
    (k) => k.toLowerCase() === name.toLowerCase(),
  );
  if (key === undefined) {
    throw new Error(`carte inconnue "${name}"`);
  }
  return GameMapType[key as keyof typeof GameMapType];
}

/** clientID du siège i (8 à 10 caractères alphanumériques, cf. GAME_ID_REGEX). */
export function seatClientID(i: number): ClientID {
  return `AGENT${String(i).padStart(3, "0")}`;
}

/** Signale chaque exécution créée pour une intention, pour savoir si le moteur l'accepte. */
class TrackingExecutor extends Executor {
  onCreate: (intent: StampedIntent, exec: Execution) => void = () => {};

  createExec(intent: StampedIntent): Execution {
    const exec = super.createExec(intent);
    this.onCreate(intent, exec);
    return exec;
  }
}

export class OpenFrontGame {
  private turnNumber = 0;
  private lastHash: number | null = null;
  private fatalError: string | undefined;
  private runner: GameRunner;
  private executor: TrackingExecutor;
  readonly seats: Player[];
  /** Tous les Turns joués, avec le hash publié par le moteur à leur tick. */
  readonly turns: Turn[] = [];
  readonly startedAt = Date.now();

  private constructor(
    readonly config: Required<ResetConfig>,
    readonly game: Game,
    readonly gameStart: GameStartInfo,
    seatClientIDs: ClientID[],
  ) {
    const gameID = gameStart.gameID;
    this.executor = new TrackingExecutor(game, gameID, undefined);
    this.runner = new GameRunner(game, this.executor, (gu) => {
      if ("errMsg" in gu) {
        this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      for (const h of hashes) {
        this.lastHash = h.hash;
        const turn = this.turns[h.tick];
        if (turn !== undefined) turn.hash = h.hash;
      }
    });
    this.runner.init();
    this.seats = seatClientIDs.map((id) => game.playerByClientID(id)!);
  }

  static async create(cfg: ResetConfig = {}): Promise<OpenFrontGame> {
    const config: Required<ResetConfig> = {
      map: cfg.map ?? "world",
      size: cfg.size ?? "Compact",
      nations: cfg.nations ?? 8,
      difficulty: cfg.difficulty ?? "Medium",
      seed: cfg.seed ?? "openfront-ai",
      agents: cfg.agents ?? 0,
    };
    const gameConfig: GameConfig = {
      gameMap: resolveMap(config.map),
      gameMapSize: GameMapSize[config.size],
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: Difficulty[config.difficulty],
      nations: config.nations > 0 ? config.nations : "disabled",
      donateGold: false,
      donateTroops: false,
      bots: 0,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const seatIDs = Array.from({ length: config.agents }, (_, i) => seatClientID(i));
    const gameStart: GameStartInfo = {
      gameID: gameIDFromSeed(config.seed),
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: seatIDs.map((clientID, i) => ({
        clientID,
        username: `Agent ${i}`,
        clanTag: null,
      })),
    };
    const terrain = await loadTerrainMap(
      await loadMapFiles(
        new NodeGameMapLoader(MAPS_DIR),
        gameConfig.gameMap,
        gameConfig.gameMapSize,
      ),
    );
    // Même ordre de tirages que createGameRunner() : humains puis nations.
    const random = new PseudoRandom(simpleHash(gameStart.gameID));
    const humans = gameStart.players.map(
      (p) =>
        new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID(), false, null, [], null),
    );
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      humans.length,
      random,
    );
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new EngineConfig(gameConfig, false),
      terrain.teamGameSpawnAreas,
    );
    const g = new OpenFrontGame(config, game, gameStart, seatIDs);
    // Sans agent, la phase de spawn n'offre aucune décision : on la joue tout de suite.
    if (config.agents === 0) g.finishSpawnPhase();
    return g;
  }

  private tick(intents: StampedIntent[] = []): void {
    const turn: Turn = { turnNumber: this.turnNumber++, intents };
    this.turns.push(turn);
    this.runner.addTurn(turn);
    this.runner.executeNextTick();
    if (this.fatalError !== undefined) {
      throw new Error(`erreur du moteur au tick ${this.game.ticks()}:\n${this.fatalError}`);
    }
  }

  /** Joue les ticks restants de la phase de spawn, intentions dans le premier. */
  finishSpawnPhase(intents: StampedIntent[] = []): void {
    let first = true;
    while (this.game.inSpawnPhase()) {
      if (this.turnNumber >= MAX_SPAWN_TURNS) {
        throw new Error(`phase de spawn non terminée après ${MAX_SPAWN_TURNS} tours`);
      }
      this.tick(first ? intents : []);
      first = false;
    }
  }

  done(): boolean {
    return this.game.getWinner() !== null;
  }

  /**
   * Avance de n ticks, ou moins si la partie se termine. Les intentions
   * partent dans le Turn du premier tick.
   */
  step(n = 10, intents: StampedIntent[] = []): GameState {
    for (let i = 0; i < n && !this.done(); i++) {
      this.tick(i === 0 ? intents : []);
    }
    return this.state();
  }

  /** Appelé pour chaque exécution créée à partir d'une intention. */
  onExecutionCreated(fn: (intent: StampedIntent, exec: Execution) => void): void {
    this.executor.onCreate = fn;
  }

  get gameID(): string {
    return this.gameStart.gameID;
  }

  /**
   * GameRecord au format officiel (comme GameServer.archiveGame) : rejoué par
   * le moteur, il redonne la même partie, les Nations étant recréées à
   * l'identique. Sans vainqueur si la partie n'est pas finie.
   */
  toRecord(): GameRecord {
    const stats = this.game.stats().stats();
    const players: PlayerRecord[] = this.gameStart.players.map((p) => ({
      clientID: p.clientID,
      username: p.username,
      clanTag: p.clanTag,
      persistentID: null,
      stats: stats[p.clientID] ?? {},
    }));
    const w = this.game.getWinner();
    const winner =
      w === null ? undefined : (this.game as unknown as { makeWinner(w: unknown): Winner }).makeWinner(w);
    // Tick de jeu = 100 ms : la durée est celle de la partie, pas du calcul.
    const end = this.startedAt + this.turns.length * 100;
    const partial = createPartialGameRecord(
      this.gameID,
      this.gameStart.config,
      players,
      this.turns,
      this.startedAt,
      end,
      winner,
      // Comme GameStartInfo : les rejeux reconstruisent la partie depuis ces champs.
      this.gameStart.lobbyCreatedAt,
    );
    return { ...partial, gitCommit: OPENFRONT_COMMIT };
  }

  state(): GameState {
    const players = this.game.allPlayers().map(playerState);
    const winner = this.game.getWinner();
    return {
      tick: this.game.ticks(),
      inSpawnPhase: this.game.inSpawnPhase(),
      alivePlayers: players.filter((p) => p.alive).length,
      players,
      winner:
        winner === null ? null : typeof winner === "string" ? winner : winner.displayName(),
      done: winner !== null,
      hash: this.lastHash,
    };
  }
}

function playerState(p: Player): PlayerState {
  return {
    id: p.smallID(),
    name: p.displayName(),
    type: p.type() === PlayerType.Nation ? "nation" : p.type().toLowerCase(),
    alive: p.isAlive(),
    tiles: p.numTilesOwned(),
    troops: Math.round(p.troops()),
    gold: Number(p.gold()),
  };
}
