/**
 * Partie OpenFront headless : même pipeline que
 * openfront/tests/perf/fullgame/FullGamePerf.ts (EngineConfig réel, GameRunner,
 * Executor, nations du manifeste de la carte), piloté tick par tick.
 */
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerType,
} from "@openfront/engine-api/game/GameTypes";
import {
  GameUpdateType,
  HashUpdate,
} from "@openfront/engine-api/game/GameUpdates";
import { GameConfig, GameStartInfo } from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import { Game, Player } from "@openfront/engine/game/Game";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeGameMapLoader } from "./NodeGameMapLoader";

const MAPS_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../openfront/resources/maps",
);
const MAX_SPAWN_TURNS = 1000;

export interface ResetConfig {
  /** Nom de carte insensible à la casse, ex. "world", "europe". */
  map?: string;
  size?: "Compact" | "Normal";
  nations?: number;
  difficulty?: "Easy" | "Medium" | "Hard" | "Impossible";
  seed?: string;
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

export class OpenFrontGame {
  private turnNumber = 0;
  private lastHash: number | null = null;
  private fatalError: string | undefined;
  private runner: GameRunner;

  private constructor(
    readonly config: Required<ResetConfig>,
    readonly game: Game,
    gameID: string,
  ) {
    this.runner = new GameRunner(game, new Executor(game, gameID, undefined), (gu) => {
      if ("errMsg" in gu) {
        this.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      if (hashes.length > 0) {
        this.lastHash = hashes[hashes.length - 1].hash;
      }
    });
    this.runner.init();
  }

  static async create(cfg: ResetConfig = {}): Promise<OpenFrontGame> {
    const config: Required<ResetConfig> = {
      map: cfg.map ?? "world",
      size: cfg.size ?? "Compact",
      nations: cfg.nations ?? 8,
      difficulty: cfg.difficulty ?? "Medium",
      seed: cfg.seed ?? "openfront-ai",
    };
    const gameConfig: GameConfig = {
      gameMap: resolveMap(config.map),
      gameMapSize: GameMapSize[config.size],
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: Difficulty[config.difficulty],
      nations: config.nations,
      donateGold: false,
      donateTroops: false,
      bots: 0,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const gameStart: GameStartInfo = {
      gameID: config.seed,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: [],
    };
    const terrain = await loadTerrainMap(
      await loadMapFiles(
        new NodeGameMapLoader(MAPS_DIR),
        gameConfig.gameMap,
        gameConfig.gameMapSize,
      ),
    );
    const nations = createNationsForGame(
      gameStart,
      terrain.nations,
      terrain.additionalNations,
      0,
      new PseudoRandom(simpleHash(gameStart.gameID)),
    );
    const game = createGame(
      [],
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      new EngineConfig(gameConfig, false),
      terrain.teamGameSpawnAreas,
    );
    const g = new OpenFrontGame(config, game, gameStart.gameID);
    // La phase de spawn n'offre aucune décision à l'agent pour l'instant.
    while (game.inSpawnPhase()) {
      if (g.turnNumber >= MAX_SPAWN_TURNS) {
        throw new Error(`phase de spawn non terminée après ${MAX_SPAWN_TURNS} tours`);
      }
      g.tick();
    }
    return g;
  }

  private tick(): void {
    this.runner.addTurn({ turnNumber: this.turnNumber++, intents: [] });
    this.runner.executeNextTick();
    if (this.fatalError !== undefined) {
      throw new Error(`erreur du moteur au tick ${this.game.ticks()}:\n${this.fatalError}`);
    }
  }

  done(): boolean {
    return this.game.getWinner() !== null;
  }

  /** Avance de n ticks, ou moins si la partie se termine. */
  step(n = 10): GameState {
    for (let i = 0; i < n && !this.done(); i++) {
      this.tick();
    }
    return this.state();
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
