/**
 * Rejeu headless d'un GameRecord, comme openfront/tests/replay/ReplayGame.ts
 * (et le lecteur côté client) : on reconstruit la partie depuis le record, on
 * rejoue ses Turns et on compare les hashes recalculés à ceux enregistrés.
 *
 *   npx tsx src/replay.ts runs/<run>/games/<gameID>.json.gz
 */
import { PlayerInfo, PlayerType } from "@openfront/engine-api/game/GameTypes";
import { GameUpdateType, HashUpdate } from "@openfront/engine-api/game/GameUpdates";
import { GameStartInfo } from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { decompressGameRecord, toWireGameStartInfo } from "@openfront/shared/SharedUtil";
import { GameRecord } from "@openfront/shared/WireSchemas";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { NodeGameMapLoader } from "./NodeGameMapLoader";
import { readRecordFile } from "./records";

const MAPS_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../openfront/resources/maps");

export interface ReplayResult {
  ticks: number;
  /** Hashes enregistrés comparés, et combien sont identiques. */
  compared: number;
  matches: number;
  firstMismatch: number | null;
  /** Dernier hash recalculé. */
  finalHash: number | null;
  winner: string | null;
}

export async function replayRecord(record: GameRecord): Promise<ReplayResult> {
  const full = decompressGameRecord(record);
  const info = full.info;
  const gameStart: GameStartInfo = toWireGameStartInfo({
    gameID: info.gameID,
    lobbyCreatedAt: info.lobbyCreatedAt,
    config: info.config,
    players: info.players,
    tribes: info.tribes,
  });
  const terrain = await loadTerrainMap(
    await loadMapFiles(new NodeGameMapLoader(MAPS_DIR), info.config.gameMap, info.config.gameMapSize),
  );
  // Mêmes tirages que createGameRunner() : humains puis nations.
  const random = new PseudoRandom(simpleHash(gameStart.gameID));
  const humans = gameStart.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
        p.teamIndex ?? null,
      ),
  );
  const nations = createNationsForGame(gameStart, terrain.nations, terrain.additionalNations, humans.length, random);
  const game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    new EngineConfig(info.config, false),
    terrain.teamGameSpawnAreas,
  );
  const computed = new Map<number, number>();
  let finalHash: number | null = null;
  let fatal: string | undefined;
  const runner = new GameRunner(game, new Executor(game, gameStart.gameID, undefined), (gu) => {
    if ("errMsg" in gu) {
      fatal = gu.errMsg;
      return;
    }
    for (const h of gu.updates[GameUpdateType.Hash] as HashUpdate[]) {
      computed.set(h.tick, h.hash);
      finalHash = h.hash;
    }
  });
  runner.init();

  let compared = 0;
  let matches = 0;
  let firstMismatch: number | null = null;
  for (const turn of full.turns) {
    runner.addTurn(turn);
    if (!runner.executeNextTick() || fatal !== undefined) {
      throw new Error(`rejeu interrompu au tour ${turn.turnNumber} : ${fatal}`);
    }
    if (turn.hash !== null && turn.hash !== undefined) {
      compared++;
      if (computed.get(turn.turnNumber) === turn.hash) matches++;
      else firstMismatch ??= turn.turnNumber;
    }
  }
  const w = game.getWinner();
  return {
    ticks: game.ticks(),
    compared,
    matches,
    firstMismatch,
    finalHash,
    winner: w === null ? null : typeof w === "string" ? w : w.displayName(),
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const file = process.argv[2];
  if (!file) {
    console.error("usage : npx tsx src/replay.ts <record.json[.gz]>");
    process.exit(2);
  }
  const log = console.log;
  console.log = console.debug = console.info = console.warn = () => {};
  const record = readRecordFile(file);
  const r = await replayRecord(record);
  const recordedWinner = record.info.winner ?? null;
  log(`${record.info.gameID} : ${r.ticks} ticks rejoués, hashes identiques ${r.matches}/${r.compared}`);
  log(`  vainqueur rejoué : ${r.winner ?? "aucun"} ; enregistré : ${recordedWinner ? JSON.stringify(recordedWinner) : "aucun"}`);
  if (r.firstMismatch !== null) {
    log(`  PREMIER ÉCART au tour ${r.firstMismatch}`);
    process.exit(1);
  }
}
