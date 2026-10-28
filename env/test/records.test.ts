import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { GameRecordSchema } from "@openfront/shared/WireSchemas";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import type { EnvConfig } from "../src/agentEnv";
import type { Action } from "../src/seat";

// Les records de test vont dans un dossier temporaire (lu au chargement de records.ts).
const RUNS = fs.mkdtempSync(path.join(os.tmpdir(), "openfront-runs-"));
process.env.OPENFRONT_RUNS_DIR = RUNS;
const { AgentEnv } = await import("../src/agentEnv");
const { ruleAction } = await import("../src/sampling");
const { listGames, readRecordFile, shouldRecord } = await import("../src/records");
const { replayRecord } = await import("../src/replay");

beforeAll(() => {
  console.log = console.debug = console.info = console.warn = () => {};
});

/** Joue jusqu'à la fin de l'épisode (ou maxSteps) et renvoie l'entrée d'index écrite. */
async function playAndRecord(config: EnvConfig, maxSteps: number) {
  const env = await AgentEnv.create(config);
  let r = env.reset();
  const random = new PseudoRandom(1);
  for (let i = 0; i < maxSteps && (r.agents.length > 0 || env.seats.length === 0) && !r.state.done; i++) {
    const actions: Record<string, Action> = {};
    for (const seat of env.seats) if (r.agents.includes(seat.name)) actions[seat.name] = ruleAction(seat, random);
    r = env.step(actions, 10);
  }
  const entry = env.finish();
  expect(entry).not.toBeNull();
  return { entry: entry!, state: r.state };
}

describe("enregistrement et rejeu", () => {
  it("un record avec agent est valide et se rejoue au même hash", async () => {
    const { entry, state } = await playAndRecord(
      { seed: "record-agent", agents: 1, nations: 7, record: { run: "test" } },
      400,
    );
    const record = readRecordFile(path.join(RUNS, entry.file));
    // Le lecteur officiel refuse un record qui ne passe pas ce schéma.
    const parsed = GameRecordSchema.safeParse(record);
    expect(parsed.success, JSON.stringify(parsed.error?.issues?.slice(0, 3))).toBe(true);
    expect(record.turns.some((t) => t.intents.length > 0)).toBe(true);

    const replay = await replayRecord(record);
    expect(replay.compared).toBeGreaterThan(0);
    expect(replay.firstMismatch).toBeNull();
    expect(replay.matches).toBe(replay.compared);
    expect(replay.finalHash).toBe(state.hash);
  });

  it("une partie finie se rejoue avec le même vainqueur", async () => {
    const { entry, state } = await playAndRecord({ seed: "record-fin", agents: 0, nations: 8, record: { run: "test" } }, 5000);
    expect(state.done).toBe(true);
    expect(entry.finished).toBe(true);
    const replay = await replayRecord(readRecordFile(path.join(RUNS, entry.file)));
    expect(replay.firstMismatch).toBeNull();
    expect(replay.winner).toBe(state.winner);
    expect(replay.finalHash).toBe(state.hash);
    expect(listGames().map((g) => g.gameID)).toContain(entry.gameID);
  });

  it("n'enregistre qu'une partie sur N", () => {
    const picks = Array.from({ length: 7 }, () => shouldRecord({ run: "un-sur-trois", every: 3 }));
    expect(picks).toEqual([true, false, false, true, false, false, true]);
  });
});
