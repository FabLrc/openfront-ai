import { beforeAll, describe, expect, it } from "vitest";
import { AgentEnv } from "../src/agentEnv";
import { T_NOOP } from "../src/seat";

beforeAll(() => {
  console.log = console.debug = console.info = console.warn = () => {};
});

async function play(observeNations: number) {
  const env = await AgentEnv.create({ seed: "tap-test", agents: 0, nations: 8, difficulty: "Hard", observeNations, maxTicks: 6000 });
  let r = env.reset();
  const labels: { kind: string; mapped: boolean; valid: boolean; action: number[] }[] = [];
  while (!r.state.done && r.state.tick < 6000) {
    r = env.step({}, 10);
    labels.push(...Object.values(r.labels ?? {}));
  }
  return { hash: r.state.hash, tick: r.state.tick, labels, observed: r.observed };
}

describe("imitation des Nations", () => {
  it("observer les Nations ne change pas la partie", async () => {
    const plain = await play(0);
    const observed = await play(4);
    expect(observed.tick).toBe(plain.tick);
    expect(observed.hash).toBe(plain.hash);
  });

  it("traduit la plupart des décisions, et les étiquettes valides respectent les masques", async () => {
    const { labels, observed } = await play(4);
    expect(observed?.length).toBeGreaterThan(0);
    const decisions = labels.filter((l) => l.kind !== "noop");
    expect(decisions.length).toBeGreaterThan(20);
    const mapped = decisions.filter((l) => l.mapped);
    expect(mapped.length / decisions.length).toBeGreaterThan(0.8);
    // « rien » n'est valide que sans aucune décision dans le pas.
    for (const l of labels) if (!l.mapped) expect(l.action[0]).toBe(T_NOOP);
    expect(mapped.filter((l) => l.valid).length / mapped.length).toBeGreaterThan(0.6);
  });
});
