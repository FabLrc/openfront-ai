import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { beforeAll, describe, expect, it } from "vitest";
import { AgentEnv, EnvConfig } from "../src/agentEnv";
import { sampleAction } from "../src/sampling";
import { ACTION_TYPES, Action, T_NOOP } from "../src/seat";

beforeAll(() => {
  // Le moteur journalise beaucoup (et avertit des intentions refusées).
  console.log = console.debug = console.info = console.warn = () => {};
});

const BASE: EnvConfig = { map: "world", size: "Compact", nations: 7, agents: 1 };

/** Joue des actions tirées au hasard dans les masques ; renvoie le hash final et le bilan. */
async function play(config: EnvConfig, steps: number, actionSeed: number) {
  const env = await AgentEnv.create(config);
  env.reset();
  const random = new PseudoRandom(actionSeed);
  const accepted: Record<string, number> = {};
  const rejected: { type: string; tick: number; action: string }[] = [];
  let result = env.step({}); // phase de spawn (spawn aléatoire valide)
  for (let i = 0; i < steps && result.agents.length > 0; i++) {
    const actions: Record<string, Action> = {};
    for (const seat of env.seats) {
      if (result.agents.includes(seat.name)) actions[seat.name] = sampleAction(seat, random);
    }
    result = env.step(actions, 10);
    for (const seat of env.seats) {
      const a = actions[seat.name];
      if (a === undefined || a[0] === T_NOOP) continue;
      const ok = result.infos[seat.name].accepted;
      const type = ACTION_TYPES[a[0]];
      if (ok === true) accepted[type] = (accepted[type] ?? 0) + 1;
      else rejected.push({ type, tick: result.state.tick, action: `${seat.lastAction} [${seat.lastRejectReason}]` });
    }
  }
  return { hash: result.state.hash, tick: result.state.tick, accepted, rejected, env };
}

describe("sièges d'agents", () => {
  it("même graine + mêmes actions = même hash", async () => {
    const a = await play({ ...BASE, seed: "agent-determinisme" }, 150, 42);
    const b = await play({ ...BASE, seed: "agent-determinisme" }, 150, 42);
    expect(a.hash).not.toBeNull();
    expect(b.tick).toBe(a.tick);
    expect(b.hash).toBe(a.hash);
    // Une autre suite d'actions change la partie.
    const c = await play({ ...BASE, seed: "agent-determinisme" }, 150, 43);
    expect(c.hash).not.toBe(a.hash);
  });

  async function sample(games: [string, number, number][]) {
    const accepted: Record<string, number> = {};
    const rejected: { type: string; tick: number; action: string }[] = [];
    for (const [seed, agents, nations] of games) {
      const r = await play({ ...BASE, seed, agents, nations }, 400, 7);
      for (const [k, v] of Object.entries(r.accepted)) accepted[k] = (accepted[k] ?? 0) + v;
      rejected.push(...r.rejected);
    }
    const total = Object.values(accepted).reduce((a, b) => a + b, 0) + rejected.length;
    console.error(`acceptées ${JSON.stringify(accepted)}, refusées ${rejected.length}/${total}`, rejected);
    return { accepted, rejected, total };
  }

  // Le moteur juge une intention un tick après l'observation, une fois les
  // exécutions en cours avancées : un front, une tuile ou un trajet par l'eau
  // peut disparaître entre-temps. Les masques prennent des marges ; il reste
  // de rares refus de ce type, jamais une cible interdite (alliée, immunisée).
  const RACES = ["plus de front commun", "pas de trajet par l'eau", "tuile perdue", "cible morte"];

  it("un agent seul : refus très rares, seulement par course avec le tick suivant", async () => {
    const { accepted, rejected, total } = await sample([
      ...Array.from({ length: 6 }, (_, i): [string, number, number] => [`masques-${i}`, 1, 7]),
      ["masques-6", 1, 3],
      ["masques-7", 1, 1],
    ]);
    expect(rejected.length / total).toBeLessThan(0.005);
    for (const r of rejected) expect(RACES.some((reason) => r.action.includes(reason)), r.action).toBe(true);
    for (const t of ["attack", "boat", "build", "alliance"]) {
      expect(accepted[t] ?? 0).toBeGreaterThan(0);
    }
  });

  it("plusieurs agents : refus rares (< 0,5 %)", async () => {
    // Les intentions de tous les agents d'un même Turn s'exécutent ensemble :
    // l'un peut, par exemple, accepter une alliance qui rend l'attaque de
    // l'autre impossible, ou prendre la tuile où l'autre construit. Les
    // masques, calculés avant, ne peuvent pas le prévoir ; on borne la fréquence.
    const { rejected, total } = await sample([
      ["masques-4", 4, 4],
      ["masques-5", 8, 0],
    ]);
    expect(rejected.length / total).toBeLessThan(0.005);
  });

  it("observations aux formes annoncées", async () => {
    const env = await AgentEnv.create({ ...BASE, seed: "formes" });
    const r = env.reset();
    const spec = env.spec() as { obs_width: number; obs_height: number; k: number; channels: string[] };
    expect(r.buffers["agent_0/spatial"].shape).toEqual([spec.channels.length, spec.obs_height, spec.obs_width]);
    expect(r.buffers["agent_0/players"].shape[0]).toBe(spec.k + 1);
    expect(r.buffers["agent_0/mask_type"].data[1]).toBe(1); // spawn possible
  });
});
