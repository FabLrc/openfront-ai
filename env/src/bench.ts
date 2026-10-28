/**
 * Mesure du débit de l'environnement multi-agents, sans socket : agents
 * aléatoires (dans les masques), temps passé par phase.
 *
 *   npx tsx src/bench.ts [--agents 1] [--nations 7] [--games 3] [--seed bench]
 *                        [--policy random|rule] [--steps 0 (sans limite)]
 */
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { parseArgs } from "node:util";
import { AgentEnv } from "./agentEnv";
import { ruleAction, sampleAction } from "./sampling";
import { Action, AgentSeat, SharedFrame } from "./seat";

const { values } = parseArgs({
  options: {
    agents: { type: "string", default: "1" },
    nations: { type: "string", default: "7" },
    games: { type: "string", default: "3" },
    seed: { type: "string", default: "bench" },
    policy: { type: "string", default: "random" },
    steps: { type: "string", default: "0" },
  },
});
const maxSteps = parseInt(values.steps, 10);
const policy = values.policy === "rule" ? ruleAction : sampleAction;

const log = console.log;
console.log = console.debug = console.info = console.warn = () => {};

// Chronométrage par phase en enveloppant les méthodes coûteuses.
const timers: Record<string, number> = {};
function timed<T extends object>(proto: T, name: keyof T & string, label: string) {
  const original = proto[name] as unknown as (...a: unknown[]) => unknown;
  (proto as Record<string, unknown>)[name] = function (this: unknown, ...a: unknown[]) {
    const t0 = performance.now();
    try {
      return original.apply(this, a);
    } finally {
      timers[label] = (timers[label] ?? 0) + performance.now() - t0;
    }
  };
}
timed(AgentSeat.prototype, "computeMasks", "masques");
timed(AgentSeat.prototype as unknown as Record<string, unknown>, "computeBoatMask", "  dont bateaux");
timed(AgentSeat.prototype as unknown as Record<string, unknown>, "computeBuildMask", "  dont constructions");
timed(AgentSeat.prototype, "observe", "observation");

let steps = 0;
let ticks = 0;
let simMs = 0;
let frameMs = 0;
const t0 = performance.now();
for (let g = 0; g < parseInt(values.games, 10); g++) {
  const env = await AgentEnv.create({
    seed: `${values.seed}-${g}`,
    agents: parseInt(values.agents, 10),
    nations: parseInt(values.nations, 10),
    boatCandidates: process.env.BOATS ? parseInt(process.env.BOATS, 10) : undefined,
  });
  const random = new PseudoRandom(g);
  const sim = env.game as unknown as { step: (...a: unknown[]) => unknown };
  const step = sim.step.bind(sim);
  sim.step = (...a: unknown[]) => {
    const s = performance.now();
    try {
      return step(...a);
    } finally {
      simMs += performance.now() - s;
    }
  };
  let r = env.reset();
  const start = env.game.game.ticks();
  for (let n = 0; r.agents.length > 0 && !r.state.done && (maxSteps === 0 || n < maxSteps); n++) {
    const actions: Record<string, Action> = {};
    for (const seat of env.seats) if (r.agents.includes(seat.name)) actions[seat.name] = policy(seat, random);
    r = env.step(actions, 10);
    steps++;
  }
  ticks += env.game.game.ticks() - start;
  const f0 = performance.now();
  new SharedFrame(env.map);
  frameMs += performance.now() - f0;
}
const total = performance.now() - t0;
log(`${values.agents} agent(s), ${values.nations} Nations : ${steps} pas, ${ticks} ticks en ${(total / 1000).toFixed(1)} s`);
log(`  ${Math.round(steps / (total / 1000))} pas/s, ${Math.round(ticks / (total / 1000))} ticks/s`);
log(`  simulation ${(simMs / steps).toFixed(2)} ms/pas, comptage de la carte ~${frameMs / parseInt(values.games, 10)} ms`);
for (const [k, v] of Object.entries(timers)) log(`  ${k} : ${(v / steps).toFixed(2)} ms/pas`);
