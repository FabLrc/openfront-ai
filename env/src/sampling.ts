/** Tirage d'une action valide au hasard, en suivant les masques (tests, mesures). */
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import {
  Action,
  AgentSeat,
  T_ATTACK,
  T_BOAT,
  T_BUILD,
  T_NOOP,
  T_SPAWN,
} from "./seat";

function pick(random: PseudoRandom, mask: ArrayLike<number>): number {
  const valid: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) valid.push(i);
  return valid.length === 0 ? -1 : valid[random.nextInt(0, valid.length)];
}

/** Action valide uniforme sur les types autorisés, puis sur leurs paramètres. */
export function sampleAction(seat: AgentSeat, random: PseudoRandom): Action {
  const type = pick(random, seat.typeMask);
  if (type <= T_NOOP) return [T_NOOP, 0, 0, 0, 0];
  const rows = seat.k + 1;
  const A = seat.map.actCells;
  const amount = pick(random, seat.amountMask);
  if (type === T_BOAT) {
    return [type, 0, pick(random, seat.cellMask.subarray(T_BOAT * A, (T_BOAT + 1) * A)), amount, 0];
  }
  if (type === T_BUILD) {
    const unit = pick(random, seat.unitMask);
    return [type, 0, pick(random, seat.unitCellMask.subarray(unit * A, (unit + 1) * A)), 0, unit];
  }
  const target = pick(random, seat.targetMask.subarray(type * rows, (type + 1) * rows));
  return [type, target, 0, type === T_ATTACK ? amount : 0, 0];
}

/** Même politique que RuleAgent (agent/openfront_ai/agents.py), pour les mesures. */
export function ruleAction(seat: AgentSeat, random: PseudoRandom, threshold = 0.2): Action {
  const rows = seat.k + 1;
  const A = seat.map.actCells;
  if (seat.typeMask[T_SPAWN]) {
    return [T_SPAWN, pick(random, seat.targetMask.subarray(T_SPAWN * rows, (T_SPAWN + 1) * rows)), 0, 0, 0];
  }
  if (seat.typeMask[T_BUILD] && seat.unitMask[0]) return [T_BUILD, 0, pick(random, seat.unitCellMask.subarray(0, A)), 0, 0];
  const p = seat.player;
  const ratio = p.troops() / Math.max(1, seat.map.game.config().maxTroops(p));
  if (seat.typeMask[T_ATTACK] && ratio > threshold) {
    const targets = seat.targetMask.subarray(T_ATTACK * rows, (T_ATTACK + 1) * rows);
    if (targets[0]) return [T_ATTACK, 0, 0, 1, 0];
    let best = -1;
    for (let r = 1; r < rows; r++) {
      const o = seat.table[r];
      if (!targets[r] || o === null || o === "neutral") continue;
      const b = seat.table[best];
      if (best < 0 || (b !== null && b !== "neutral" && o.troops() < b.troops())) best = r;
    }
    const o = seat.table[best];
    if (best > 0 && o !== null && o !== "neutral" && o.troops() < p.troops()) return [T_ATTACK, best, 0, 2, 0];
  }
  return [T_NOOP, 0, 0, 0, 0];
}
