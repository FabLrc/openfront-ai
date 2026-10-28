/**
 * Données d'imitation : ce que décident des Nations, traduit dans notre espace
 * d'actions factorisé.
 *
 * Sans modifier le moteur : on enveloppe le tick() des NationExecution
 * observées et game.addExecution(). Toute exécution ajoutée pendant le tick
 * d'une Nation observée est une de ses décisions (attaque, bateau,
 * construction, demande d'alliance…). Les Nations acceptent ou rompent une
 * alliance sans passer par une exécution : on compare ses alliés avant et
 * après son tick.
 */
import { AllianceRequestExecution } from "@openfront/engine/execution/alliance/AllianceRequestExecution";
import { AttackExecution } from "@openfront/engine/execution/AttackExecution";
import { ConstructionExecution } from "@openfront/engine/execution/ConstructionExecution";
import { NationExecution } from "@openfront/engine/execution/NationExecution";
import { TransportShipExecution } from "@openfront/engine/execution/TransportShipExecution";
import { Execution, Game, Player } from "@openfront/engine/game/Game";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { UnitType } from "@openfront/engine-api/game/GameTypes";
import {
  Action,
  AgentSeat,
  AMOUNTS,
  BUILD_UNITS,
  T_ALLIANCE,
  T_ATTACK,
  T_BOAT,
  T_BREAK,
  T_BUILD,
  T_NOOP,
} from "./seat";

/** Décision brute d'une Nation pendant un pas. */
type Captured =
  | { kind: "attack"; exec: AttackExecution; troopsBefore: number }
  | { kind: "boat"; exec: TransportShipExecution; troopsBefore: number }
  | { kind: "build"; exec: ConstructionExecution }
  | { kind: "alliance"; other: Player }
  | { kind: "break"; other: Player }
  | { kind: "other"; name: string };

export interface NationLabel {
  /** Action dans notre espace (« rien » si la Nation n'a rien fait de traduisible). */
  action: Action;
  /** Nature de la décision retenue : noop, attack, boat, build, alliance, break, ou le nom d'une exécution non traduite. */
  kind: string;
  /** Vrai si l'action retenue est une décision traduite (pas « rien »). */
  mapped: boolean;
  /** Vrai si l'action retenue respecte nos masques au moment de l'observation. */
  valid: boolean;
  /** Toutes les décisions du pas, traduites ou non (pour les statistiques). */
  all: string[];
}

const nearestAmount = (ratio: number) => {
  let best = 0;
  for (let i = 1; i < AMOUNTS.length; i++) {
    if (Math.abs(AMOUNTS[i] - ratio) < Math.abs(AMOUNTS[best] - ratio)) best = i;
  }
  return best;
};

export class NationTap {
  /** Sièges observés (même observation et mêmes masques que pour un agent). */
  readonly seats: AgentSeat[] = [];
  private captured = new Map<Player, Captured[]>();
  private current: Player | null = null;
  private wrapped = false;

  constructor(
    private readonly game: Game,
    private readonly count: number,
    private readonly seed: string,
  ) {
    const g = game as unknown as { addExecution: (...e: Execution[]) => void };
    const add = g.addExecution.bind(game);
    g.addExecution = (...execs: Execution[]) => {
      const p = this.current;
      if (p !== null) {
        for (const e of execs) this.capture(p, e);
      }
      add(...execs);
    };
  }

  /**
   * Après la phase de spawn : choisit les Nations observées (au hasard,
   * selon la graine) et enveloppe leur tick. `makeSeat` crée leur siège.
   */
  attach(makeSeat: (p: Player, i: number) => AgentSeat): void {
    if (this.wrapped) return;
    this.wrapped = true;
    const impl = this.game as unknown as { execs: Execution[]; unInitExecs: Execution[] };
    const nations = [...impl.execs, ...impl.unInitExecs].filter(
      (e): e is NationExecution => e instanceof NationExecution,
    );
    const players = nations.map((e) => (e as unknown as { player: Player | null }).player);
    const random = new PseudoRandom(simpleHash(this.seed) + 31337);
    const order = nations.map((_, i) => i).filter((i) => players[i]?.isAlive());
    for (let i = order.length - 1; i > 0; i--) {
      const j = random.nextInt(0, i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    for (const i of order.slice(0, this.count)) {
      const exec = nations[i];
      const p = players[i]!;
      this.seats.push(makeSeat(p, this.seats.length));
      this.captured.set(p, []);
      const tick = exec.tick.bind(exec);
      exec.tick = (t: number) => {
        const allies = new Set(p.allies());
        this.current = p;
        try {
          tick(t);
        } finally {
          this.current = null;
        }
        const after = new Set(p.allies());
        for (const a of after) if (!allies.has(a)) this.captured.get(p)!.push({ kind: "alliance", other: a });
        for (const a of allies) if (!after.has(a)) this.captured.get(p)!.push({ kind: "break", other: a });
      };
    }
  }

  private capture(p: Player, e: Execution): void {
    const list = this.captured.get(p);
    if (list === undefined) return;
    if (e instanceof AttackExecution) list.push({ kind: "attack", exec: e, troopsBefore: p.troops() });
    else if (e instanceof TransportShipExecution) list.push({ kind: "boat", exec: e, troopsBefore: p.troops() });
    else if (e instanceof ConstructionExecution) list.push({ kind: "build", exec: e });
    else if (e instanceof AllianceRequestExecution) {
      const id = (e as unknown as { recipientID: string }).recipientID;
      if (this.game.hasPlayer(id)) list.push({ kind: "alliance", other: this.game.player(id) });
    } else list.push({ kind: "other", name: e.constructor.name });
  }

  /**
   * Étiquette de chaque siège observé pour le pas qui vient d'être joué,
   * traduite avec la table et les masques de l'observation précédente (celle
   * que la Nation « voyait » en décidant). Vide la mémoire du pas.
   */
  labels(): Map<AgentSeat, NationLabel> {
    const out = new Map<AgentSeat, NationLabel>();
    for (const seat of this.seats) {
      const list = this.captured.get(seat.player) ?? [];
      this.captured.set(seat.player, []);
      const translated = list.map((c) => ({ c, action: this.translate(seat, c) }));
      const all = list.map((c) => (c.kind === "other" ? c.name : c.kind));
      // La première décision traduisible et autorisée par nos masques, sinon
      // la première traduisible, sinon « rien ».
      const pick =
        translated.find((t) => t.action !== null && seat.isValid(t.action)) ??
        translated.find((t) => t.action !== null);
      if (pick === undefined || pick.action === null) {
        out.set(seat, {
          action: [T_NOOP, 0, 0, 0, 0],
          kind: list.length > 0 ? `non traduite:${all[0]}` : "noop",
          mapped: false,
          valid: list.length === 0,
          all,
        });
        continue;
      }
      out.set(seat, { action: pick.action, kind: pick.c.kind, mapped: true, valid: seat.isValid(pick.action), all });
    }
    return out;
  }

  private translate(seat: AgentSeat, c: Captured): Action | null {
    const rowOf = (o: Player | null) => (o === null ? 0 : seat.table.indexOf(o));
    const cellOf = (t: number) => seat.map.tileActCell[t];
    switch (c.kind) {
      case "attack": {
        const e = c.exec as unknown as { startTroops: number | null; _targetID: string | null };
        const target = e._targetID === null ? null : this.game.player(e._targetID);
        const row = rowOf(target);
        if (row < 0) return null;
        const troops =
          e.startTroops ?? this.game.config().attackAmount(seat.player, target ?? this.game.terraNullius());
        return [T_ATTACK, row, 0, nearestAmount(troops / Math.max(1, c.troopsBefore)), 0];
      }
      case "boat": {
        const e = c.exec as unknown as { ref: number; troops: number | null };
        const troops = e.troops ?? 0;
        return [T_BOAT, 0, cellOf(e.ref), nearestAmount(troops / Math.max(1, c.troopsBefore)), 0];
      }
      case "build": {
        const e = c.exec as unknown as { constructionType: UnitType; tile: number };
        const unit = BUILD_UNITS.indexOf(e.constructionType as (typeof BUILD_UNITS)[number]);
        if (unit < 0) return null; // bombe, navire de guerre : hors de notre espace d'actions
        return [T_BUILD, 0, cellOf(e.tile), 0, unit];
      }
      case "alliance": {
        const row = rowOf(c.other);
        return row <= 0 ? null : [T_ALLIANCE, row, 0, 0, 0];
      }
      case "break": {
        const row = rowOf(c.other);
        return row <= 0 ? null : [T_BREAK, row, 0, 0, 0];
      }
      default:
        return null;
    }
  }
}
