/**
 * Environnement multi-agents : une partie, N sièges, des pas de n ticks.
 *
 * Récompense d'un siège :
 *   - à chaque pas : shaping × (variation de sa part des terres) ;
 *   - à la fin (mort du siège, fin ou troncature de la partie) :
 *     +1 en cas de victoire, sinon -1 + (n - rang) / (n - 1), entre -1 (dernier)
 *     et presque 0 (deuxième). Le rang classe les vivants par tuiles, puis les
 *     morts du plus tardif au plus précoce.
 */
import { StampedIntent } from "@openfront/engine-api/Schemas";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { Player } from "@openfront/engine/game/Game";
import { GameState, OpenFrontGame, ResetConfig } from "./game";
import { Buffers } from "./protocol";
import {
  ACTION_TYPES,
  AMOUNTS,
  Action,
  AgentSeat,
  BUILD_UNITS,
  CHANNELS,
  DEFAULT_SEAT_OPTIONS,
  PLAYER_FEATURES,
  SCALARS,
  MapIndex,
  SeatDisplay,
  SeatOptions,
  SharedFrame,
  T_NOOP,
  T_SPAWN,
} from "./seat";

export interface EnvConfig extends ResetConfig, Partial<SeatOptions> {
  /** Coefficient du terme intermédiaire (variation de part de territoire). */
  shaping?: number;
  /** Spawn aléatoire valide pour un siège qui ne choisit pas. */
  autoSpawn?: boolean;
  /** Troncature au-delà de ce tick (0 : jamais). */
  maxTicks?: number;
}

export interface StepResult {
  state: GameState;
  /** Agents encore actifs après ce pas. */
  agents: string[];
  rewards: Record<string, number>;
  terminations: Record<string, boolean>;
  truncations: Record<string, boolean>;
  infos: Record<string, Record<string, unknown>>;
  buffers: Buffers;
}

export class AgentEnv {
  readonly seats: AgentSeat[];
  readonly map: MapIndex;
  private active: Set<string>;
  private prevShare = new Map<string, number>();
  private deathTick = new Map<Player, number>();
  private numSpawned = 0;

  private constructor(
    readonly game: OpenFrontGame,
    readonly config: EnvConfig,
  ) {
    const opts: SeatOptions = { ...DEFAULT_SEAT_OPTIONS };
    for (const key of Object.keys(DEFAULT_SEAT_OPTIONS) as (keyof SeatOptions)[]) {
      if (config[key] !== undefined) opts[key] = config[key]!;
    }
    this.map = new MapIndex(game.game, opts);
    this.seats = game.seats.map(
      (p, i) => new AgentSeat(i, `agent_${i}`, p, this.map, game.config.seed),
    );
    this.active = new Set(this.seats.map((s) => s.name));
    const byClient = new Map(this.seats.map((s) => [s.player.clientID(), s]));
    game.onExecutionCreated((intent, exec) => byClient.get(intent.clientID)?.attachExecution(exec));
  }

  static async create(config: EnvConfig): Promise<AgentEnv> {
    const game = await OpenFrontGame.create(config);
    return new AgentEnv(game, config);
  }

  reset(): StepResult {
    for (const s of this.seats) this.prevShare.set(s.name, 0);
    return this.result(new Map(), new Set());
  }

  step(
    actions: Record<string, Action>,
    ticks = 10,
    annotations: Record<string, unknown> = {},
  ): StepResult {
    const g = this.game;
    const intents: StampedIntent[] = [];
    const invalid = new Set<string>();
    for (const seat of this.seats) {
      if (!this.active.has(seat.name)) continue;
      seat.annotations = annotations[seat.name];
      let action = actions[seat.name] ?? [T_NOOP, 0, 0, 0, 0];
      if (g.game.inSpawnPhase() && action[0] !== T_SPAWN && this.config.autoSpawn !== false) {
        action = this.randomSpawn(seat);
      }
      if (action[0] !== T_NOOP && !seat.isValid(action)) invalid.add(seat.name);
      const intent = seat.toIntent(action);
      if (intent !== null) intents.push(intent);
    }

    if (g.game.inSpawnPhase()) {
      g.finishSpawnPhase(intents);
      this.numSpawned = g.game.players().length;
    } else {
      g.step(ticks, intents);
    }

    const accepted = new Map<string, boolean | null>();
    for (const seat of this.seats) {
      if (this.active.has(seat.name)) accepted.set(seat.name, seat.resolvePending());
    }

    for (const p of g.game.allPlayers()) {
      if (!p.isAlive() && !this.deathTick.has(p)) this.deathTick.set(p, g.game.ticks());
    }
    return this.result(accepted, invalid);
  }

  private randomSpawn(seat: AgentSeat): Action {
    seat.computeMasks();
    const rows = seat.k + 1;
    const valid: number[] = [];
    for (let i = 0; i < rows; i++) if (seat.targetMask[T_SPAWN * rows + i]) valid.push(i);
    if (valid.length === 0) return [T_NOOP, 0, 0, 0, 0];
    const random = new PseudoRandom(simpleHash(this.game.config.seed) + seat.index * 104729);
    return [T_SPAWN, valid[random.nextInt(0, valid.length)], 0, 0, 0];
  }

  /** Rang (1 = premier) et nombre de participants. */
  rank(p: Player): { rank: number; n: number } {
    const winner = this.game.game.getWinner();
    const all = this.game.game.allPlayers();
    const score = (q: Player): [number, number] =>
      q === winner
        ? [2, 0]
        : q.isAlive()
          ? [1, q.numTilesOwned()]
          : [0, this.deathTick.get(q) ?? 0];
    const [a, b] = score(p);
    let better = 0;
    for (const q of all) {
      if (q === p) continue;
      const [qa, qb] = score(q);
      if (qa > a || (qa === a && qb > b)) better++;
    }
    return { rank: better + 1, n: all.length };
  }

  private result(
    accepted: Map<string, boolean | null>,
    invalid: Set<string>,
  ): StepResult {
    const g = this.game.game;
    const state = this.game.state();
    const shaping = this.config.shaping ?? 1;
    const maxTicks = this.config.maxTicks ?? 0;
    const truncated = maxTicks > 0 && g.ticks() >= maxTicks && !state.done;
    const frame = this.active.size > 0 ? new SharedFrame(this.map) : null;
    const rewards: Record<string, number> = {};
    const terminations: Record<string, boolean> = {};
    const truncations: Record<string, boolean> = {};
    const infos: Record<string, Record<string, unknown>> = {};
    let buffers: Buffers = {};
    const totalLand = this.map.totalLand;

    for (const seat of this.seats) {
      if (!this.active.has(seat.name)) continue;
      const p = seat.player;
      const share = p.numTilesOwned() / totalLand;
      let reward = shaping * (share - (this.prevShare.get(seat.name) ?? 0));
      this.prevShare.set(seat.name, share);
      const dead = !g.inSpawnPhase() && !p.isAlive();
      const over = state.done || dead || truncated;
      const info: Record<string, unknown> = {
        tick: g.ticks(),
        tiles: p.numTilesOwned(),
        troops: Math.round(p.troops()),
        gold: Number(p.gold()),
        accepted: accepted.get(seat.name) ?? null,
        invalid: invalid.has(seat.name),
        last_action: seat.lastAction,
      };
      if (over) {
        const { rank, n } = this.rank(p);
        const won = g.getWinner() === p;
        reward += won ? 1 : -1 + (n - rank) / Math.max(1, n - 1);
        Object.assign(info, { rank, players: n, won });
      }
      seat.computeMasks(frame!);
      buffers = { ...buffers, ...seat.observe(frame!, this.numSpawned) };
      if (g.inSpawnPhase()) {
        info.spawn_candidates = seat.spawnCandidates.map((t) => [t % this.map.width, Math.floor(t / this.map.width)]);
      }
      rewards[seat.name] = reward;
      terminations[seat.name] = over && !truncated;
      truncations[seat.name] = truncated;
      infos[seat.name] = info;
      if (over) this.active.delete(seat.name);
    }
    return {
      state,
      agents: [...this.active],
      rewards,
      terminations,
      truncations,
      infos,
      buffers,
    };
  }

  /** Description des espaces d'observation et d'action, envoyée au reset. */
  spec(): Record<string, unknown> {
    const o = this.map.opts;
    return {
      possible_agents: this.seats.map((s) => s.name),
      obs_width: o.obsWidth,
      obs_height: o.obsHeight,
      action_width: o.actionWidth,
      action_height: o.actionHeight,
      k: o.k,
      channels: CHANNELS,
      player_features: PLAYER_FEATURES,
      scalars: SCALARS,
      action_types: ACTION_TYPES,
      amounts: AMOUNTS,
      build_units: BUILD_UNITS,
      map_width: this.map.width,
      map_height: this.map.height,
    };
  }

  displays(): SeatDisplay[] {
    return this.seats.map((s) => s.display());
  }
}
