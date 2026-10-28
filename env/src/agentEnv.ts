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
import { NationLabel, NationTap } from "./nationTap";
import { Buffers } from "./protocol";
import { appendMetrics, IndexEntry, RecordOptions, shouldRecord, writeRecord } from "./records";
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
  /** Enregistrement des parties (runs/<run>/games) ; absent : pas d'enregistrement. */
  record?: RecordOptions;
  /** Série de métriques : une ligne par siège fini dans runs/<series>/metrics.jsonl. */
  series?: string;
  /**
   * Données d'imitation : nombre de Nations observées. Leurs observations
   * (nation_0, nation_1…) et l'action de chacune pendant le pas (`labels`)
   * sont renvoyées à chaque pas. Voir nationTap.ts.
   */
  observeNations?: number;
}

/** Compteurs d'un siège sur l'épisode, pour les métriques. */
interface SeatStats {
  steps: number;
  reward: number;
  actions: Record<string, number>;
  /** Actions envoyées mais refusées par le moteur, par type. */
  refused: Record<string, number>;
  /** Actions hors masque (remplacées par « rien »). */
  invalid: number;
  /** Bateaux annulés car leur masque en cache était périmé. */
  stale: number;
}

export interface StepResult {
  state: GameState;
  /** Action de chaque Nation observée pendant ce pas (observeNations). */
  labels?: Record<string, NationLabel>;
  /** Nations observées encore en vie, dont l'observation est jointe. */
  observed?: string[];
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
  /** Bilan de chaque siège à sa fin (rang, victoire). */
  private outcomes = new Map<string, { rank: number; players: number; won: boolean }>();
  private recording: boolean;
  private stats = new Map<string, SeatStats>();
  private tap: NationTap | null = null;
  /** Entrée d'index une fois la partie écrite. */
  recorded: IndexEntry | null = null;

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
    if (!game.game.inSpawnPhase()) this.numSpawned = game.game.players().length;
    if ((config.observeNations ?? 0) > 0) {
      this.tap = new NationTap(game.game, config.observeNations!, game.config.seed);
    }
    this.recording = config.record !== undefined && shouldRecord(config.record);
    const byClient = new Map(this.seats.map((s) => [s.player.clientID(), s]));
    game.onExecutionCreated((intent, exec) => byClient.get(intent.clientID)?.attachExecution(exec));
  }

  static async create(config: EnvConfig): Promise<AgentEnv> {
    const game = await OpenFrontGame.create(config);
    return new AgentEnv(game, config);
  }

  reset(): StepResult {
    for (const s of this.seats) {
      this.prevShare.set(s.name, 0);
      this.stats.set(s.name, { steps: 0, reward: 0, actions: {}, refused: {}, invalid: 0, stale: 0 });
    }
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
    const sent = new Map<string, number>();
    for (const seat of this.seats) {
      if (!this.active.has(seat.name)) continue;
      seat.annotations = annotations[seat.name];
      let action = actions[seat.name] ?? [T_NOOP, 0, 0, 0, 0];
      if (g.game.inSpawnPhase() && action[0] !== T_SPAWN && this.config.autoSpawn !== false) {
        action = this.randomSpawn(seat);
      }
      if (action[0] !== T_NOOP && !seat.isValid(action)) invalid.add(seat.name);
      const intent = seat.toIntent(action);
      if (intent !== null) {
        intents.push(intent);
        sent.set(seat.name, action[0]);
      }
      const st = this.stats.get(seat.name)!;
      st.steps++;
      if (action[0] !== T_NOOP) {
        const t = ACTION_TYPES[action[0]] ?? String(action[0]);
        st.actions[t] = (st.actions[t] ?? 0) + 1;
        if (invalid.has(seat.name)) st.invalid++;
        else if (seat.lastStale) st.stale++;
      }
    }

    if (g.game.inSpawnPhase()) {
      g.finishSpawnPhase(intents);
      this.numSpawned = g.game.players().length;
    } else {
      g.step(ticks, intents);
    }
    // Décisions des Nations observées pendant ces ticks, traduites avec
    // l'observation précédente (avant de la recalculer dans result()).
    const labels: Record<string, NationLabel> = {};
    if (this.tap !== null) {
      for (const [seat, label] of this.tap.labels()) labels[seat.name] = label;
    }

    const accepted = new Map<string, boolean | null>();
    for (const seat of this.seats) {
      if (!this.active.has(seat.name)) continue;
      const ok = seat.resolvePending();
      accepted.set(seat.name, ok);
      const type = sent.get(seat.name);
      if (ok === false && type !== undefined) {
        const st = this.stats.get(seat.name)!;
        const t = ACTION_TYPES[type];
        st.refused[t] = (st.refused[t] ?? 0) + 1;
      }
    }

    for (const p of g.game.allPlayers()) {
      if (!p.isAlive() && !this.deathTick.has(p)) this.deathTick.set(p, g.game.ticks());
    }
    const out = this.result(accepted, invalid);
    if (this.tap !== null) out.labels = labels;
    return out;
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
        stale: seat.lastStale,
        last_action: seat.lastAction,
      };
      if (over) {
        const { rank, n } = this.rank(p);
        const won = g.getWinner() === p;
        reward += won ? 1 : -1 + (n - rank) / Math.max(1, n - 1);
        Object.assign(info, { rank, players: n, won });
        this.outcomes.set(seat.name, { rank, players: n, won });
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
      const st = this.stats.get(seat.name)!;
      st.reward += reward;
      if (over) {
        this.active.delete(seat.name);
        if (this.config.series) {
          appendMetrics(this.config.series, {
            date: new Date().toISOString(),
            gameID: this.game.gameID,
            seed: this.game.config.seed,
            map: this.game.config.map,
            agents: this.seats.length,
            nations: this.game.config.nations,
            agent: seat.name,
            won: info.won,
            rank: info.rank,
            players: info.players,
            share,
            ticks: g.ticks(),
            truncated,
            ...st,
          });
        }
      }
    }
    let observed: string[] | undefined;
    if (this.tap !== null && !g.inSpawnPhase()) {
      this.tap.attach((p, i) => new AgentSeat(1000 + i, `nation_${i}`, p, this.map, this.game.config.seed));
      observed = [];
      const nframe = frame ?? new SharedFrame(this.map);
      for (const seat of this.tap.seats) {
        if (!seat.player.isAlive()) continue;
        seat.computeMasks(nframe);
        buffers = { ...buffers, ...seat.observe(nframe, this.numSpawned) };
        observed.push(seat.name);
      }
    }
    if ((this.seats.length > 0 && this.active.size === 0) || state.done) this.finish();
    return {
      state,
      observed,
      agents: [...this.active],
      rewards,
      terminations,
      truncations,
      infos,
      buffers,
    };
  }

  /**
   * Fin de l'épisode (tous les sièges finis, partie terminée, ou partie
   * abandonnée : nouveau reset, connexion fermée) : écrit le record si cette
   * partie doit être enregistrée. Sans effet la seconde fois.
   */
  finish(): IndexEntry | null {
    const g = this.game;
    if (!this.recording || this.recorded !== null || g.turns.length === 0) return this.recorded;
    // Le moteur force un vainqueur au plus tard à 170 min de jeu.
    if (this.config.record!.playToEnd && !g.game.inSpawnPhase()) while (!g.done()) g.step(1000);
    const state = g.state();
    this.recorded = writeRecord(
      g.toRecord(),
      {
        gameID: g.gameID,
        date: new Date().toISOString(),
        map: g.config.map,
        size: g.config.size,
        seed: g.config.seed,
        nations: g.config.nations,
        agents: this.seats.map((s) => {
          const { rank, n } = this.rank(s.player);
          const o = this.outcomes.get(s.name) ?? { rank, players: n, won: false };
          return { name: s.name, clientID: s.player.clientID()!, ...o };
        }),
        winner: state.winner,
        finished: state.done,
        ticks: state.tick,
        turns: g.turns.length,
      },
      this.config.record!.run,
    );
    return this.recorded;
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
