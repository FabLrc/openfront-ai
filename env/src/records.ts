/**
 * Parties enregistrées : runs/<run>/games/<gameID>.json.gz (GameRecord au
 * format officiel, compressé) et runs/<run>/index.jsonl (une ligne par partie).
 */
import { replacer } from "@openfront/shared/SharedUtil";
import { GameRecord } from "@openfront/shared/WireSchemas";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

export const RUNS_DIR = process.env.OPENFRONT_RUNS_DIR
  ? path.resolve(process.env.OPENFRONT_RUNS_DIR)
  : path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../runs");

export interface RecordOptions {
  /** Nom de la série (dossier runs/<run>). */
  run: string;
  /** N'enregistrer qu'une partie sur N (1 : toutes). */
  every?: number;
  /**
   * Quand l'épisode finit avant la partie (mort des agents), jouer la suite
   * sans eux jusqu'au vainqueur avant d'écrire le record.
   */
  playToEnd?: boolean;
}

export interface IndexEntry {
  gameID: string;
  run: string;
  date: string;
  map: string;
  size: string;
  seed: string;
  nations: number;
  agents: { name: string; clientID: string; rank: number | null; players: number; won: boolean }[];
  /** Nom du vainqueur, null si la partie s'est arrêtée avant (par ex. mort de l'agent). */
  winner: string | null;
  finished: boolean;
  ticks: number;
  turns: number;
  file: string;
}

const safe = (name: string) => name.replace(/[^\w.-]+/g, "_");

/** Nombre de parties vues par série, pour « une sur N ». */
const seen = new Map<string, number>();

/** Vrai si la prochaine partie de cette série doit être enregistrée. */
export function shouldRecord(opts: RecordOptions): boolean {
  const n = seen.get(opts.run) ?? 0;
  seen.set(opts.run, n + 1);
  return n % Math.max(1, opts.every ?? 1) === 0;
}

export function writeRecord(record: GameRecord, entry: Omit<IndexEntry, "file" | "run">, run: string): IndexEntry {
  const dir = path.join(RUNS_DIR, safe(run));
  const gamesDir = path.join(dir, "games");
  fs.mkdirSync(gamesDir, { recursive: true });
  const file = path.join(gamesDir, `${safe(record.info.gameID)}.json.gz`);
  // Bigints (statistiques) en chaînes décimales, comme src/server/Archive.ts.
  fs.writeFileSync(file, zlib.gzipSync(JSON.stringify(record, replacer)));
  const full: IndexEntry = { ...entry, run, file: path.relative(RUNS_DIR, file) };
  fs.appendFileSync(path.join(dir, "index.jsonl"), JSON.stringify(full) + "\n");
  return full;
}

/** Ajoute une ligne à runs/<series>/metrics.jsonl (une écriture par ligne : sûr entre processus). */
export function appendMetrics(series: string, line: Record<string, unknown>): void {
  const dir = path.join(RUNS_DIR, safe(series));
  fs.mkdirSync(dir, { recursive: true });
  fs.appendFileSync(path.join(dir, "metrics.jsonl"), JSON.stringify(line) + "\n");
}

export function readRecordFile(file: string): GameRecord {
  const raw = fs.readFileSync(file);
  const text = file.endsWith(".gz") ? zlib.gunzipSync(raw).toString("utf8") : raw.toString("utf8");
  return JSON.parse(text) as GameRecord;
}

/** Toutes les entrées de toutes les séries, la plus récente en premier. */
export function listGames(): IndexEntry[] {
  if (!fs.existsSync(RUNS_DIR)) return [];
  const out: IndexEntry[] = [];
  for (const run of fs.readdirSync(RUNS_DIR)) {
    const index = path.join(RUNS_DIR, run, "index.jsonl");
    if (!fs.existsSync(index)) continue;
    for (const line of fs.readFileSync(index, "utf8").split("\n")) {
      if (line.trim() === "") continue;
      try {
        out.push(JSON.parse(line) as IndexEntry);
      } catch {
        // ligne tronquée (écriture interrompue) : ignorée
      }
    }
  }
  return out.sort((a, b) => b.date.localeCompare(a.date));
}

/** Chemin du record d'une partie (la plus récente si l'identifiant revient dans plusieurs séries). */
export function findRecordFile(gameID: string): string | null {
  const entry = listGames().find((e) => e.gameID === gameID);
  if (entry === undefined) return null;
  const file = path.join(RUNS_DIR, entry.file);
  return fs.existsSync(file) ? file : null;
}
