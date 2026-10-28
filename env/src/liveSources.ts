/**
 * Pages /live des processus d'une série parallèle (VecOpenFrontEnv) : chaque
 * processus en sert une sur son port et s'inscrit dans runs/live-sources.json.
 * La page /live du serveur principal propose de les suivre.
 */
import fs from "node:fs";
import path from "node:path";
import { RUNS_DIR } from "./records";

export interface LiveSource {
  label: string;
  http_port: number;
  env_port: number;
  pid: number;
}

const alive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

/** Sources encore vivantes, sans ce serveur lui-même. */
export function liveSources(ownHttpPort: number): LiveSource[] {
  const file = path.join(RUNS_DIR, "live-sources.json");
  if (!fs.existsSync(file)) return [];
  try {
    const entries = JSON.parse(fs.readFileSync(file, "utf8")) as LiveSource[];
    return entries.filter((e) => e.http_port !== ownHttpPort && typeof e.pid === "number" && alive(e.pid));
  } catch {
    return [];
  }
}
