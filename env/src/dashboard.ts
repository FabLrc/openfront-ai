/**
 * Tableau de bord (http://localhost:8080/) : métriques des séries de runs/.
 *
 * Chaque série (runs/<série>/) peut contenir :
 *   metrics.jsonl     une ligne par siège d'agent fini (écrite par l'environnement) ;
 *   throughput.jsonl  le débit par intervalle (écrit par VecOpenFrontEnv) ;
 *   train.jsonl       des métriques libres d'entraînement (perte, entropie…) :
 *                     tout champ numérique devient une courbe, en fonction de
 *                     `step` s'il existe.
 */
import fs from "node:fs";
import path from "node:path";
import { RUNS_DIR } from "./records";

const FILES = ["metrics.jsonl", "throughput.jsonl", "train.jsonl"] as const;
const MAX_LINES = 20000;

export interface SeriesInfo {
  name: string;
  updated: string;
  episodes: number;
}

/** Séries ayant des métriques, la plus récemment mise à jour en premier. */
export function listSeries(): SeriesInfo[] {
  if (!fs.existsSync(RUNS_DIR)) return [];
  const out: (SeriesInfo & { mtime: number })[] = [];
  for (const name of fs.readdirSync(RUNS_DIR)) {
    let mtime = 0;
    for (const f of FILES) {
      const file = path.join(RUNS_DIR, name, f);
      if (fs.existsSync(file)) mtime = Math.max(mtime, fs.statSync(file).mtimeMs);
    }
    if (mtime === 0) continue;
    const metrics = path.join(RUNS_DIR, name, "metrics.jsonl");
    const episodes = fs.existsSync(metrics) ? countLines(metrics) : 0;
    out.push({ name, updated: new Date(mtime).toISOString(), episodes, mtime });
  }
  return out.sort((a, b) => b.mtime - a.mtime).map(({ mtime: _, ...s }) => s);
}

function countLines(file: string): number {
  const text = fs.readFileSync(file, "utf8");
  let n = 0;
  for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10) n++;
  return n;
}

function readJsonl(file: string): Record<string, unknown>[] {
  if (!fs.existsSync(file)) return [];
  const lines = fs.readFileSync(file, "utf8").split("\n").filter((l) => l.trim() !== "");
  const out: Record<string, unknown>[] = [];
  for (const line of lines.slice(-MAX_LINES)) {
    try {
      out.push(JSON.parse(line));
    } catch {
      // ligne en cours d'écriture
    }
  }
  return out;
}

export function readSeries(name: string) {
  const dir = path.join(RUNS_DIR, path.basename(name));
  return {
    series: path.basename(name),
    episodes: readJsonl(path.join(dir, "metrics.jsonl")),
    throughput: readJsonl(path.join(dir, "throughput.jsonl")),
    train: readJsonl(path.join(dir, "train.jsonl")),
  };
}

export function renderDashboard(): string {
  return DASHBOARD_HTML;
}

const DASHBOARD_HTML = `<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenFront IA — tableau de bord</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root { --bg: #0f1419; --panel: #171e26; --text: #e6e9ee; --muted: #8a96a3; --line: #2a3440; --accent: #4da3ff; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, -apple-system, sans-serif; }
  header { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  a { color: var(--accent); }
  select, input { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; font: inherit; }
  .muted { color: var(--muted); }
  main { padding: 16px; }
  .tiles { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .tile { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
  .tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .tile .k { color: var(--muted); font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 0 0 8px; }
  .card canvas { width: 100%; height: 220px; }
  .empty { color: var(--muted); padding: 40px 0; text-align: center; }
  @media (max-width: 500px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>OpenFront IA — tableau de bord</h1>
  <label class="muted">Série <select id="series"></select></label>
  <label class="muted">Fenêtre glissante <input id="window" type="number" min="1" value="50" style="width:70px"> épisodes</label>
  <span class="muted" id="updated"></span>
  <span style="margin-left:auto"><a href="/live">Partie en direct</a> · <a href="/games">Parties enregistrées</a></span>
</header>
<main>
  <div class="tiles" id="tiles"></div>
  <div class="grid" id="grid">
    <div class="card"><h2>Victoires et rang (glissants)</h2><canvas id="c-win"></canvas></div>
    <div class="card"><h2>Récompense par épisode</h2><canvas id="c-reward"></canvas></div>
    <div class="card"><h2>Durée des épisodes (ticks)</h2><canvas id="c-ticks"></canvas></div>
    <div class="card"><h2>Débit</h2><canvas id="c-speed"></canvas></div>
  </div>
  <h2 class="muted" style="font-size:13px;margin:20px 0 8px" id="train-title" hidden>ENTRAÎNEMENT</h2>
  <div class="grid" id="train"></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
Chart.defaults.color = css("--muted");
Chart.defaults.borderColor = css("--line");
Chart.defaults.animation = false;
const COLORS = ["#4da3ff", "#f5a623", "#6fd38a", "#ff7b7b", "#b88cff", "#4fd1c5"];
const charts = {};

function rolling(values, w) {
  const out = []; let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= w) sum -= values[i - w];
    out.push(sum / Math.min(i + 1, w));
  }
  return out;
}

function lineChart(id, datasets, opts = {}) {
  const cfg = {
    type: "line",
    data: { datasets: datasets.map((d, i) => ({ borderColor: d.color ?? COLORS[i % COLORS.length], backgroundColor: d.color ?? COLORS[i % COLORS.length], pointRadius: d.points ? 1.5 : 0, showLine: !d.points, borderWidth: 1.5, ...d })) },
    options: {
      responsive: true, maintainAspectRatio: false, parsing: false,
      plugins: { legend: { display: datasets.length > 1, labels: { boxWidth: 10 } } },
      scales: {
        x: { type: "linear", title: { display: true, text: opts.x ?? "épisode" } },
        y: { title: { display: !!opts.y, text: opts.y } , ...(opts.yScale ?? {}) },
        ...(opts.extraScales ?? {}),
      },
    },
  };
  if (charts[id]) { charts[id].data = cfg.data; charts[id].options = cfg.options; charts[id].update(); }
  else { const canvas = $(id); canvas.parentElement.style.height = "260px"; charts[id] = new Chart(canvas, cfg); }
}

const fmt = (n, d = 0) => Number.isFinite(n) ? n.toLocaleString("fr-FR", { maximumFractionDigits: d, minimumFractionDigits: d }) : "—";

function render(data) {
  const w = Math.max(1, parseInt($("window").value, 10) || 50);
  const eps = data.episodes;
  const xs = eps.map((_, i) => i + 1);
  const won = eps.map((e) => (e.won ? 1 : 0));
  const rank = eps.map((e) => e.rank ?? NaN);
  const winRate = rolling(won, w).map((v) => v * 100);
  const meanRank = rolling(rank, w);
  const pts = (ys) => ys.map((y, i) => ({ x: xs[i], y }));

  lineChart("c-win", [
    { label: "victoires % (glissant)", data: pts(winRate), yAxisID: "y" },
    { label: "rang moyen (glissant)", data: pts(meanRank), yAxisID: "y2" },
  ], { y: "% de victoires", yScale: { min: 0, max: 100 }, extraScales: { y2: { position: "right", reverse: true, title: { display: true, text: "rang" }, grid: { drawOnChartArea: false } } } });

  const reward = eps.map((e) => e.reward ?? 0);
  lineChart("c-reward", [
    { label: "épisode", data: pts(reward), points: true, color: "#4da3ff66" },
    { label: "moyenne glissante", data: pts(rolling(reward, w)), color: "#f5a623" },
  ]);
  const ticks = eps.map((e) => e.ticks ?? 0);
  lineChart("c-ticks", [
    { label: "épisode", data: pts(ticks), points: true, color: "#6fd38a66" },
    { label: "moyenne glissante", data: pts(rolling(ticks, w)), color: "#f5a623" },
  ]);

  const tp = data.throughput;
  const t0 = tp.length ? Date.parse(tp[0].date) : 0;
  const tx = (r) => (Date.parse(r.date) - t0) / 60000;
  lineChart("c-speed", [
    { label: "décisions/s", data: tp.map((r) => ({ x: tx(r), y: r.decisions_per_s })), yAxisID: "y" },
    { label: "ticks/s", data: tp.map((r) => ({ x: tx(r), y: r.ticks_per_s })), yAxisID: "y2" },
  ], { x: "minutes", y: "décisions/s", extraScales: { y2: { position: "right", title: { display: true, text: "ticks/s" }, grid: { drawOnChartArea: false } } } });

  // Métriques d'entraînement : une courbe par champ numérique.
  const train = data.train;
  const keys = [...new Set(train.flatMap((r) => Object.keys(r).filter((k) => typeof r[k] === "number" && k !== "step")))];
  $("train-title").hidden = keys.length === 0;
  for (const k of keys) {
    const id = "t-" + k.replace(/\\W+/g, "_");
    if (!$(id)) {
      const card = document.createElement("div");
      card.className = "card";
      card.innerHTML = '<h2></h2><canvas id="' + id + '"></canvas>';
      card.querySelector("h2").textContent = k;
      $("train").appendChild(card);
    }
    lineChart(id, [{ label: k, data: train.filter((r) => typeof r[k] === "number").map((r, i) => ({ x: r.step ?? i, y: r[k] })) }], { x: "step" });
  }

  const last = eps.slice(-w);
  const refused = eps.reduce((a, e) => a + Object.values(e.refused ?? {}).reduce((x, y) => x + y, 0), 0);
  const actions = eps.reduce((a, e) => a + Object.values(e.actions ?? {}).reduce((x, y) => x + y, 0), 0);
  const lastTp = tp[tp.length - 1];
  const tiles = [
    ["épisodes", fmt(eps.length)],
    [\`victoires (\${last.length} derniers)\`, fmt(last.length ? (100 * last.filter((e) => e.won).length) / last.length : NaN, 1) + " %"],
    [\`rang moyen (\${last.length} derniers)\`, fmt(last.reduce((a, e) => a + (e.rank ?? 0), 0) / last.length, 2)],
    ["part de territoire finale", fmt(100 * last.reduce((a, e) => a + (e.share ?? 0), 0) / last.length, 1) + " %"],
    ["actions refusées", fmt(actions ? (100 * refused) / actions : NaN, 2) + " %"],
    ["décisions/s", lastTp ? fmt(lastTp.decisions_per_s) : "—"],
    ["ticks/s", lastTp ? fmt(lastTp.ticks_per_s) : "—"],
  ];
  $("tiles").innerHTML = tiles.map(([k, v]) => '<div class="tile"><div class="v">' + v + '</div><div class="k">' + k + "</div></div>").join("");
  $("updated").textContent = "mis à jour " + new Date().toLocaleTimeString("fr-FR");
}

async function refreshSeries() {
  const list = await (await fetch("/api/series")).json();
  const sel = $("series");
  const current = sel.value || new URLSearchParams(location.search).get("series");
  sel.innerHTML = list.map((s) => '<option value="' + s.name + '">' + s.name + " (" + s.episodes + ")</option>").join("");
  if (current && list.some((s) => s.name === current)) sel.value = current;
  if (list.length === 0) $("grid").innerHTML = '<div class="empty">Aucune série dans runs/. Lancez scripts/run-eval.sh.</div>';
}

async function refresh() {
  try {
    await refreshSeries();
    const name = $("series").value;
    if (name) render(await (await fetch("/api/metrics?series=" + encodeURIComponent(name))).json());
  } catch (e) { $("updated").textContent = "erreur : " + e; }
}
$("series").addEventListener("change", refresh);
$("window").addEventListener("change", refresh);
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>`;
