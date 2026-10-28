/** Page http://localhost:8080/games : les parties enregistrées, avec un lien vers le lecteur officiel. */
import { IndexEntry, listGames } from "./records";

const esc = (s: unknown) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

function agentCell(e: IndexEntry): string {
  if (e.agents.length === 0) return "—";
  return e.agents
    .map((a) => (a.won ? "victoire" : a.rank === null ? "?" : `${a.rank}/${a.players}`))
    .join(", ");
}

export function renderGamesPage(viewerUrl: string): string {
  const games = listGames();
  const rows = games
    .map((e) => {
      const date = new Date(e.date);
      const minutes = Math.floor(e.ticks / 600);
      const seconds = Math.floor((e.ticks % 600) / 10);
      return `<tr>
        <td>${esc(date.toLocaleString("fr-FR"))}</td>
        <td>${esc(e.run)}</td>
        <td>${esc(e.map)} <span class="muted">${esc(e.size)}</span></td>
        <td>${esc(e.winner ?? (e.finished ? "—" : "partie non finie"))}</td>
        <td>${esc(agentCell(e))}</td>
        <td>${minutes} min ${String(seconds).padStart(2, "0")} s <span class="muted">(${e.ticks.toLocaleString("fr-FR")} ticks)</span></td>
        <td class="muted">${esc(e.seed)}</td>
        <td><a href="${esc(viewerUrl)}/#replay-viewer=${encodeURIComponent(e.gameID)}" target="_blank">Voir le replay</a></td>
      </tr>`;
    })
    .join("");
  return `<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parties enregistrées</title>
<style>
  :root { --bg: #0f1419; --panel: #171e26; --text: #e6e9ee; --muted: #8a96a3; --line: #2a3440; --accent: #4da3ff; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, -apple-system, sans-serif; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--line); display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; }
  h1 { font-size: 16px; margin: 0; }
  main { padding: 16px; overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  .muted { color: var(--muted); }
  a { color: var(--accent); }
  p.hint { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Parties enregistrées</h1>
  <span class="muted">${games.length} partie(s) dans runs/</span>
  <a href="/live">Partie en direct</a>
</header>
<main>
  ${
    games.length === 0
      ? `<p class="hint">Aucune partie. Lancez par exemple <code>agent/.venv/bin/python agent/evaluate.py --record essai</code>.</p>`
      : `<table>
    <thead><tr><th>Date</th><th>Série</th><th>Carte</th><th>Vainqueur</th><th>Rang de l'agent</th><th>Durée de jeu</th><th>Graine</th><th></th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p class="hint">Le lecteur officiel doit tourner : <code>scripts/replay-viewer.sh</code> (client OpenFront sur ${esc(viewerUrl)}).</p>`
  }
</main>
</body>
</html>`;
}
