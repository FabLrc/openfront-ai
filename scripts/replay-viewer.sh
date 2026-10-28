#!/usr/bin/env bash
# Ouvre une partie enregistrée (runs/) dans le lecteur de replays officiel d'OpenFront.
#
#   scripts/replay-viewer.sh [gameID]
#
# 1. lance le serveur d'environnement s'il ne tourne pas : il sert les records
#    sur http://localhost:8787/game/<id>, l'adresse que le client de
#    développement interroge par défaut (getApiBase, sans API_DOMAIN) ;
# 2. lance le client OpenFront en mode développement (vite, port 9000), sans
#    rien modifier dans le sous-module ;
# 3. ouvre http://localhost:9000/#replay-viewer=<gameID> (sans identifiant :
#    la liste des parties, http://localhost:8080/games).
#
# Ctrl-C arrête ce que le script a lancé. Variables facultatives : comme pour
# setup.sh, NODE_BIN (node 24) et NPM_BIN (npm 12) ; NO_OPEN=1 pour ne pas
# ouvrir le navigateur.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${NODE_BIN:-}" ] && export PATH="$NODE_BIN:$PATH"
[ -n "${NPM_BIN:-}" ] && export PATH="$NPM_BIN:$PATH"

pids=()
cleanup() {
  for pid in ${pids[@]+"${pids[@]}"}; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

# localhost : vite écoute parfois seulement en IPv6 (::1).
up() { nc -z localhost "$1" 2>/dev/null; }
wait_for() {
  for _ in $(seq 1 120); do up "$1" && return 0; sleep 0.5; done
  echo "Le port $1 ne répond pas." >&2
  exit 1
}

if ! up 8787; then
  echo "==> Serveur d'environnement (API des replays sur :8787)"
  (cd "$ROOT/env" && exec npx tsx src/main.ts) &
  pids+=($!)
  wait_for 8787
fi

if ! up 9000; then
  echo "==> Client OpenFront (vite, :9000)"
  (cd "$ROOT/openfront" && exec npm run start:client -- --strictPort) &
  pids+=($!)
  wait_for 9000
fi

if [ $# -ge 1 ]; then
  url="http://localhost:9000/#replay-viewer=$1"
else
  url="http://localhost:8080/games"
fi
echo "==> $url"
if [ -z "${NO_OPEN:-}" ]; then
  open "$url" 2>/dev/null || xdg-open "$url" 2>/dev/null || true
fi

if [ ${#pids[@]} -gt 0 ]; then
  echo "Ctrl-C pour arrêter."
  wait
fi
