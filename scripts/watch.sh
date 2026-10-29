#!/usr/bin/env bash
# Regarder le modèle jouer une partie solo contre 7 Nations, à vitesse réelle.
#
#   scripts/watch.sh [checkpoint] [--difficulty Hard] [--speed 2|4] [--greedy] [--seed abc]
#
# 1. lance son propre serveur d'environnement, sur des ports à part pour ne
#    pas gêner un entraînement ou un serveur déjà ouvert : socket :5565, page
#    http://localhost:8095/live, API des replays sur :8787 si le port est libre ;
# 2. joue la partie avec agent/watch.py et ouvre la page /live ;
# 3. enregistre la partie dans runs/watch/ : la commande pour la revoir dans
#    le lecteur officiel (scripts/replay-viewer.sh <id>) est affichée à la fin.
#
# Sans checkpoint : runs/ppo2/ckpt/best_eval.pt, sinon runs/ppo1/ckpt/last.pt,
# sinon runs/bc1/ckpt/best.pt. Options : voir agent/watch.py --help.
# Variables facultatives : NODE_BIN (node 24), comme pour setup.sh ;
# WATCH_PORT et WATCH_HTTP_PORT pour d'autres ports.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${NODE_BIN:-}" ] && export PATH="$NODE_BIN:$PATH"

pids=()
cleanup() {
  for pid in ${pids[@]+"${pids[@]}"}; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

up() { nc -z 127.0.0.1 "$1" 2>/dev/null; }

PORT="${WATCH_PORT:-5565}"
HTTP_PORT="${WATCH_HTTP_PORT:-8095}"
API_PORT=8787
up "$API_PORT" && API_PORT=0  # déjà servie (serveur principal) : les records sont lus dans runs/ de toute façon
for p in "$PORT" "$HTTP_PORT"; do
  if up "$p"; then echo "Le port $p est déjà pris (une autre partie regardée ?)." >&2; exit 1; fi
done

echo "==> Serveur d'environnement (:$PORT, page /live sur :$HTTP_PORT)"
(cd "$ROOT/env" && exec npx tsx src/main.ts --port "$PORT" --http-port "$HTTP_PORT" --api-port "$API_PORT") &
pids+=($!)
for _ in $(seq 1 120); do up "$PORT" && up "$HTTP_PORT" && break; sleep 0.5; done
up "$PORT" || { echo "Le serveur ne répond pas sur :$PORT." >&2; exit 1; }

"$ROOT/agent/.venv/bin/python" "$ROOT/agent/watch.py" --port "$PORT" --live-url "http://localhost:$HTTP_PORT/live" "$@"

# Le serveur lancé ici reste ouvert : /live garde la dernière image, et le
# lecteur de replays a besoin de son API (:8787).
echo "Serveur toujours ouvert (/live, replays). Ctrl-C pour l'arrêter."
wait
