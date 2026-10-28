#!/usr/bin/env bash
# Lance une série d'évaluation en parallèle (agent à règles par défaut) qui
# alimente le tableau de bord http://localhost:8080/.
#
#   scripts/run-eval.sh <série> [options de agent/run_eval.py…]
#   scripts/run-eval.sh regles --minutes 10 --procs 4 --games-per-proc 2
#
# Pour une longue série, empêcher la veille du Mac : caffeinate -i scripts/run-eval.sh …
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -lt 1 ]; then
  echo "usage : $0 <série> [options de agent/run_eval.py]" >&2
  exit 2
fi
series="$1"
shift
exec "$ROOT/agent/.venv/bin/python" "$ROOT/agent/run_eval.py" --series "$series" "$@"
