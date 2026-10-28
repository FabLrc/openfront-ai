#!/usr/bin/env bash
# Entraînement PPO par segments reprenables, avec une évaluation après chaque
# segment (40 parties à graines fixes contre 7 Nations Medium puis Hard),
# écrite dans le train.jsonl de la série (courbes eval_* du tableau de bord).
#
#   scripts/train-ppo.sh <série> [heures=4] [minutes par segment=30]
#
# Sur macOS, caffeinate -i empêche la veille pendant le script (ailleurs, voir
# docs/pc-windows.md). On peut l'interrompre et le relancer : l'entraînement
# reprend au dernier checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/agent/.venv/bin/python"
series="${1:?usage : $0 <série> [heures] [minutes par segment]}"
hours="${2:-4}"
segment="${3:-30}"
end=$(( $(date +%s) + ${hours%.*} * 3600 ))
awake=()
command -v caffeinate > /dev/null && awake=(caffeinate -i)

while [ "$(date +%s)" -lt "$end" ]; do
  ${awake[@]+"${awake[@]}"} "$PY" "$ROOT/agent/train_ppo.py" --series "$series" --minutes "$segment"
  ckpt="$ROOT/runs/$series/ckpt/last.pt"
  step=$("$PY" -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['progress']['decisions'])" "$ckpt")
  for diff in Medium Hard; do
    tag="$series-eval-$step-$(echo "$diff" | tr '[:upper:]' '[:lower:]')"
    rm -rf "$ROOT/runs/$tag"
    ${awake[@]+"${awake[@]}"} "$PY" "$ROOT/agent/run_eval.py" --series "$tag" --agent policy --checkpoint "$ckpt" \
      --episodes 40 --seed evalfix --difficulty "$diff" --no-dashboard --base-port 6700 \
      --log-to "$series" --log-step "$step" | tail -2
  done
done
