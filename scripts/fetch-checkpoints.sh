#!/usr/bin/env bash
# Télécharge les modèles et le run PPO publiés dans une release GitHub
# (runs/ n'est pas versionné) et les décompresse dans runs/.
#
#   scripts/fetch-checkpoints.sh [tag=ppo1-reprise]
#
# Utilise gh s'il est installé et connecté (dépôt privé compris), sinon curl
# (dépôt public seulement). Les fichiers existants de runs/ sont remplacés.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="${1:-ppo1-reprise}"
asset="$tag.tgz"
repo="FabLrc/openfront-ai"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if command -v gh > /dev/null && gh auth status > /dev/null 2>&1; then
  gh release download "$tag" --repo "$repo" --pattern "$asset" --dir "$tmp"
else
  curl -fL "https://github.com/$repo/releases/download/$tag/$asset" -o "$tmp/$asset"
fi
tar xzf "$tmp/$asset" -C "$ROOT"
echo "Décompressé dans $ROOT/runs :"
tar tzf "$tmp/$asset" | grep -E 'ckpt/(best|last)\.pt$'
