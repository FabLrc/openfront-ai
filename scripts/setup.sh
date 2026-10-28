#!/usr/bin/env bash
# Prépare l'environnement de développement :
#   1. récupère le sous-module openfront/ (OpenFrontIO figé sur un commit) ;
#   2. installe ses dépendances (npm run inst = npm ci --ignore-scripts) ;
#   3. installe les dépendances du serveur d'environnement (env/) ;
#   4. crée le venv Python de l'agent (agent/.venv).
#
# Prérequis : Node 24 et npm 12 (exigés par OpenFrontIO), Python ≥ 3.11.
# Variables facultatives :
#   NODE_BIN  dossier contenant node 24 (ex. /opt/homebrew/opt/node@24/bin)
#   NPM_BIN   dossier contenant npm 12
#   PYTHON    interpréteur Python ≥ 3.11 (défaut : python3.12, sinon python3)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -n "${NODE_BIN:-}" ] && export PATH="$NODE_BIN:$PATH"
[ -n "${NPM_BIN:-}" ] && export PATH="$NPM_BIN:$PATH"

echo "node $(node --version), npm $(npm --version)"

echo "==> Sous-module openfront/"
git submodule update --init openfront

echo "==> Dépendances d'OpenFrontIO"
(cd openfront && npm run inst)

echo "==> Dépendances du serveur d'environnement"
(cd env && npm ci)

echo "==> Venv Python de l'agent"
PYTHON="${PYTHON:-$(command -v python3.12 || command -v python3)}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), "Python >= 3.11 requis"'
"$PYTHON" -m venv agent/.venv
agent/.venv/bin/pip install --quiet --upgrade pip
agent/.venv/bin/pip install --quiet -r agent/requirements.txt
agent/.venv/bin/pip install --quiet -e agent

echo "Terminé."
