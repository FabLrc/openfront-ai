# Entraîner sur un PC Windows (WSL2 + GPU NVIDIA)

Sur le PC, tout tourne dans WSL2 (Ubuntu) : les scripts sont en bash et le
jeu tourne sous Node. Le GPU NVIDIA est utilisable depuis WSL2 avec le seul
pilote Windows, sans pilote Linux à installer. PyTorch choisit CUDA tout seul
(`best_device()`), et sur CUDA toutes les couches du réseau sont entraînées.
Sur le GPU du Mac, les deux premières convolutions restent gelées.

## 1. WSL2 et pilote NVIDIA

Dans un PowerShell administrateur :

```powershell
wsl --install -d Ubuntu-24.04
```

Redémarrer, ouvrir « Ubuntu », créer l'utilisateur. Installer le pilote
NVIDIA récent pour Windows (GeForce Experience ou nvidia.com). Puis, dans
Ubuntu, `nvidia-smi` doit afficher la RTX 3080.

Le dépôt doit être dans le système de fichiers Linux (`~/…`), pas dans
`/mnt/c/…` : les accès disque y sont bien plus rapides.

## 2. Outils

Dans Ubuntu :

```bash
sudo apt update && sudo apt install -y git build-essential python3.12 python3.12-venv netcat-openbsd
# Node 24 et npm 12 (exigés par OpenFrontIO)
curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 24
npm install -g npm@12
```

## 3. Dépôt et dépendances

```bash
git clone https://github.com/FabLrc/openfront-ai.git
cd openfront-ai
./scripts/setup.sh
# PyTorch avec CUDA (remplace la version CPU installée par requirements.txt)
agent/.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch
agent/.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Le dernier test doit afficher `True NVIDIA GeForce RTX 3080`.

## 4. Récupérer les modèles

`runs/` n'est pas versionné. Le modèle imité (`runs/bc1`) et le run PPO
(`runs/ppo1`, arrêté à la mise à jour 91 et ~186 000 décisions, avec ses
checkpoints, ses métriques et ses évaluations) sont publiés dans la release
GitHub `ppo1-reprise`. Pour les télécharger et les décompresser au bon endroit :

```bash
scripts/fetch-checkpoints.sh
```

Pour une autre sauvegarde, faite sur le Mac avec
`tar czf sauvegarde.tgz runs/bc1/ckpt/best.pt runs/ppo1`, la copier vers le PC
puis `tar xzf /mnt/c/Users/<moi>/Downloads/sauvegarde.tgz` depuis le dossier
du dépôt.

Ne pas entraîner la même série sur les deux machines à la fois. Sinon, partir
d'une nouvelle série sur le PC (par exemple `ppo-pc`) : elle démarre depuis
`runs/bc1/ckpt/best.pt`.

## 5. Lancer

```bash
scripts/train-ppo.sh ppo1 8 30     # reprend ppo1 : 8 h, segments de 30 min, évaluation après chaque segment
```

Au premier segment sur CUDA, le message « Optimiseur réinitialisé » est
normal : sur le Mac, deux convolutions étaient gelées, et sur le PC tout le
réseau est entraîné. Les poids et la progression (mise à jour, curriculum)
sont repris, seul l'état de l'optimiseur repart de zéro.

Le nombre de processus de jeu vaut par défaut le nombre de cœurs physiques
(par exemple 8 sur un i7 à 8 cœurs). Le choisir avec
`agent/.venv/bin/python agent/train_ppo.py --series ppo1 --procs 6`. Le jeu
tourne sur le CPU et le réseau sur le GPU : au-delà d'un processus par cœur
physique, on ne gagne plus rien.

Tests, pour vérifier l'installation :

```bash
(cd env && npm test) && agent/.venv/bin/python -m pytest agent/tests -q
```

## 6. Tableau de bord depuis Windows

WSL2 relaie `localhost` vers Windows : ouvrir dans le navigateur Windows
<http://localhost:8080/> (tableau de bord), <http://localhost:8080/live> et
<http://localhost:8080/games>. Pendant un entraînement, le premier processus
de jeu sert ces pages si le port 8080 est libre. Sinon, lancer
`cd env && npm start` dans un autre terminal Ubuntu.

## 7. Empêcher la veille

`caffeinate` n'existe que sur macOS. Sous Windows, dans un PowerShell, pendant
l'entraînement :

```powershell
powercfg /change standby-timeout-ac 0     # jamais de veille sur secteur
powercfg /change hibernate-timeout-ac 0
```

Remettre ensuite une valeur, par exemple `powercfg /change standby-timeout-ac 30`.
Les réglages d'alimentation de Windows font la même chose : « Mise en veille :
jamais ». WSL2 s'arrête aussi quand plus aucun terminal Ubuntu n'est ouvert :
laisser la fenêtre ouverte, ou lancer l'entraînement dans `tmux`.
