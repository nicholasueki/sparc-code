#!/usr/bin/env bash
# Deploy sparc to node(s): scripts/deploy.sh {a|b|c|all}
# Rsyncs the repo, creates the node venv (system-site-packages so Hailo/Picamera2
# system libs stay importable), and installs a committed dependency group through
# the reviewed constraints file. Portable to bash 3.2.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

node_params() {
  case "$1" in
    a) HOST="robot-vision@robot-vision.local"; DEST="/home/robot-vision/sparc"
       EXTRA="node-a" ;;
    b) HOST="robot-genai@robot-genai.local"; DEST="/home/robot-genai/sparc"
       EXTRA="node-b" ;;
    c) HOST="tokenator@10.1.215.33"; DEST="/Users/tokenator/sparc"
       EXTRA="node-c" ;;
    *) echo "unknown node $1"; exit 1 ;;
  esac
}

deploy_node() {
  node_params "$1"
  echo "== deploying node_$1 -> $HOST:$DEST"
  rsync -az --delete \
    --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude '.pytest_cache' --exclude '*.egg-info' --exclude 'models/' \
    --exclude '*.hef' --exclude '*.rpk' --exclude '*.gguf' --exclude '*.safetensors' \
    "$REPO/" "$HOST:$DEST/"
  if [ "$1" = "c" ]; then
    ssh "$HOST" "cd '$DEST' && ~/.local/bin/uv pip install --python ~/mlx312/bin/python -q -c 'constraints-${EXTRA}.txt' '.[${EXTRA}]' && echo deps-ok"
  else
    ssh "$HOST" "python3 -m venv --system-site-packages ~/sparc_venv 2>/dev/null || true; \
                 cd '$DEST' && ~/sparc_venv/bin/pip install -q -c 'constraints-${EXTRA}.txt' '.[${EXTRA}]' && echo deps-ok"
  fi
  echo "== node_$1 deployed"
}

case "${1:-all}" in
  a|b|c) deploy_node "$1" ;;
  all) deploy_node a; deploy_node b; deploy_node c ;;
  *) echo "usage: $0 {a|b|c|all}"; exit 1 ;;
esac
