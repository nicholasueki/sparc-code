#!/usr/bin/env bash
# Deploy sar to node(s): scripts/deploy.sh {a|b|c|all}
# Rsyncs the repo, creates the node venv (system-site-packages so hailo/picamera2
# system libs stay importable), installs per-node deps. Portable to bash 3.2.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

node_params() {
  case "$1" in
    a) HOST="robot-vision@robot-vision.local"; DEST="/home/robot-vision/sar"
       REQS="pydantic paho-mqtt pyyaml httpx numpy fastapi uvicorn" ;;
    b) HOST="robot-genai@robot-genai.local"; DEST="/home/robot-genai/sar"
       REQS="pydantic paho-mqtt pyyaml httpx fastapi uvicorn" ;;
    c) HOST="tokenator@10.1.215.33"; DEST="/Users/tokenator/sar"
       REQS="pydantic paho-mqtt pyyaml httpx fastapi uvicorn fastembed sqlite-vec" ;;
    *) echo "unknown node $1"; exit 1 ;;
  esac
}

deploy_node() {
  node_params "$1"
  echo "== deploying node_$1 -> $HOST:$DEST"
  rsync -az --delete \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
    "$REPO/" "$HOST:$DEST/"
  if [ "$1" = "c" ]; then
    ssh "$HOST" "~/.local/bin/uv pip install --python ~/mlx312/bin/python -q $REQS && echo deps-ok"
  else
    ssh "$HOST" "python3 -m venv --system-site-packages ~/sar_venv 2>/dev/null || true; \
                 ~/sar_venv/bin/pip install -q $REQS && echo deps-ok"
  fi
  echo "== node_$1 deployed"
}

case "${1:-all}" in
  a|b|c) deploy_node "$1" ;;
  all) deploy_node a; deploy_node b; deploy_node c ;;
  *) echo "usage: $0 {a|b|c|all}"; exit 1 ;;
esac
