#!/usr/bin/env bash
# Deploy lucas to node(s): scripts/deploy.sh {a|b|c|all}
# Rsyncs the repo, creates the node venv (system-site-packages so hailo/picamera2
# system libs stay importable), installs per-node deps.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"

declare -A HOST=( [a]="robot-vision@robot-vision.local"
                  [b]="robot-genai@robot-genai.local"
                  [c]="tokenator@10.1.215.33" )
declare -A DEST=( [a]="/home/robot-vision/lucas"
                  [b]="/home/robot-genai/lucas"
                  [c]="/Users/tokenator/lucas" )
declare -A REQS=( [a]="pydantic paho-mqtt pyyaml httpx numpy fastapi uvicorn"
                  [b]="pydantic paho-mqtt pyyaml httpx fastapi uvicorn"
                  [c]="pydantic paho-mqtt pyyaml httpx fastapi uvicorn fastembed sqlite-vec" )

deploy_node() {
  local n="$1" host="${HOST[$1]}" dest="${DEST[$1]}"
  echo "== deploying node_$n -> $host:$dest"
  rsync -az --delete \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO/" "$host:$dest/"
  if [[ "$n" == "c" ]]; then
    # Node C: install into the existing uv-managed mlx312 venv (has mlx-vlm)
    ssh "$host" "~/.local/bin/uv pip install --python ~/mlx312/bin/python -q ${REQS[$n]} && echo deps-ok"
  else
    ssh "$host" "python3 -m venv --system-site-packages ~/lucas_venv 2>/dev/null || true; \
                 ~/lucas_venv/bin/pip install -q ${REQS[$n]} && echo deps-ok"
  fi
  echo "== node_$n deployed"
}

case "${1:-all}" in
  a|b|c) deploy_node "$1" ;;
  all) deploy_node a; deploy_node b; deploy_node c ;;
  *) echo "usage: $0 {a|b|c|all}"; exit 1 ;;
esac
