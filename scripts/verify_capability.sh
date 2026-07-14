#!/usr/bin/env bash
# Verify functional readiness: scripts/verify_capability.sh {v0.3|v0.4|active}
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SPARC_PYTHON:-$REPO/.venv/bin/python}"

case "${1:-}" in
  v0.3|v0.4|active) ;;
  *) echo "usage: $0 {v0.3|v0.4|active}" >&2; exit 2 ;;
esac
[ "$#" -eq 1 ] || { echo "usage: $0 {v0.3|v0.4|active}" >&2; exit 2; }
[ -x "$PYTHON" ] || {
  echo "verifier error: Python not found at $PYTHON; bootstrap the dev environment first" >&2
  exit 1
}

cd "$REPO"
export SPARC_REPO="$REPO"
export PYTHONPATH="$REPO/packages/common${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m sparc_common.readiness "$1"
