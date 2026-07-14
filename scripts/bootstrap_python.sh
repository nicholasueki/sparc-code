#!/usr/bin/env bash
# Install one Lucas dependency group from its complete hash-locked graph.
# Usage: PYTHON=/path/to/python scripts/bootstrap_python.sh {dev|node-a|node-b|node-c}
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-}"

fail() {
  echo "bootstrap error: $*" >&2
  exit 1
}

require_python() {
  expected="$1"
  actual=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  [ "$actual" = "$expected" ] || fail "$TARGET requires Python $expected (found $actual at $PYTHON)"
}

require_darwin() {
  [ "$(uname -s)" = "Darwin" ] || fail "$TARGET requires macOS"
}

case "$TARGET" in
  dev)
    PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
    LOCK="$REPO/locks/dev-macos-py312.txt"
    [ -x "$PYTHON" ] || fail "Python interpreter is not executable: $PYTHON"
    require_darwin
    require_python 3.12
    ;;
  node-a)
    PYTHON="${PYTHON:-$HOME/lucas_venv/bin/python}"
    LOCK="$REPO/locks/node-a-debian13-arm64-py313.txt"
    [ -x "$PYTHON" ] || fail "Python interpreter is not executable: $PYTHON"
    require_python 3.13
    ;;
  node-b)
    PYTHON="${PYTHON:-$HOME/lucas_venv/bin/python}"
    LOCK="$REPO/locks/node-b-debian13-arm64-py313.txt"
    [ -x "$PYTHON" ] || fail "Python interpreter is not executable: $PYTHON"
    require_python 3.13
    ;;
  node-c)
    PYTHON="${PYTHON:-$HOME/mlx312/bin/python}"
    LOCK="$REPO/locks/node-c-macos14-arm64-py312.txt"
    require_darwin
    [ "$(uname -m)" = "arm64" ] || fail "node-c requires Apple-silicon arm64"
    command -v sw_vers >/dev/null 2>&1 || fail "cannot determine the macOS version (sw_vers missing)"
    macos_version=$(sw_vers -productVersion)
    macos_major=${macos_version%%.*}
    case "$macos_major" in
      ''|*[!0-9]*) fail "cannot parse macOS version: $macos_version" ;;
    esac
    [ "$macos_major" -ge 14 ] || fail "node-c requires macOS 14 or newer (found $macos_version)"
    [ -x "$PYTHON" ] || fail "Python interpreter is not executable: $PYTHON"
    require_python 3.12
    ;;
  *)
    fail "usage: $0 {dev|node-a|node-b|node-c}"
    ;;
esac

"$PYTHON" -m pip install --disable-pip-version-check --no-deps \
  --require-hashes -r "$REPO/locks/bootstrap.txt"
"$PYTHON" -m pip install --disable-pip-version-check --only-binary=:all: \
  --require-hashes -r "$LOCK"
"$PYTHON" -m pip install --disable-pip-version-check --no-deps \
  --no-build-isolation "$REPO"
"$PYTHON" -m pip check

echo "$TARGET dependency lock installed with $PYTHON"
