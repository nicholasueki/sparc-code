#!/usr/bin/env bash
# Regenerate every supported-platform lock with the pinned resolver.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

UV_BIN="${UV_BIN:-uv}"
EXPECTED_UV="uv 0.9.28"
EXCLUDE_NEWER="2026-07-14T18:17:36Z"

actual_uv=$($UV_BIN --version)
case "$actual_uv" in
  "$EXPECTED_UV"*) ;;
  *) echo "lock error: expected $EXPECTED_UV, found $actual_uv" >&2; exit 1 ;;
esac

compile_group() {
  extra="$1"
  constraints="$2"
  python_version="$3"
  platform="$4"
  output="$5"
  $UV_BIN pip compile pyproject.toml \
    --extra "$extra" \
    --constraints "$constraints" \
    --python-version "$python_version" \
    --python-platform "$platform" \
    --only-binary :all: \
    --generate-hashes \
    --exclude-newer "$EXCLUDE_NEWER" \
    --no-emit-package sparc-robot \
    --custom-compile-command scripts/lock_dependencies.sh \
    --upgrade \
    --output-file "$output"
}

$UV_BIN pip compile locks/bootstrap.in \
  --universal \
  --python-version 3.12 \
  --only-binary :all: \
  --generate-hashes \
  --exclude-newer "$EXCLUDE_NEWER" \
  --custom-compile-command scripts/lock_dependencies.sh \
  --upgrade \
  --output-file locks/bootstrap.txt

$UV_BIN pip compile locks/lock-tools.in \
  --python-version 3.12 \
  --python-platform aarch64-apple-darwin \
  --only-binary :all: \
  --generate-hashes \
  --exclude-newer "$EXCLUDE_NEWER" \
  --custom-compile-command scripts/lock_dependencies.sh \
  --upgrade \
  --output-file locks/lock-tools.txt

compile_group dev constraints-dev.txt 3.12 aarch64-apple-darwin \
  locks/dev-macos-py312.txt
compile_group node-a constraints-node-a.txt 3.13 aarch64-manylinux_2_39 \
  locks/node-a-debian13-arm64-py313.txt
compile_group node-b constraints-node-b.txt 3.13 aarch64-manylinux_2_39 \
  locks/node-b-debian13-arm64-py313.txt
MACOSX_DEPLOYMENT_TARGET=14.0 compile_group \
  node-c constraints-node-c.txt 3.12 aarch64-apple-darwin \
  locks/node-c-macos14-arm64-py312.txt

# The dev lock supports both macOS architectures only while they resolve to the
# same exact graph. Hash generation includes distributions for both.
dev_arm=$(mktemp "${TMPDIR:-/tmp}/sparc-dev-arm.XXXXXX")
dev_x86=$(mktemp "${TMPDIR:-/tmp}/sparc-dev-x86.XXXXXX")
cleanup() { rm -f "$dev_arm" "$dev_x86"; }
trap cleanup EXIT
for spec in "aarch64-apple-darwin:$dev_arm" "x86_64-apple-darwin:$dev_x86"; do
  platform=${spec%%:*}
  output=${spec#*:}
  $UV_BIN pip compile pyproject.toml \
    --extra dev \
    --constraints constraints-dev.txt \
    --python-version 3.12 \
    --python-platform "$platform" \
    --only-binary :all: \
    --exclude-newer "$EXCLUDE_NEWER" \
    --no-emit-package sparc-robot \
    --no-annotate \
    --no-header \
    --upgrade \
    --output-file "$output"
done
cmp -s "$dev_arm" "$dev_x86" || {
  echo "lock error: macOS arm64/x86_64 dev graphs differ" >&2
  exit 1
}

echo "dependency locks regenerated with $actual_uv (upload cutoff $EXCLUDE_NEWER)"
