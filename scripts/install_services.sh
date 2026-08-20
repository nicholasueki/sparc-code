#!/usr/bin/env bash
# Install, enable, and restart every active SPARC daemon on selected nodes.
# scripts/deploy.sh must have already synchronized code and Python environments.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NODE_A="${SPARC_NODE_A:-robot-vision@robot-vision.local}"
NODE_B="${SPARC_NODE_B:-robot-genai@robot-genai.local}"
NODE_C="${SPARC_NODE_C:-tokenator@10.1.215.33}"

install_a() {
  echo "== Node A: orchestrator + tripwire + enrich =="
  ssh "$NODE_A" 'sudo install -d -o robot-vision -g robot-vision /var/log/sparc'
  scp -q \
    "$REPO/scripts/systemd/sparc-orchestrator.service" \
    "$REPO/scripts/systemd/sparc-tripwire.service" \
    "$REPO/scripts/systemd/sparc-enrich.service" \
    "$NODE_A:/tmp/"
  ssh "$NODE_A" 'set -eu
    sudo install -m 0644 /tmp/sparc-orchestrator.service /etc/systemd/system/sparc-orchestrator.service
    sudo install -m 0644 /tmp/sparc-tripwire.service /etc/systemd/system/sparc-tripwire.service
    sudo install -m 0644 /tmp/sparc-enrich.service /etc/systemd/system/sparc-enrich.service
    rm -f /tmp/sparc-orchestrator.service /tmp/sparc-tripwire.service /tmp/sparc-enrich.service
    sudo systemctl daemon-reload
    sudo systemctl enable sparc-orchestrator.service sparc-tripwire.service sparc-enrich.service
    sudo systemctl restart sparc-orchestrator.service sparc-tripwire.service sparc-enrich.service
    systemctl is-enabled --quiet sparc-orchestrator.service sparc-tripwire.service sparc-enrich.service
    systemctl is-active --quiet sparc-orchestrator.service sparc-tripwire.service sparc-enrich.service'
}

install_b() {
  echo "== Node B: genaid =="
  ssh "$NODE_B" 'sudo install -d -o robot-genai -g robot-genai /var/log/sparc'
  scp -q "$REPO/scripts/systemd/sparc-genaid.service" "$NODE_B:/tmp/"
  ssh "$NODE_B" 'set -eu
    sudo install -m 0644 /tmp/sparc-genaid.service /etc/systemd/system/sparc-genaid.service
    rm -f /tmp/sparc-genaid.service
    sudo systemctl daemon-reload
    sudo systemctl enable sparc-genaid.service
    sudo systemctl restart sparc-genaid.service
    systemctl is-enabled --quiet sparc-genaid.service
    systemctl is-active --quiet sparc-genaid.service'
}

install_c() {
  echo "== Node C: cortexd + earsd =="
  ssh "$NODE_C" 'mkdir -p /Users/tokenator/Library/LaunchAgents'
  scp -q \
    "$REPO/scripts/launchd/com.sparc.cortexd.plist" \
    "$REPO/scripts/launchd/com.sparc.earsd.plist" \
    "$NODE_C:/Users/tokenator/Library/LaunchAgents/"
  ssh "$NODE_C" 'set -eu
    domain="gui/$(id -u)"
    for label in com.sparc.cortexd com.sparc.earsd; do
      plist="/Users/tokenator/Library/LaunchAgents/$label.plist"
      if launchctl print "$domain/$label" >/dev/null 2>&1; then
        launchctl bootout "$domain/$label"
      fi
      launchctl bootstrap "$domain" "$plist"
      launchctl enable "$domain/$label"
      launchctl kickstart -k "$domain/$label"
      launchctl print "$domain/$label" | grep -q "state = running"
    done'
}

case "${1:-}" in
  a) install_a ;;
  b) install_b ;;
  c) install_c ;;
  all) install_a; install_b; install_c ;;
  *) echo "usage: $0 {a|b|c|all}" >&2; exit 2 ;;
esac

echo "== service installation complete =="
