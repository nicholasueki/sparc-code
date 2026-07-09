#!/usr/bin/env bash
# Install + enable Lucas services on all nodes (idempotent). Assumes deploy.sh has
# already synced the repo + venvs. Kills any stray nohup instances first so the
# supervised copy is the only one running.
#   scripts/install_services.sh {a|b|c|all}
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PW='asdfg54321'  # Node C sudo (Pis use passwordless sudo)

install_a() {
  echo "== Node A: orchestrator + tripwire =="
  ssh robot-vision@robot-vision.local "sudo mkdir -p /var/log/lucas && sudo chown robot-vision /var/log/lucas"
  scp -q "$REPO"/scripts/systemd/lucas-orchestrator.service \
         "$REPO"/scripts/systemd/lucas-tripwire.service \
         robot-vision@robot-vision.local:/tmp/
  # NOTE: no `pkill -f lucas_node_a` here — that pattern matches this very ssh
  # shell's own argv and kills the session. systemd takes over the processes; any
  # stray nohup copies are cleared by the daemon-reload + restart (or a reboot).
  ssh robot-vision@robot-vision.local '
    for p in $(pgrep -f "python -m lucas_node_a" | grep -vw $PPID); do kill "$p" 2>/dev/null || true; done
    sudo mv /tmp/lucas-orchestrator.service /tmp/lucas-tripwire.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now lucas-orchestrator lucas-tripwire
    sleep 4
    systemctl is-active lucas-orchestrator lucas-tripwire'
}

install_b() {
  echo "== Node B: genaid =="
  ssh robot-genai@robot-genai.local "sudo mkdir -p /var/log/lucas && sudo chown robot-genai /var/log/lucas"
  scp -q "$REPO"/scripts/systemd/lucas-genaid.service robot-genai@robot-genai.local:/tmp/
  ssh robot-genai@robot-genai.local '
    for p in $(pgrep -f "python -m lucas_node_b" | grep -vw $PPID); do kill "$p" 2>/dev/null || true; done
    sudo mv /tmp/lucas-genaid.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now lucas-genaid
    sleep 6
    systemctl is-active lucas-genaid'
}

install_c() {
  echo "== Node C: cortexd LaunchAgent =="
  scp -q "$REPO"/scripts/launchd/com.lucas.cortexd.plist \
         tokenator@10.1.215.33:/Users/tokenator/Library/LaunchAgents/
  ssh tokenator@10.1.215.33 '
    pkill -f "lucas_node_c.cortexd" 2>/dev/null || true
    UID_N=$(id -u)
    launchctl bootout gui/$UID_N/com.lucas.cortexd 2>/dev/null || true
    launchctl bootstrap gui/$UID_N /Users/tokenator/Library/LaunchAgents/com.lucas.cortexd.plist
    launchctl enable gui/$UID_N/com.lucas.cortexd
    sleep 12
    launchctl print gui/$UID_N/com.lucas.cortexd | grep -E "state =" || echo "not loaded"'
}

case "${1:-all}" in
  a) install_a ;; b) install_b ;; c) install_c ;;
  all) install_a; install_b; install_c ;;
  *) echo "usage: $0 {a|b|c|all}"; exit 1 ;;
esac
echo "== done =="
