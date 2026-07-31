#!/usr/bin/env bash
# Install + enable SAR services on all nodes (idempotent). Assumes deploy.sh has
# already synced the repo + venvs. Kills any stray nohup instances first so the
# supervised copy is the only one running.
#   scripts/install_services.sh {a|b|c|all}
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PW='asdfg54321'  # Node C sudo (Pis use passwordless sudo)

install_a() {
  echo "== Node A: orchestrator + tripwire =="
  ssh robot-vision@robot-vision.local "sudo mkdir -p /var/log/sar && sudo chown robot-vision /var/log/sar"
  scp -q "$REPO"/scripts/systemd/sar-orchestrator.service \
         "$REPO"/scripts/systemd/sar-tripwire.service \
         robot-vision@robot-vision.local:/tmp/
  # NOTE: no `pkill -f sar_node_a` here — that pattern matches this very ssh
  # shell's own argv and kills the session. systemd takes over the processes; any
  # stray nohup copies are cleared by the daemon-reload + restart (or a reboot).
  ssh robot-vision@robot-vision.local '
    for p in $(pgrep -f "python -m sar_node_a" | grep -vw $PPID); do kill "$p" 2>/dev/null || true; done
    sudo mv /tmp/sar-orchestrator.service /tmp/sar-tripwire.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now sar-orchestrator sar-tripwire
    sleep 4
    systemctl is-active sar-orchestrator sar-tripwire'
}

install_b() {
  echo "== Node B: genaid =="
  ssh robot-genai@robot-genai.local "sudo mkdir -p /var/log/sar && sudo chown robot-genai /var/log/sar"
  scp -q "$REPO"/scripts/systemd/sar-genaid.service robot-genai@robot-genai.local:/tmp/
  ssh robot-genai@robot-genai.local '
    for p in $(pgrep -f "python -m sar_node_b" | grep -vw $PPID); do kill "$p" 2>/dev/null || true; done
    sudo mv /tmp/sar-genaid.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now sar-genaid
    sleep 6
    systemctl is-active sar-genaid'
}

install_c() {
  echo "== Node C: cortexd LaunchAgent =="
  scp -q "$REPO"/scripts/launchd/com.sar.cortexd.plist \
         tokenator@10.1.215.33:/Users/tokenator/Library/LaunchAgents/
  ssh tokenator@10.1.215.33 '
    pkill -f "sar_node_c.cortexd" 2>/dev/null || true
    UID_N=$(id -u)
    launchctl bootout gui/$UID_N/com.sar.cortexd 2>/dev/null || true
    launchctl bootstrap gui/$UID_N /Users/tokenator/Library/LaunchAgents/com.sar.cortexd.plist
    launchctl enable gui/$UID_N/com.sar.cortexd
    sleep 12
    launchctl print gui/$UID_N/com.sar.cortexd | grep -E "state =" || echo "not loaded"'
}

case "${1:-all}" in
  a) install_a ;; b) install_b ;; c) install_c ;;
  all) install_a; install_b; install_c ;;
  *) echo "usage: $0 {a|b|c|all}"; exit 1 ;;
esac
echo "== done =="
