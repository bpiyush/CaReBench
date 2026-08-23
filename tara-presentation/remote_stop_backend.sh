#!/usr/bin/env bash
# Stop the remote demo backend for a given port (run on remote).
set -euo pipefail

PORT="${1:-7860}"
DEMO_ROOT="${DEMO_ROOT:-/users/piyush/projects/CaReBench/tara-presentation}"
PID_FILE="${DEMO_ROOT}/logs/backend-${PORT}.pid"
READY_FILE="${DEMO_ROOT}/logs/backend-${PORT}.ready"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    echo "[remote] Killing backend pid=${PID}"
    kill "$PID" 2>/dev/null || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi
rm -f "$READY_FILE"

if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
echo "[remote] Stopped backend on port ${PORT}"
