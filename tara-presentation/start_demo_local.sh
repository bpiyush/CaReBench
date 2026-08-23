#!/usr/bin/env bash
# =============================================================================
# start_demo_local.sh
#
# Run this on your LAPTOP / local machine. It will:
#   1. Open one multiplexed SSH session to the remote (survives jump hosts)
#   2. Start the Gradio video-search backend on free GPUs
#   3. Port-forward over the same session
#   4. Print the local URL
#
# Usage:
#   ./start_demo_local.sh --remote_server_name dev1 --port 7860
#   ./start_demo_local.sh -r dev1 -p 7860
#   ./start_demo_local.sh -r dev1 -p 7860 --stop
#
# Recommended ~/.ssh/config on your laptop (dev1 is not reachable directly):
#
#   Host athena
#     HostName athena.robots.ox.ac.uk
#     User piyush
#
#   Host dev1
#     HostName 10.190.30.11          # or whatever internal name you use
#     User piyush
#     ProxyJump athena
#     ServerAliveInterval 30
#     ServerAliveCountMax 3
#
# If unsure of HostName, on athena run:  getent hosts dev1
# =============================================================================

set -euo pipefail

REMOTE=""
PORT=7860
STOP=0
MAX_GPUS=4
MEM_FREE_MIB=""

# ---------------------------------------------------------------------------
# Predefined remote profiles
# ---------------------------------------------------------------------------
# NOTE: Gradio+TARA use carebench; Qwen3VL still uses the qwen worker env.
# ---------------------------------------------------------------------------
remote_type() {
  case "$1" in
    dev1) echo "direct" ;;
    athena|civo) echo "slurm" ;;
    *) echo "unknown" ;;
  esac
}

remote_demo_root() {
  case "$1" in
    dev1|athena|civo)
      echo "/users/piyush/projects/CaReBench/tara-presentation"
      ;;
    *) echo "" ;;
  esac
}

remote_python() {
  case "$1" in
    dev1|athena|civo)
      echo "/users/piyush/miniconda3/envs/carebench/bin/python"
      ;;
    *) echo "" ;;
  esac
}

usage() {
  cat <<EOF
Usage: $(basename "$0") --remote_server_name NAME --port PORT [--stop] [--max-gpus N]

  --remote_server_name, -r   SSH host alias (e.g. dev1)
  --port, -p                 Gradio port on remote AND local (default: 7860)
  --max-gpus                 Max free GPUs to claim on direct machines (default: 4)
  --mem-free-mib             GPU used-memory below this (MiB) counts as free (default: 1500)
  --stop                     Tear down remote backend + local SSH session
  -h, --help                 Show this help

Supported remotes today:
  dev1   direct GPUs (find free, up to --max-gpus, device_map=auto)

TODO (Slurm login nodes — not wired yet): athena / civo
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote_server_name|-r) REMOTE="$2"; shift 2 ;;
    --port|-p) PORT="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --mem-free-mib) MEM_FREE_MIB="$2"; shift 2 ;;
    --stop) STOP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

MEM_FREE_MIB="${MEM_FREE_MIB:-1500}"
if [[ -z "$REMOTE" ]]; then
  echo "ERROR: --remote_server_name is required" >&2
  usage
  exit 1
fi

TYPE="$(remote_type "$REMOTE")"
DEMO_ROOT="$(remote_demo_root "$REMOTE")"
DEMO_PYTHON="$(remote_python "$REMOTE")"
STATE_DIR="${HOME}/.cache/tara-presentation-demo"
mkdir -p "$STATE_DIR"
CTRL_DIR="${STATE_DIR}/ssh"
mkdir -p "$CTRL_DIR"
CTRL_SOCK="${CTRL_DIR}/${REMOTE}-${PORT}.sock"
MASTER_PID_FILE="${STATE_DIR}/master-${REMOTE}-${PORT}.pid"

ssh_base() {
  # Reuse one TCP path through any ProxyJump; avoids "Connection closed" on 2nd hop.
  ssh \
    -o ControlMaster=auto \
    -o "ControlPath=${CTRL_SOCK}" \
    -o ControlPersist=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "$@"
}

ssh_remote() {
  ssh_base -o BatchMode=yes "$REMOTE" "$@"
}

master_running() {
  ssh_base -O check "$REMOTE" >/dev/null 2>&1
}

start_master() {
  if master_running; then
    echo "[local] Reusing existing SSH master to ${REMOTE}"
    return 0
  fi
  rm -f "$CTRL_SOCK"
  echo "[local] Opening SSH master to '${REMOTE}' (keeps jump-host path warm) ..."
  # -M -N -f : master, no remote command, background after auth
  # Port forward is attached here so we only need one connection.
  if ! ssh_base -f -N -M \
      -L "${PORT}:localhost:${PORT}" \
      -o BatchMode=yes \
      -o ConnectTimeout=30 \
      "$REMOTE"; then
    cat >&2 <<EOF
ERROR: cannot open SSH master to '${REMOTE}'.

Common fix — put this in ~/.ssh/config on your laptop:

  Host athena
    HostName athena.robots.ox.ac.uk
    User piyush

  Host dev1
    HostName 10.190.30.11
    User piyush
    ProxyJump athena

Then test:  ssh ${REMOTE} hostname
EOF
    return 1
  fi
  # Best-effort pid record
  if command -v pgrep >/dev/null 2>&1; then
    TPID="$(pgrep -n -f "ssh .*ControlPath=${CTRL_SOCK}" || true)"
    [[ -n "${TPID}" ]] && echo "$TPID" > "$MASTER_PID_FILE"
  fi
}

stop_master() {
  if master_running; then
    echo "[local] Closing SSH master to ${REMOTE}"
    ssh_base -O exit "$REMOTE" >/dev/null 2>&1 || true
  fi
  rm -f "$CTRL_SOCK" "$MASTER_PID_FILE"
}

if [[ "$STOP" -eq 1 ]]; then
  echo "[local] Stopping demo on ${REMOTE}:${PORT}"
  if master_running || start_master; then
    ssh_remote "bash '${DEMO_ROOT}/remote_stop_backend.sh' '${PORT}'" || true
  else
    echo "[local] (could not reach remote to stop backend; closing local tunnel only)"
  fi
  stop_master
  echo "[local] Done."
  exit 0
fi

if [[ "$TYPE" == "unknown" || -z "$DEMO_ROOT" ]]; then
  echo "ERROR: unknown remote '${REMOTE}'. Add it to the profile maps in this script." >&2
  exit 1
fi

if [[ "$TYPE" == "slurm" ]]; then
  cat >&2 <<EOF
ERROR: Slurm remotes (${REMOTE}) are not implemented yet.

TODO:
  1. sbatch a 12h GPU job that runs remote_start_backend.sh on the node
  2. Poll until Gradio is ready
  3. Tunnel: local -> login -> compute_node:${PORT}

For now:  $(basename "$0") --remote_server_name dev1 --port ${PORT}
EOF
  exit 1
fi

start_master

echo "[local] Checking remote host ..."
REMOTE_HOST="$(ssh_remote "hostname" | tr -d '\r')"
echo "[local] Remote hostname: ${REMOTE_HOST}"

echo "[local] Starting backend on ${REMOTE} (port=${PORT}, max_gpus=${MAX_GPUS}) ..."
if ! ssh_remote \
  "bash '${DEMO_ROOT}/remote_start_backend.sh' --port '${PORT}' --max-gpus '${MAX_GPUS}' --mem-free-mib '${MEM_FREE_MIB}' --python '${DEMO_PYTHON}' --demo-root '${DEMO_ROOT}'"; then
  echo "[local] ERROR: remote backend failed to start. Dumping remote log tail:" >&2
  ssh_remote "tail -n 50 '${DEMO_ROOT}/logs/backend-${PORT}.log' 2>/dev/null || true" >&2 || true
  exit 1
fi

# Probe via tunnel
if command -v curl >/dev/null 2>&1; then
  for _ in 1 2 3 4 5 6 7 8; do
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/" || true)"
    if [[ "$code" == "200" || "$code" == "302" ]]; then
      break
    fi
    sleep 1
  done
fi

GPUS_INFO="$(ssh_remote "cat '${DEMO_ROOT}/logs/backend-${PORT}.gpus' 2>/dev/null || echo unknown" | tr -d '\r')"

cat <<EOF

============================================================
  Demo is ready.

  Open:  http://localhost:${PORT}

  Remote : ${REMOTE} (${REMOTE_HOST}, type=${TYPE})
  GPUs   : ${GPUS_INFO}
  Stop   : ~/bin/$(basename "$0") --remote_server_name ${REMOTE} --port ${PORT} --stop
============================================================
EOF
