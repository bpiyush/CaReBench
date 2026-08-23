#!/usr/bin/env bash
# Remote-side backend launcher for the video-search demo.
# Finds free GPUs (up to MAX_GPUS), starts Gradio, writes a ready file.
#
# Usage (on remote):
#   ./remote_start_backend.sh --port 7860
#
# Env overrides:
#   DEMO_ROOT, DEMO_PYTHON, MAX_GPUS, MEM_FREE_MIB

set -euo pipefail

PORT=7860
MAX_GPUS="${MAX_GPUS:-4}"
# Treat a GPU as free if used memory is below this (MiB). Prefer least-used.
MEM_FREE_MIB="${MEM_FREE_MIB:-1500}"
DEMO_ROOT="${DEMO_ROOT:-/users/piyush/projects/CaReBench/tara-presentation}"
# Gradio + TARA need carebench; Qwen model still runs via qwen worker subprocess.
DEMO_PYTHON="${DEMO_PYTHON:-/users/piyush/miniconda3/envs/carebench/bin/python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --max-gpus) MAX_GPUS="$2"; shift 2 ;;
    --mem-free-mib) MEM_FREE_MIB="$2"; shift 2 ;;
    --demo-root) DEMO_ROOT="$2"; shift 2 ;;
    --python) DEMO_PYTHON="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOG_DIR="${DEMO_ROOT}/logs"
mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/backend-${PORT}.pid"
READY_FILE="${LOG_DIR}/backend-${PORT}.ready"
LOG_FILE="${LOG_DIR}/backend-${PORT}.log"
GPU_FILE="${LOG_DIR}/backend-${PORT}.gpus"

find_free_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  # Sort by memory.used ascending; pick up to MAX_GPUS under the free threshold.
  local selected
  selected="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' -v thr="$MEM_FREE_MIB" -v maxg="$MAX_GPUS" '
          {
            gsub(/ /,"",$1); gsub(/ /,"",$2);
            if ($2+0 < thr) print $2+0, $1
          }
        ' \
      | sort -n \
      | awk -v maxg="$MAX_GPUS" '{ ids[++n]=$2 } END {
          if (n==0) exit 0
          lim = (n < maxg) ? n : maxg
          for (i=1; i<=lim; i++) {
            printf "%s%s", ids[i], (i<lim ? "," : "")
          }
          print ""
        }'
  )"
  echo "$selected"
}

# Stop previous instance on this port if we own the pid file
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[remote] Stopping previous backend pid=${OLD_PID}"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE" "$READY_FILE"
fi

# Also free anything already bound to the port
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
elif command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -tiTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "${PIDS}" ]]; then
    kill ${PIDS} 2>/dev/null || true
    sleep 1
  fi
fi

GPUS="$(find_free_gpus)"
if [[ -z "$GPUS" ]]; then
  echo "[remote] ERROR: no free GPUs found (need memory.used < ${MEM_FREE_MIB} MiB)." >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2 || true
  exit 1
fi
echo "$GPUS" > "$GPU_FILE"
echo "[remote] Using GPUs: ${GPUS} (device_map=auto)"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="${DEMO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
# Mark readiness via a tiny wrapper: Gradio prints "Running on local URL"
rm -f "$READY_FILE"

cd "$DEMO_ROOT"
nohup "$DEMO_PYTHON" -u server.py --host 127.0.0.1 --port "$PORT" \
  >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "[remote] Started pid=$(cat "$PID_FILE") log=${LOG_FILE}"

# Wait until the FastAPI app serves HTTP 200
for i in $(seq 1 180); do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[remote] ERROR: backend exited early. Tail of log:" >&2
    tail -n 60 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if command -v curl >/dev/null 2>&1; then
    code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/" || true)"
    if [[ "$code" == "200" || "$code" == "302" ]]; then
      date -u +%Y-%m-%dT%H:%M:%SZ > "$READY_FILE"
      echo "[remote] Backend ready on port ${PORT} (http ${code})"
      exit 0
    fi
  fi
  sleep 2
done

echo "[remote] ERROR: timed out waiting for backend. Tail of log:" >&2
tail -n 60 "$LOG_FILE" >&2 || true
exit 1
