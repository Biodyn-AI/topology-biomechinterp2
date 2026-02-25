#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime"
PID_FILE="$RUNTIME_DIR/autoloop.pid"
STOP_FILE="$RUNTIME_DIR/STOP"
NOHUP_LOG="$RUNTIME_DIR/autoloop.nohup.log"
PY_SCRIPT="$ROOT_DIR/loop/run_codex_topology_autoloop.py"
CONFIG_FILE="$ROOT_DIR/loop/config.json"

is_pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

mkdir -p "$RUNTIME_DIR"
rm -f "$STOP_FILE"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" || true)"
  if is_pid_running "${OLD_PID}"; then
    echo "Autoloop already running with PID ${OLD_PID}"
    exit 1
  fi
  rm -f "$PID_FILE"
fi

# Use a double-fork launcher to survive parent-shell cleanup in detached mode.
python3 - "$PID_FILE" "$NOHUP_LOG" "$PY_SCRIPT" "$CONFIG_FILE" "$ROOT_DIR" "$@" <<'PY'
import os
import sys

pid_file = sys.argv[1]
log_file = sys.argv[2]
py_script = sys.argv[3]
config_file = sys.argv[4]
root_dir = sys.argv[5]
extra_args = sys.argv[6:]
cmd = ["python3", py_script, "--config", config_file, *extra_args]

first = os.fork()
if first > 0:
    sys.exit(0)

os.setsid()
second = os.fork()
if second > 0:
    sys.exit(0)

os.chdir(root_dir)
null_fd = os.open("/dev/null", os.O_RDONLY)
log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(null_fd, 0)
os.dup2(log_fd, 1)
os.dup2(log_fd, 2)
for fd in (null_fd, log_fd):
    if fd > 2:
        os.close(fd)

with open(pid_file, "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))

os.execvp(cmd[0], cmd)
PY

for _ in {1..50}; do
  if [[ -s "$PID_FILE" ]]; then
    break
  fi
  sleep 0.1
done

if [[ ! -s "$PID_FILE" ]]; then
  echo "Failed to start autoloop: PID file was not created."
  exit 1
fi

NEW_PID="$(cat "$PID_FILE" || true)"
if ! is_pid_running "${NEW_PID}"; then
  echo "Autoloop failed to stay running (PID ${NEW_PID})."
  exit 1
fi

echo "Started Codex autoloop (PID ${NEW_PID})"
echo "Log: $NOHUP_LOG"
