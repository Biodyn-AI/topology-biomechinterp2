#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime"
PID_FILE="$RUNTIME_DIR/autoloop.pid"
STOP_FILE="$RUNTIME_DIR/STOP"
PY_SCRIPT="$ROOT_DIR/loop/run_codex_topology_autoloop.py"

mkdir -p "$RUNTIME_DIR"
touch "$STOP_FILE"

find_loop_pid() {
  pgrep -f "$PY_SCRIPT" 2>/dev/null | tail -n 1 || true
}

PID=""
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" || true)"
fi

if [[ -z "$PID" ]] || ! kill -0 "$PID" 2>/dev/null; then
  PID="$(find_loop_pid)"
fi

if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  kill "${PID}" || true
  sleep 1
  if kill -0 "${PID}" 2>/dev/null; then
    kill -9 "${PID}" || true
  fi
  echo "Stopped autoloop process ${PID}"
  rm -f "$PID_FILE"
else
  echo "No running autoloop process found. STOP flag created."
fi
