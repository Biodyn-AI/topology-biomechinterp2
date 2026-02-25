#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime"
PID_FILE="$RUNTIME_DIR/autoloop.pid"
STATUS_FILE="$RUNTIME_DIR/loop_status.json"
EVENTS_FILE="$RUNTIME_DIR/loop_events.jsonl"
NOHUP_LOG="$RUNTIME_DIR/autoloop.nohup.log"
PY_SCRIPT="$ROOT_DIR/loop/run_codex_topology_autoloop.py"

find_loop_pid() {
  pgrep -f "$PY_SCRIPT" 2>/dev/null | tail -n 1 || true
}

RUNNING_PID=""
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" || true)"
  if [[ -n "${PID}" ]] && kill -0 "${PID}" 2>/dev/null; then
    RUNNING_PID="$PID"
  fi
fi

if [[ -z "$RUNNING_PID" ]]; then
  DISCOVERED_PID="$(find_loop_pid)"
  if [[ -n "$DISCOVERED_PID" ]] && kill -0 "$DISCOVERED_PID" 2>/dev/null; then
    RUNNING_PID="$DISCOVERED_PID"
    echo "$RUNNING_PID" > "$PID_FILE"
  fi
fi

if [[ -n "$RUNNING_PID" ]]; then
  echo "Process: RUNNING (PID ${RUNNING_PID})"
elif [[ -f "$PID_FILE" ]]; then
  echo "Process: NOT RUNNING (stale PID file)"
else
  echo "Process: UNKNOWN (no PID file)"
fi

if [[ -f "$STATUS_FILE" ]]; then
  echo
  echo "Status file:"
  cat "$STATUS_FILE"
fi

if [[ -f "$EVENTS_FILE" ]]; then
  echo
  echo "Recent events:"
  tail -n 8 "$EVENTS_FILE"
fi

if [[ -f "$NOHUP_LOG" ]]; then
  echo
  echo "Recent nohup log lines:"
  tail -n 20 "$NOHUP_LOG"
fi
