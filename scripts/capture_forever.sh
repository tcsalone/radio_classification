#!/usr/bin/env bash
set -euo pipefail

# Manual foreground supervisor for long-term capture.
#
# This script intentionally does not install itself as a service. Start it when
# the host is awake and you want collection running; Ctrl-C stops after the
# current child process receives the signal.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_PATH="${DB_PATH:-data/store/broadcast.db}"
BLOCK_COUNT="${BLOCK_COUNT:-48}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-30}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

echo "capture_forever: db=$DB_PATH"
echo "capture_forever: block_count=$BLOCK_COUNT"
echo "capture_forever: restart_backoff_seconds=$RESTART_BACKOFF_SECONDS"
echo "capture_forever: retention_days=$RETENTION_DAYS"

stop_requested=0
child_pid=""

on_int() {
  stop_requested=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -INT "$child_pid" 2>/dev/null || true
  fi
}
trap on_int INT TERM

while [[ "$stop_requested" -eq 0 ]]; do
  echo
  echo "=== capture_forever iteration start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  (
    RETENTION_DAYS="$RETENTION_DAYS" \
    bash scripts/continuous_capture_blocks.sh "$BLOCK_COUNT" --append-db "$DB_PATH"
  ) &
  child_pid="$!"
  set +e
  wait "$child_pid"
  rc="$?"
  set -e
  child_pid=""

  if [[ "$stop_requested" -ne 0 ]]; then
    echo "capture_forever: stop requested"
    break
  fi

  if [[ "$rc" -eq 0 ]]; then
    echo "capture_forever: iteration completed; starting next iteration"
    continue
  fi

  echo "capture_forever: iteration failed rc=$rc; sleeping ${RESTART_BACKOFF_SECONDS}s before restart" >&2
  sleep "$RESTART_BACKOFF_SECONDS"
done
