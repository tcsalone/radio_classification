#!/usr/bin/env bash
set -euo pipefail

# Resilient bounded capture supervisor.
#
# Goal: collect N hours of actual audio, not merely run for N wall-clock hours.
# If rtl_fm / USB/IP drops mid-run, the completed and partial chunks are kept,
# classified, appended to the persistent DB, and the next iteration resumes
# until the accumulated sidecar durations reach the requested target.
#
# Usage:
#   ./scripts/capture_until_audio_hours.sh 20
#
# Environment:
#   DB_PATH=data/store/broadcast.db
#   BLOCK_SECONDS=1800
#   RESTART_BACKOFF_SECONDS=30
#   RETENTION_DAYS=7
#   FREQUENCY=105300000
#   DEVICE_INDEX=0
#   SAMPLE_RATE=48000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_HOURS="${1:-${TARGET_HOURS:-20}}"
DB_PATH="${DB_PATH:-data/store/broadcast.db}"
BLOCK_SECONDS="${BLOCK_SECONDS:-1800}"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-30}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"

mkdir -p data/logs

TARGET_SECONDS="$(
  .venv/bin/python - "$TARGET_HOURS" <<'PY'
from decimal import Decimal, InvalidOperation
import sys

try:
    hours = Decimal(sys.argv[1])
except (InvalidOperation, IndexError):
    raise SystemExit("TARGET_HOURS must be numeric, e.g. 20 or 1.5")
if hours <= 0:
    raise SystemExit("TARGET_HOURS must be > 0")
print(int(hours * Decimal(3600)))
PY
)"

if [[ "$TARGET_SECONDS" -le 0 ]]; then
  echo "capture_until_audio_hours: target resolved to zero seconds" >&2
  exit 2
fi

echo "capture_until_audio_hours: target_hours=$TARGET_HOURS"
echo "capture_until_audio_hours: target_seconds=$TARGET_SECONDS"
echo "capture_until_audio_hours: db=$DB_PATH"
echo "capture_until_audio_hours: block_seconds=$BLOCK_SECONDS"
echo "capture_until_audio_hours: restart_backoff_seconds=$RESTART_BACKOFF_SECONDS"
echo "capture_until_audio_hours: retention_days=$RETENTION_DAYS"

captured_seconds=0
iteration=0

sidecar_duration_sum() {
  local run_dir="$1"
  .venv/bin/python - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
total = 0.0
for sidecar in sorted(run_dir.glob("*.json")):
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        continue
    try:
        total += float(data.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        continue
print(int(total))
PY
}

while [[ "$captured_seconds" -lt "$TARGET_SECONDS" ]]; do
  iteration=$((iteration + 1))
  remaining=$((TARGET_SECONDS - captured_seconds))
  blocks=$(((remaining + BLOCK_SECONDS - 1) / BLOCK_SECONDS))
  iter_block_seconds=$(((remaining + blocks - 1) / blocks))
  iter_log="data/logs/capture_until_${TARGET_HOURS}h_iter${iteration}_$(date -u +%Y%m%dT%H%M%SZ).log"

  echo
  echo "=== resilient capture iteration ${iteration} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "captured_seconds=$captured_seconds"
  echo "remaining_seconds=$remaining"
  echo "iteration_blocks=$blocks"
  echo "iteration_block_seconds=$iter_block_seconds"
  echo "iteration_log=$iter_log"

  set +e
  RETENTION_DAYS="$RETENTION_DAYS" \
    BLOCK_SECONDS="$iter_block_seconds" \
    bash scripts/continuous_capture_blocks.sh "$blocks" --append-db "$DB_PATH" \
    2>&1 | tee "$iter_log"
  rc="${PIPESTATUS[0]}"
  set -e

  run_dir="$(
    awk -F= '/^run_dir=/{print $2; exit}' "$iter_log"
  )"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "capture_until_audio_hours: could not resolve run_dir from $iter_log (rc=$rc)" >&2
    echo "capture_until_audio_hours: sleeping ${RESTART_BACKOFF_SECONDS}s before retry" >&2
    sleep "$RESTART_BACKOFF_SECONDS"
    continue
  fi

  iteration_seconds="$(sidecar_duration_sum "$run_dir")"
  captured_seconds=$((captured_seconds + iteration_seconds))
  echo "capture_until_audio_hours: iteration_seconds=$iteration_seconds"
  echo "capture_until_audio_hours: captured_seconds=$captured_seconds/$TARGET_SECONDS"

  if [[ "$iteration_seconds" -le 0 ]]; then
    echo "capture_until_audio_hours: no audio captured; sleeping ${RESTART_BACKOFF_SECONDS}s before retry" >&2
    sleep "$RESTART_BACKOFF_SECONDS"
  fi
done

echo
echo "capture_until_audio_hours: target reached captured_seconds=$captured_seconds target_seconds=$TARGET_SECONDS"
echo "capture_until_audio_hours: recent runs:"
.venv/bin/python -m radio_classifier runs list --db-path "$DB_PATH" --limit 10
