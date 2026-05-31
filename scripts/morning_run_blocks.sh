#!/usr/bin/env bash
set -euo pipefail

# Capture consecutive blocks and classify into one SQLite DB.
#
# Usage:
#   cd /home/eamon/dev/radio-classifier
#   ./scripts/morning_run_blocks.sh              # default: 4x 30-minute blocks
#   ./scripts/morning_run_blocks.sh 6            # 6x 30-minute blocks (3 hours)
#
# Notes:
# - Requires rtl_fm + RTL-SDR access for the capture step.
# - Classification uses the offline pipeline with Tier 1 batch mode and progress.
# - Per-block resilience: if a single block's capture or classify step fails
#   (e.g. RTL-SDR not attached after a sleep/restart), it is logged to
#   data/logs/<run-id>_skips.log and the loop continues with the next block.
#   This avoids losing all subsequent blocks to one transient device hiccup.
# - By default, classification is pipelined: while block N+1 is capturing,
#   block N can classify in the background. Set CLASSIFY_PARALLEL=0 to restore
#   the old capture-then-classify sequence.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BLOCK_COUNT="${1:-${BLOCK_COUNT:-4}}"
BLOCK_SECONDS="${BLOCK_SECONDS:-1800}"
CLASSIFY_PARALLEL="${CLASSIFY_PARALLEL:-1}"

RUN_DATE_UTC="$(date -u +%Y%m%d)"
RUN_ID="morning_${RUN_DATE_UTC}_${BLOCK_COUNT}x$((BLOCK_SECONDS / 60))m"
RUN_DIR="data/captures/${RUN_ID}"
DB_PATH="data/eval/${RUN_ID}.db"
SKIPS_LOG="data/logs/${RUN_ID}_skips.log"

mkdir -p "$RUN_DIR" "data/eval" "data/logs"

echo "run_dir=$RUN_DIR"
echo "db_path=$DB_PATH"
echo "block_count=$BLOCK_COUNT"
echo "block_seconds=$BLOCK_SECONDS"
echo "classify_parallel=$CLASSIFY_PARALLEL"
echo "skips_log=$SKIPS_LOG"

.venv/bin/python -m radio_classifier db init --db-path "$DB_PATH" >/dev/null

# Counters for the final summary so an operator can tell at a glance whether
# the run was clean or partially degraded.
COMPLETED=0
CAPTURE_FAILED=0
CLASSIFY_FAILED=0
PENDING_CLASSIFY_PID=""
PENDING_CLASSIFY_BLOCK=""
PENDING_CLASSIFY_LOG=""
PENDING_CLASSIFY_TS=""

wait_for_pending_classify() {
  if [[ -z "$PENDING_CLASSIFY_PID" ]]; then
    return 0
  fi

  echo
  echo "=== block ${PENDING_CLASSIFY_BLOCK}/${BLOCK_COUNT} classify wait ==="
  echo "log=$PENDING_CLASSIFY_LOG"
  if wait "$PENDING_CLASSIFY_PID"; then
    COMPLETED=$((COMPLETED + 1))
  else
    CLASSIFY_FAILED=$((CLASSIFY_FAILED + 1))
    echo "[${PENDING_CLASSIFY_TS}] block ${PENDING_CLASSIFY_BLOCK}: classify FAILED (see ${PENDING_CLASSIFY_LOG})" \
      | tee -a "$SKIPS_LOG" >&2
  fi

  PENDING_CLASSIFY_PID=""
  PENDING_CLASSIFY_BLOCK=""
  PENDING_CLASSIFY_LOG=""
  PENDING_CLASSIFY_TS=""
}

start_classify() {
  local block_index="$1"
  local wav_path="$2"
  local cls_log="$3"
  local block_start_ts="$4"

  echo
  echo "=== block ${block_index}/${BLOCK_COUNT} classify (into one DB) ==="
  echo "db=$DB_PATH"
  echo "log=$cls_log"

  if [[ "$CLASSIFY_PARALLEL" == "1" ]]; then
    (
      .venv/bin/python -m radio_classifier classify \
        -i "$wav_path" \
        --enable-shazam \
        --persist \
        --db-path "$DB_PATH" \
        --progress \
        2>&1 | tee "$cls_log"
    ) &
    PENDING_CLASSIFY_PID="$!"
    PENDING_CLASSIFY_BLOCK="$block_index"
    PENDING_CLASSIFY_LOG="$cls_log"
    PENDING_CLASSIFY_TS="$block_start_ts"
  else
    if .venv/bin/python -m radio_classifier classify \
          -i "$wav_path" \
          --enable-shazam \
          --persist \
          --db-path "$DB_PATH" \
          --progress \
          2>&1 | tee "$cls_log"; then
      COMPLETED=$((COMPLETED + 1))
    else
      CLASSIFY_FAILED=$((CLASSIFY_FAILED + 1))
      echo "[${block_start_ts}] block ${block_index}: classify FAILED (see ${cls_log})" \
        | tee -a "$SKIPS_LOG" >&2
    fi
  fi
}

for i in $(seq 1 "$BLOCK_COUNT"); do
  WAV_PATH="${RUN_DIR}/block${i}.wav"
  CAP_LOG="data/logs/${RUN_ID}_block${i}_capture.log"
  CLS_LOG="data/logs/${RUN_ID}_block${i}_classify.log"
  BLOCK_START_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  echo
  echo "=== block ${i}/${BLOCK_COUNT} capture ($((BLOCK_SECONDS / 60))m) ==="
  echo "wav=$WAV_PATH"
  echo "log=$CAP_LOG"

  # Capture only (disable tiers to minimize load during RF capture).
  # ``if !`` gates the failure so ``set -e`` does NOT abort the whole script
  # when a single block fails — we want the next block to still get a chance.
  if ! .venv/bin/python -m radio_classifier ingest \
        --duration-limit "$BLOCK_SECONDS" \
        --capture-wav "$WAV_PATH" \
        --no-tier1 --no-tier2 --no-tier3 \
        2>&1 | tee "$CAP_LOG"; then
    CAPTURE_FAILED=$((CAPTURE_FAILED + 1))
    echo "[${BLOCK_START_TS}] block ${i}: capture FAILED — skipping classify (see ${CAP_LOG})" \
      | tee -a "$SKIPS_LOG" >&2
    continue
  fi

  if [[ ! -s "$WAV_PATH" ]]; then
    CAPTURE_FAILED=$((CAPTURE_FAILED + 1))
    echo "[${BLOCK_START_TS}] block ${i}: capture produced no/empty WAV — skipping classify" \
      | tee -a "$SKIPS_LOG" >&2
    continue
  fi

  # Keep at most one classifier alive. In parallel mode this wait usually
  # happens after the next block's capture, so capture and classify overlap.
  wait_for_pending_classify
  start_classify "$i" "$WAV_PATH" "$CLS_LOG" "$BLOCK_START_TS"
done

wait_for_pending_classify

echo
echo "=== run summary ==="
echo "blocks completed: ${COMPLETED}/${BLOCK_COUNT}"
echo "blocks with capture failure: ${CAPTURE_FAILED}"
echo "blocks with classify failure: ${CLASSIFY_FAILED}"
if [[ -s "$SKIPS_LOG" ]]; then
  echo "skip log: $SKIPS_LOG"
fi

# Only print reports if at least one block made it into the DB. Running them
# against an empty DB just dumps "(no rows)" tables five times.
if (( COMPLETED > 0 )); then
  echo
  echo "=== combined reports (same DB) ==="
  .venv/bin/python -m radio_classifier report summary     --db-path "$DB_PATH" --since 1d
  .venv/bin/python -m radio_classifier report commercials --db-path "$DB_PATH" --since 1d --top 25
  .venv/bin/python -m radio_classifier report brands      --db-path "$DB_PATH" --since 1d --top 25
  .venv/bin/python -m radio_classifier report songs       --db-path "$DB_PATH" --since 1d --top 25
  .venv/bin/python -m radio_classifier report artists     --db-path "$DB_PATH" --since 1d --top 25
  .venv/bin/python -m radio_classifier songs discovered   --db-path "$DB_PATH" --since 1d
else
  echo
  echo "no blocks completed — skipping reports. Check $SKIPS_LOG and the per-block logs." >&2
  exit 1
fi

