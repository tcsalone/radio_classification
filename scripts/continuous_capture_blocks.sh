#!/usr/bin/env bash
set -euo pipefail

# Continuously capture one RTL-SDR stream into fixed WAV chunks while one
# classifier worker processes completed chunks into a single SQLite DB.
#
# Unlike morning_run_blocks.sh, this does not restart rtl_fm for each block.
# Capture is uninterrupted; classification can lag behind without losing audio.
#
# Usage:
#   ./scripts/continuous_capture_blocks.sh              # 4x 30-minute chunks
#   ./scripts/continuous_capture_blocks.sh 24           # 12 hours
#   ./scripts/continuous_capture_blocks.sh 24 --append-db data/store/broadcast.db
#
# Tunables:
#   BLOCK_SECONDS=1800
#   FREQUENCY=105300000
#   DEVICE_INDEX=0
#   SAMPLE_RATE=48000
#   RUN_ID=continuous_my_test   # override the auto-generated run id
#   APPEND_DB=data/store/broadcast.db  # reuse an existing long-term DB
#
# Run identity:
#   By default each invocation gets a unique run id that includes the UTC
#   start *time* (not just date), so two captures the same day will never
#   collide. The script refuses to start if the resolved run dir already exists.
#   By default it also refuses to use an existing run-specific DB. Pass
#   --append-db (or APPEND_DB) when you intentionally want to write a fresh run's
#   events into a long-term persistent SQLite store.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/continuous_capture_blocks.sh [BLOCK_COUNT] [--append-db PATH]

Examples:
  ./scripts/continuous_capture_blocks.sh
  ./scripts/continuous_capture_blocks.sh 24
  ./scripts/continuous_capture_blocks.sh 24 --append-db data/store/broadcast.db

Environment:
  BLOCK_SECONDS=1800
  FREQUENCY=105300000
  DEVICE_INDEX=0
  SAMPLE_RATE=48000
  RUN_ID=continuous_my_test
  APPEND_DB=data/store/broadcast.db
EOF
}

BLOCK_COUNT="${BLOCK_COUNT:-4}"
APPEND_DB="${APPEND_DB:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --append-db)
      if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "continuous_capture_blocks: --append-db requires a path" >&2
        exit 2
      fi
      APPEND_DB="$2"
      shift 2
      ;;
    --append-db=*)
      APPEND_DB="${1#--append-db=}"
      shift
      ;;
    -*)
      echo "continuous_capture_blocks: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      BLOCK_COUNT="$1"
      shift
      ;;
  esac
done

BLOCK_SECONDS="${BLOCK_SECONDS:-1800}"
FREQUENCY="${FREQUENCY:-105300000}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
SAMPLE_RATE="${SAMPLE_RATE:-48000}"
TOTAL_SECONDS=$((BLOCK_COUNT * BLOCK_SECONDS))

RUN_STARTED_ISO="$(date -u +%Y-%m-%dT%H:%M:%S.000Z)"
RUN_START_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-continuous_${RUN_START_UTC}_${BLOCK_COUNT}x$((BLOCK_SECONDS / 60))m}"
RUN_DIR="data/captures/${RUN_ID}"
APPEND_MODE=0
if [[ -n "$APPEND_DB" ]]; then
  DB_PATH="$APPEND_DB"
  APPEND_MODE=1
else
  DB_PATH="data/eval/${RUN_ID}.db"
fi
CAPTURE_LOG="data/logs/${RUN_ID}_capture.log"
SKIPS_LOG="data/logs/${RUN_ID}_skips.log"

# Refuse to start if anything from a prior run with the same id is already on
# disk. This makes the failure mode loud — operator must pick a fresh RUN_ID
# or move the old artifacts aside — instead of silently mixing two runs.
COLLISIONS=()
if [[ -e "$RUN_DIR" ]]; then COLLISIONS+=("$RUN_DIR"); fi
if [[ "$APPEND_MODE" -eq 0 && -e "$DB_PATH" ]]; then COLLISIONS+=("$DB_PATH"); fi
if (( ${#COLLISIONS[@]} > 0 )); then
  {
    echo "continuous_capture_blocks: run id '$RUN_ID' already has artifacts on disk:"
    for path in "${COLLISIONS[@]}"; do
      echo "  $path"
    done
    echo "Pick a different RUN_ID env var, or move/delete the existing files."
  } >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "data/eval" "data/logs" "$(dirname "$DB_PATH")"

echo "run_id=$RUN_ID"
echo "run_start_utc=$RUN_START_UTC"
echo "run_dir=$RUN_DIR"
echo "db_path=$DB_PATH"
echo "append_db=$APPEND_MODE"
echo "block_count=$BLOCK_COUNT"
echo "block_seconds=$BLOCK_SECONDS"
echo "total_seconds=$TOTAL_SECONDS"
echo "capture_log=$CAPTURE_LOG"
echo "skips_log=$SKIPS_LOG"

.venv/bin/python -m radio_classifier db init --db-path "$DB_PATH" >/dev/null
CAPTURE_RUN_DB_ID="$(
  .venv/bin/python -m radio_classifier runs start \
    --db-path "$DB_PATH" \
    --run-id "$RUN_ID" \
    --started-utc "$RUN_STARTED_ISO"
)"
echo "capture_run_id=$CAPTURE_RUN_DB_ID"

echo
echo "=== continuous capture start ==="
(
  .venv/bin/python -m radio_classifier capture chunks \
    --frequency "$FREQUENCY" \
    --device-index "$DEVICE_INDEX" \
    --sample-rate "$SAMPLE_RATE" \
    --chunk-seconds "$BLOCK_SECONDS" \
    --duration-limit "$TOTAL_SECONDS" \
    --out-dir "$RUN_DIR" \
    --run-id "$RUN_ID" \
    2>&1 | tee "$CAPTURE_LOG"
) &
CAPTURE_PID="$!"

COMPLETED=0
CLASSIFY_FAILED=0

metadata_value() {
  local json_path="$1"
  local key="$2"
  .venv/bin/python - "$json_path" "$key" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
value = data.get(key)
if value is None:
    raise SystemExit(f"missing key {key!r} in {path}")
print(value)
PY
}

wait_for_sidecar() {
  local sidecar="$1"
  while [[ ! -s "$sidecar" ]]; do
    if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
      if [[ ! -s "$sidecar" ]]; then
        echo "capture process ended before sidecar appeared: $sidecar" >&2
        return 1
      fi
    fi
    sleep 2
  done
}

for i in $(seq 1 "$BLOCK_COUNT"); do
  block_name="$(printf '%s_block%04d' "$RUN_ID" "$i")"
  sidecar="${RUN_DIR}/${block_name}.json"
  cls_log="data/logs/${RUN_ID}_block${i}_classify.log"

  echo
  echo "=== block ${i}/${BLOCK_COUNT} wait for captured chunk ==="
  echo "sidecar=$sidecar"
  if ! wait_for_sidecar "$sidecar"; then
    CLASSIFY_FAILED=$((CLASSIFY_FAILED + 1))
    echo "block ${i}: capture chunk missing; skipping classify" | tee -a "$SKIPS_LOG" >&2
    continue
  fi

  wav_path="$(metadata_value "$sidecar" wav_path)"
  capture_start_utc="$(metadata_value "$sidecar" capture_start_utc)"
  complete="$(metadata_value "$sidecar" complete)"

  if [[ "$complete" != "True" && "$complete" != "true" ]]; then
    echo "block ${i}: chunk is partial; classifying anyway after capture ended" | tee -a "$SKIPS_LOG" >&2
  fi
  if [[ ! -s "$wav_path" ]]; then
    CLASSIFY_FAILED=$((CLASSIFY_FAILED + 1))
    echo "block ${i}: WAV missing/empty: $wav_path" | tee -a "$SKIPS_LOG" >&2
    continue
  fi

  echo
  echo "=== block ${i}/${BLOCK_COUNT} classify ==="
  echo "wav=$wav_path"
  echo "capture_start_utc=$capture_start_utc"
  echo "db=$DB_PATH"
  echo "log=$cls_log"
  if .venv/bin/python -m radio_classifier classify \
      -i "$wav_path" \
      --capture-start-utc "$capture_start_utc" \
      --enable-shazam \
      --persist \
      --db-path "$DB_PATH" \
      --capture-run-id "$CAPTURE_RUN_DB_ID" \
      --progress \
      2>&1 | tee "$cls_log"; then
    COMPLETED=$((COMPLETED + 1))
    if [[ -x scripts/prune_old_wavs.sh ]]; then
      RETENTION_DAYS="${RETENTION_DAYS:-7}" scripts/prune_old_wavs.sh data/captures || true
    fi
  else
    CLASSIFY_FAILED=$((CLASSIFY_FAILED + 1))
    echo "block ${i}: classify FAILED (see ${cls_log})" | tee -a "$SKIPS_LOG" >&2
  fi
done

echo
echo "=== waiting for capture process ==="
if ! wait "$CAPTURE_PID"; then
  echo "continuous capture process failed; see $CAPTURE_LOG" | tee -a "$SKIPS_LOG" >&2
fi

echo
echo "=== run summary ==="
echo "blocks classified: ${COMPLETED}/${BLOCK_COUNT}"
echo "blocks with classify failure: ${CLASSIFY_FAILED}"
if [[ -s "$SKIPS_LOG" ]]; then
  echo "skip log: $SKIPS_LOG"
fi
.venv/bin/python -m radio_classifier runs end \
  --db-path "$DB_PATH" \
  --run-id "$RUN_ID" \
  --ended-utc "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" >/dev/null

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
  echo "no blocks classified; check $CAPTURE_LOG and $SKIPS_LOG" >&2
  exit 1
fi
