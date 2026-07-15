#!/usr/bin/env bash
set -euo pipefail

# macOS hardware validation checklist (run on the M4 MacBook before long captures).
#
# Usage:
#   source macos/env.defaults
#   bash macos/scripts/validate.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "validate.sh: intended for macOS (uname=$(uname -s)); run unit tests on Linux instead" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$ROOT_DIR/macos/env.defaults"

echo "=== Step 1: preflight ==="
bash macos/scripts/preflight.sh

echo
echo "=== Step 2: prereq-check ==="
.venv/bin/python -m radio_classifier prereq-check --ollama

echo
echo "=== Step 3: rtl_test ==="
rtl_test -t

echo
echo "=== Step 4: 2-minute capture smoke ==="
SMOKE_DIR="data/captures/mac_validate_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="$(basename "$SMOKE_DIR")"
mkdir -p "$SMOKE_DIR"
.venv/bin/python -m radio_classifier capture chunks \
  --frequency "${FREQUENCY:-105300000}" \
  --device-index "${DEVICE_INDEX:-0}" \
  --sample-rate "${SAMPLE_RATE:-48000}" \
  --chunk-seconds 120 \
  --duration-limit 120 \
  --out-dir "$SMOKE_DIR" \
  --run-id "$RUN_ID"

WAV="$(ls -t "$SMOKE_DIR"/*.wav | head -1)"
JSON="$(ls -t "$SMOKE_DIR"/*.json | head -1)"
START="$(.venv/bin/python -c "import json; print(json.load(open('$JSON'))['capture_start_utc'])")"

echo
echo "=== Step 5: classify smoke chunk ==="
.venv/bin/python -m radio_classifier classify \
  -i "$WAV" \
  --capture-start-utc "$START" \
  --whisper-backend "${WHISPER_BACKEND:-faster-whisper}" \
  --whisper-model "${WHISPER_MODEL:-medium.en}" \
  --whisper-device "${WHISPER_DEVICE:-cpu}" \
  --whisper-compute-type "${WHISPER_COMPUTE_TYPE:-int8}" \
  --enable-shazam \
  --persist \
  --db-path data/store/broadcast.db \
  --progress

echo
echo "=== Step 6: 30-minute audio-hours supervisor ==="
echo "Starting capture_until_audio_hours.sh 0.5 ..."
bash macos/scripts/capture_until_audio_hours.sh 0.5

echo
echo "validate.sh: all steps completed successfully"
