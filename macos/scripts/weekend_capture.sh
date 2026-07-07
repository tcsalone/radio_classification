#!/usr/bin/env bash
set -uo pipefail

# macOS fork of scripts/weekend_capture.sh
#
# Bounded capture + post-run cleanup + full-window reports.
#
# Usage:
#   nohup bash macos/scripts/weekend_capture.sh 88 > data/logs/mac_weekend_master.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
source "$ROOT_DIR/macos/env.defaults"

TARGET_HOURS="${1:-88}"
DB_PATH="data/store/broadcast.db"

export MAX_ZERO_PROGRESS_ITERS="${MAX_ZERO_PROGRESS_ITERS:-120}"
export RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-60}"

RTL_WAIT_SECONDS="${RTL_WAIT_SECONDS:-3600}"

mkdir -p data/logs
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== MAC WEEKEND CAPTURE MASTER START ${START_UTC} (target_hours=${TARGET_HOURS}) ==="

if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "ABORT: Tier-3 LLM unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  echo "Start Ollama.app or: ollama serve" >&2
  exit 3
fi
echo "preflight: ollama OK ($RADIO_CLASSIFIER_OLLAMA_HOST)"

waited=0
until rtl_test -t >/dev/null 2>&1; do
  if [[ "$waited" -ge "$RTL_WAIT_SECONDS" ]]; then
    echo "ABORT: RTL dongle not detected after ${RTL_WAIT_SECONDS}s." >&2
    echo "  Plug the dongle directly into USB (avoid unpowered hubs)." >&2
    echo "  Verify: rtl_test -t" >&2
    exit 4
  fi
  if [[ "$waited" -eq 0 ]]; then
    echo "waiting for RTL dongle... plug it in and ensure rtl_test -t succeeds"
  fi
  sleep 15
  waited=$((waited + 15))
done
echo "preflight: RTL dongle OK (waited ${waited}s)"

echo "=== PHASE 1 capture ${TARGET_HOURS}h $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DB_PATH="$DB_PATH" bash macos/scripts/capture_until_audio_hours.sh "$TARGET_HOURS"
echo "=== PHASE 1 capture exited rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

echo "=== PHASE 2 cleanup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash macos/scripts/post_run_cleanup.sh "$DB_PATH"

END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== PHASE 3 full-window reports ${START_UTC} -> ${END_UTC} ==="
.venv/bin/python -m radio_classifier report dashboard --db-path "$DB_PATH" \
  --from "$START_UTC" --to "$END_UTC" --out data/reports/dashboard.html || true
.venv/bin/python -m radio_classifier report artist-plays --db-path "$DB_PATH" \
  --from "$START_UTC" --to "$END_UTC" --top 3 --out data/reports/artist-plays.html || true

echo "=== MAC WEEKEND CAPTURE MASTER DONE rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
