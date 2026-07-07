#!/usr/bin/env bash
set -uo pipefail

# Weekend bounded-capture master, appended to the long-term DB.
#
# Plan: capture TARGET_HOURS of *audio* (real-time, so wall-clock >= audio hours),
# then run the standard post-run cleanup, then regenerate the HTML reports over
# the FULL window of this run (so they are not empty under the default 24h
# window when reviewed later).
#
# Detached via nohup at launch so a Cursor disconnect never interrupts it. A
# `wsl --shutdown` / host reboot WILL still kill it (whole-VM event).
#
# Usage:
#   nohup bash scripts/weekend_capture.sh 88 > data/logs/weekend_capture_master.log 2>&1 &

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_HOURS="${1:-88}"
DB_PATH="data/store/broadcast.db"
export RADIO_CLASSIFIER_OLLAMA_HOST="${RADIO_CLASSIFIER_OLLAMA_HOST:-http://127.0.0.1:11435}"

# Long unattended run: tolerate a lengthy capture outage (e.g. a USB/IP drop
# that Windows --auto-attach restores). The stall-watchdog in
# continuous_capture_blocks.sh kills a wedged rtl_fm quickly; with 60s backoff,
# 120 zero-progress iterations ~= up to ~2h of device outage before giving up.
export MAX_ZERO_PROGRESS_ITERS="${MAX_ZERO_PROGRESS_ITERS:-120}"
export RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-60}"

# How long to wait for the RTL dongle to be attached from Windows before giving
# up (lets us "queue" the run now and attach the dongle within the window).
RTL_WAIT_SECONDS="${RTL_WAIT_SECONDS:-3600}"

mkdir -p data/logs
START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== WEEKEND CAPTURE MASTER START ${START_UTC} (target_hours=${TARGET_HOURS}) ==="

# --- preflight: Tier-3 LLM ---
if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "ABORT: Tier-3 LLM unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  exit 3
fi
echo "preflight: ollama OK ($RADIO_CLASSIFIER_OLLAMA_HOST)"

# --- preflight: wait for RTL dongle (attach from Windows: usbipd attach --wsl --busid <id>) ---
waited=0
until lsusb 2>/dev/null | grep -qiE "RTL2838|Realtek"; do
  if [[ "$waited" -ge "$RTL_WAIT_SECONDS" ]]; then
    echo "ABORT: RTL dongle not visible after ${RTL_WAIT_SECONDS}s." >&2
    echo "  From Windows PowerShell: usbipd list ; usbipd attach --wsl --busid <BUSID>" >&2
    exit 4
  fi
  if [[ "$waited" -eq 0 ]]; then
    echo "waiting for RTL dongle... attach it from Windows: usbipd attach --wsl --busid <BUSID>"
  fi
  sleep 15
  waited=$((waited + 15))
done
echo "preflight: RTL dongle visible (waited ${waited}s)"

# --- phase 1: bounded capture (CPU Whisper via continuous_capture_blocks default) ---
echo "=== PHASE 1 capture ${TARGET_HOURS}h $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DB_PATH="$DB_PATH" bash scripts/capture_until_audio_hours.sh "$TARGET_HOURS"
echo "=== PHASE 1 capture exited rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- phase 2: standard cleanup + reports ---
echo "=== PHASE 2 cleanup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash scripts/post_run_cleanup.sh "$DB_PATH"

# --- phase 3: regenerate reports over THIS run's full window (avoid empty 24h reports) ---
END_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== PHASE 3 full-window reports ${START_UTC} -> ${END_UTC} ==="
.venv/bin/python -m radio_classifier report dashboard --db-path "$DB_PATH" \
  --from "$START_UTC" --to "$END_UTC" --out data/reports/dashboard.html || true
.venv/bin/python -m radio_classifier report artist-plays --db-path "$DB_PATH" \
  --from "$START_UTC" --to "$END_UTC" --top 3 --out data/reports/artist-plays.html || true

echo "=== WEEKEND CAPTURE MASTER DONE rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
