#!/usr/bin/env bash
set -uo pipefail

# Chained recovery + capture, all on Whisper-CPU for GPU-PV stability:
#   1. Reclassify the 20 surviving WAVs from the crashed 48h run into run 447.
#   2. Run a fresh 48h capture (appended to the same long-term DB).
#   3. Run post-run cleanup (commercial dedupe, song dedupe, block-boundary
#      stitch, reports).
#
# Detached (nohup) so a Cursor crash never interrupts it. Stop only matters on
# `wsl --shutdown` / reboot.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RADIO_CLASSIFIER_OLLAMA_HOST="${RADIO_CLASSIFIER_OLLAMA_HOST:-http://127.0.0.1:11435}"
DB_PATH="data/store/broadcast.db"

echo "=== RECOVER+CAPTURE MASTER START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- preflight ---
if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "ABORT: Tier-3 LLM unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  exit 3
fi
echo "preflight: ollama OK ($RADIO_CLASSIFIER_OLLAMA_HOST)"

# --- phase 1: reclassify the crashed run's 20 WAVs (CPU Whisper) ---
echo "=== PHASE 1 reclassify run 447 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash scripts/reclassify_run447.sh
echo "=== PHASE 1 done rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- phase 2: fresh 48h capture (RTL required; CPU Whisper via script default) ---
if ! lsusb 2>/dev/null | grep -qiE "RTL2838|Realtek"; then
  echo "ABORT before capture: RTL dongle not visible (run usbipd attach --wsl --busid 5-3)" >&2
  exit 4
fi
echo "=== PHASE 2 capture 48h $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DB_PATH="$DB_PATH" bash scripts/capture_until_audio_hours.sh 48
echo "=== PHASE 2 capture exited rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# --- phase 3: cleanup + reports ---
echo "=== PHASE 3 cleanup $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
bash scripts/post_run_cleanup.sh "$DB_PATH"

echo "=== RECOVER+CAPTURE MASTER DONE rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
