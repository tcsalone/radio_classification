#!/usr/bin/env bash
set -euo pipefail

# Sleep until a target local wall-clock time, then run morning_run_blocks.sh.
#
# Usage:
#   ./scripts/morning_run_at.sh             # defaults to 06:00 local
#   ./scripts/morning_run_at.sh 06:15       # custom HH:MM (24h, local time)
#   ./scripts/morning_run_at.sh 06:00 6     # 6x 30-minute blocks (3 hours)
#
# Recommended launch (survives shell close, keeps WSL alive):
#   nohup bash scripts/morning_run_at.sh 06:00 \
#     > data/logs/morning_at_$(date -u +%Y%m%dT%H%M%SZ).out 2>&1 &
#   disown
#
# Notes:
# - Requires the Windows host to stay powered on; disable Windows sleep.
# - WSL2 stays up as long as this process is running.
# - All script output is also tee'd by morning_run_blocks.sh into per-block logs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p data/logs

TARGET_HHMM="${1:-06:00}"
BLOCK_COUNT="${2:-${BLOCK_COUNT:-4}}"

now_epoch="$(date +%s)"
target_today_epoch="$(date -d "today ${TARGET_HHMM}" +%s)"
if (( target_today_epoch <= now_epoch )); then
    target_epoch="$(date -d "tomorrow ${TARGET_HHMM}" +%s)"
else
    target_epoch="$target_today_epoch"
fi

sleep_seconds=$(( target_epoch - now_epoch ))
human_target="$(date -d "@${target_epoch}")"

echo "morning_run_at: scheduled for ${human_target}"
echo "morning_run_at: sleeping ${sleep_seconds}s (until target)..."
echo "morning_run_at: block_count=${BLOCK_COUNT}"
echo "morning_run_at: pid=$$"

sleep "${sleep_seconds}"

echo
echo "morning_run_at: waking at $(date)"
echo "morning_run_at: launching scripts/morning_run_blocks.sh ${BLOCK_COUNT}"
echo

exec bash scripts/morning_run_blocks.sh "${BLOCK_COUNT}"
