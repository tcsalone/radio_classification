#!/usr/bin/env bash
set -uo pipefail

# One-off recovery: reclassify the 2026-06-07 partial 48h capture (run 447)
# with Tier-3 reachable. The original run dropped all COMMERCIAL/DJ/STATION/
# PSA speech to "unknown" because RADIO_CLASSIFIER_OLLAMA_HOST was unset and the
# classifier defaulted to the dead :11434. The WAVs survived, so we purge the
# song-only events and re-run the full funnel into the same capture_run_id.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RADIO_CLASSIFIER_OLLAMA_HOST="${RADIO_CLASSIFIER_OLLAMA_HOST:-http://127.0.0.1:11435}"
DB_PATH="data/store/broadcast.db"
RUN_DIR="data/captures/continuous_20260607T192305Z_96x30m"
RUN_PK=447

echo "=== reclassify run ${RUN_PK} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "tier3_llm=$RADIO_CLASSIFIER_OLLAMA_HOST"
if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Tier-3 LLM unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  exit 3
fi

echo "--- purging existing run ${RUN_PK} events ---"
.venv/bin/python - "$DB_PATH" "$RUN_PK" <<'PY'
import sqlite3, sys
db, run = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
before = c.execute("SELECT COUNT(*) FROM broadcast_events WHERE capture_run_id=?", (run,)).fetchone()[0]
c.execute("DELETE FROM broadcast_events WHERE capture_run_id=?", (run,))
c.commit()
print(f"deleted {before} event(s) from run {run}")
c.close()
PY

# Emit "wav<TAB>capture_start_utc" for each valid, non-empty block in order.
mapfile -t BLOCKS < <(
  .venv/bin/python - "$RUN_DIR" <<'PY'
import json, glob, os, sys
run_dir = sys.argv[1]
for sc in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    try:
        d = json.load(open(sc))
    except Exception:
        continue
    wav = d.get("wav_path"); start = d.get("capture_start_utc")
    if not wav or not start:
        continue
    if not os.path.exists(wav) or os.path.getsize(wav) == 0:
        continue
    print(f"{wav}\t{start}")
PY
)

echo "--- reclassifying ${#BLOCKS[@]} block(s) ---"
ok=0; fail=0; idx=0
for line in "${BLOCKS[@]}"; do
  idx=$((idx + 1))
  wav="${line%%$'\t'*}"
  start="${line##*$'\t'}"
  cls_log="data/logs/reclass447_block$(printf '%03d' "$idx").log"
  echo "[$idx/${#BLOCKS[@]}] $(date -u +%H:%M:%SZ) $(basename "$wav")"
  if .venv/bin/python -m radio_classifier classify \
      -i "$wav" \
      --capture-start-utc "$start" \
      --whisper-device cpu \
      --whisper-compute-type int8 \
      --enable-shazam \
      --persist \
      --db-path "$DB_PATH" \
      --capture-run-id "$RUN_PK" \
      --progress > "$cls_log" 2>&1; then
    ok=$((ok + 1))
    tail -1 "$cls_log" | sed 's/^/    /'
  else
    fail=$((fail + 1))
    echo "    FAILED (see $cls_log)" >&2
  fi
done

echo "=== reclassify done ok=${ok} fail=${fail} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python - "$DB_PATH" "$RUN_PK" <<'PY'
import sqlite3, sys
db, run = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
print(f"run {run} category breakdown after reclassify:")
for cat, n in c.execute("SELECT category, COUNT(*) FROM broadcast_events WHERE capture_run_id=? GROUP BY category ORDER BY 2 DESC", (run,)):
    print(f"  {cat:<12} {n}")
c.close()
PY
