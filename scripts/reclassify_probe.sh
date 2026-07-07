#!/usr/bin/env bash
set -uo pipefail

# Stability probe: reclassify the first N blocks of run 447 with Whisper on the
# CPU so the GPU has a single CUDA consumer (Ollama only). Validates the
# dual-context-crash theory before committing to all 20 blocks. Does NOT purge
# (run 447 is already empty after the crashed recovery attempt).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RADIO_CLASSIFIER_OLLAMA_HOST="${RADIO_CLASSIFIER_OLLAMA_HOST:-http://127.0.0.1:11435}"
DB_PATH="data/store/broadcast.db"
RUN_DIR="data/captures/continuous_20260607T192305Z_96x30m"
RUN_PK=447
LIMIT="${1:-2}"

echo "=== reclassify PROBE start $(date -u +%Y-%m-%dT%H:%M:%SZ) limit=${LIMIT} whisper=CPU/int8 ==="
if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "ERROR: Tier-3 LLM unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  exit 3
fi

mapfile -t BLOCKS < <(
  .venv/bin/python - "$RUN_DIR" "$LIMIT" <<'PY'
import json, glob, os, sys
run_dir, limit = sys.argv[1], int(sys.argv[2])
emitted = 0
for sc in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
    if emitted >= limit:
        break
    try:
        d = json.load(open(sc))
    except Exception:
        continue
    wav, start = d.get("wav_path"), d.get("capture_start_utc")
    if not wav or not start or not os.path.exists(wav) or os.path.getsize(wav) == 0:
        continue
    print(f"{wav}\t{start}")
    emitted += 1
PY
)

echo "--- probing ${#BLOCKS[@]} block(s) ---"
ok=0; fail=0; idx=0
for line in "${BLOCKS[@]}"; do
  idx=$((idx + 1))
  wav="${line%%$'\t'*}"
  start="${line##*$'\t'}"
  cls_log="data/logs/probe447_block$(printf '%03d' "$idx").log"
  echo "[$idx/${#BLOCKS[@]}] $(date -u +%H:%M:%SZ) start $(basename "$wav")"
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
    echo "    done: $(tail -1 "$cls_log")"
  else
    fail=$((fail + 1))
    echo "    FAILED (see $cls_log)" >&2
  fi
done

echo "=== probe done ok=${ok} fail=${fail} $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
.venv/bin/python - "$DB_PATH" "$RUN_PK" <<'PY'
import sqlite3, sys
db, run = sys.argv[1], int(sys.argv[2])
c = sqlite3.connect(db)
print(f"run {run} category breakdown:")
for cat, n in c.execute("SELECT category, COUNT(*) FROM broadcast_events WHERE capture_run_id=? GROUP BY category ORDER BY 2 DESC", (run,)):
    print(f"  {cat:<12} {n}")
c.close()
PY
