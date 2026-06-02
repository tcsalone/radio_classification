#!/usr/bin/env bash
set -euo pipefail

# Prune old completed capture WAVs while keeping JSON sidecars for audit.
#
# Usage:
#   RETENTION_DAYS=7 ./scripts/prune_old_wavs.sh data/captures

ROOT="${1:-data/captures}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" - "$ROOT" "$RETENTION_DAYS" <<'PY'
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
retention_days = float(sys.argv[2])
cutoff = time.time() - retention_days * 24 * 60 * 60

if not root.exists():
    raise SystemExit(0)

deleted = 0
for wav in root.glob("*/*.wav"):
    try:
        if wav.stat().st_mtime >= cutoff:
            continue
    except OSError:
        continue
    sidecar = wav.with_suffix(".json")
    if not sidecar.is_file():
        continue
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if metadata.get("complete") is not True:
        continue
    try:
        wav.unlink()
        deleted += 1
    except OSError as exc:
        print(f"prune_old_wavs: failed to delete {wav}: {exc}", file=sys.stderr)

print(f"prune_old_wavs: deleted={deleted} root={root} retention_days={retention_days:g}")
PY
