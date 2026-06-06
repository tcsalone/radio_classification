#!/usr/bin/env bash
set -euo pipefail

# Automated Post-Run Cleanup and Maintenance Script
#
# Runs a full suite of safe DB maintenance, deduplication, and reporting tasks.
# Can be chained after a capture run finishes.
#
# Usage:
#   ./scripts/post_run_cleanup.sh [DB_PATH]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DB_PATH="${1:-data/store/broadcast.db}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
DATE_STR="$(date -u +'%Y-%m-%d %H:%M:%S UTC')"

echo "=== Starting Post-Run Cleanup: ${DATE_STR} ==="
echo "Target Database: ${DB_PATH}"

# 1. Back up the database
echo "--- 1. Creating Database Backup ---"
mkdir -p data/backups
BACKUP_PATH="data/backups/broadcast-${TS}.db"
.venv/bin/python -c "
import sqlite3
src = sqlite3.connect('${DB_PATH}')
bak = sqlite3.connect('${BACKUP_PATH}')
with bak:
    src.backup(bak)
src.close(); bak.close()
"
echo "Backup written to ${BACKUP_PATH}"

# Integrity check
INTEGRITY="$(.venv/bin/python -c "import sqlite3; db = sqlite3.connect('${DB_PATH}'); print(db.execute('PRAGMA integrity_check;').fetchone()[0]); db.close()")"
echo "Integrity check: ${INTEGRITY}"
if [ "${INTEGRITY}" != "ok" ]; then
    echo "CRITICAL: Database integrity check failed on ${DB_PATH}. Aborting mutation steps!" >&2
    exit 1
fi

# 2. Merge boundary commercials (overlapping windows)
echo "--- 2. Merging Boundary Commercials ---"
.venv/bin/python -m radio_classifier commercials merge-boundaries --db-path "${DB_PATH}" --apply

# 3. Deterministic brand backfill
echo "--- 3. Backfilling Brands (Deterministic) ---"
.venv/bin/python -m radio_classifier commercials backfill-brands --db-path "${DB_PATH}" --apply

# 4. Deduplicate commercials
echo "--- 4. Deduplicating Commercials ---"
.venv/bin/python -m radio_classifier commercials dedupe --db-path "${DB_PATH}" --apply

# 5. Deduplicate songs
echo "--- 5. Deduplicating Songs ---"
.venv/bin/python -m radio_classifier songs dedupe --db-path "${DB_PATH}"

# 6. Enrich song release dates from MusicBrainz
echo "--- 6. Enriching Song Release Dates ---"
set +e
.venv/bin/python -m radio_classifier songs enrich-releases --db-path "${DB_PATH}"
set -e

# 7. Prune old WAVs (retains sidecars)
echo "--- 7. Pruning WAVs Older than 7 Days ---"
if [ -f "./scripts/prune_old_wavs.sh" ]; then
    ./scripts/prune_old_wavs.sh data/captures
else
    echo "Warning: ./scripts/prune_old_wavs.sh not found, skipping."
fi

# 8. Generate Reports
echo "--- 8. Generating Reports ---"
mkdir -p data/reports
.venv/bin/python -m radio_classifier report dashboard --db-path "${DB_PATH}" --since 24h --out data/reports/dashboard.html
.venv/bin/python -m radio_classifier report artist-plays --db-path "${DB_PATH}" --since 24h --top 3 --out data/reports/artist-plays.html

# 9. Print Discoveries list so the operator can see what was found
echo "--- 9. Active Shazam Discoveries Outstanding ---"
.venv/bin/python -m radio_classifier songs discovered --db-path "${DB_PATH}" --since 24h --top 20 --min-plays 1 | tee "data/reports/discovered-summary-${TS}.txt"

echo "=== Post-Run Cleanup Completed Successfully: $(date -u +'%Y-%m-%d %H:%M:%S UTC') ==="
