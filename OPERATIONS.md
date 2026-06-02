# Operations

This guide covers the current long-term capture workflow. The persistent
SQLite store is the source of truth for ongoing station metrics.

## Persistent Store

The default database path is:

```bash
data/store/broadcast.db
```

Most `radio-classifier` commands now use that path when `--db-path` is omitted.
Pass `--db-path` explicitly when inspecting an older per-run database under
`data/eval/`.

To initialize or migrate the store:

```bash
.venv/bin/python -m radio_classifier db init --db-path data/store/broadcast.db
```

Schema v3 migration runs automatically the first time an older database is
opened. It adds capture-run provenance and backfills existing events into a
synthetic `legacy_pre_v3` run.

Verify the schema version:

```bash
sqlite3 data/store/broadcast.db \
  "SELECT value FROM schema_meta WHERE key = 'version';"
```

Verify recent capture runs:

```bash
.venv/bin/python -m radio_classifier report runs --since 7d
```

## Starting Capture

Start the foreground supervisor from the repository root:

```bash
./scripts/capture_forever.sh
```

Defaults:

- Database: `data/store/broadcast.db`
- Blocks per supervisor iteration: `48`
- Block length: `1800` seconds, inherited by `continuous_capture_blocks.sh`
- Restart backoff after failure: `30` seconds
- WAV retention: `7` days

Useful overrides:

```bash
DB_PATH=data/store/broadcast.db \
BLOCK_COUNT=24 \
BLOCK_SECONDS=1800 \
FREQUENCY=105300000 \
DEVICE_INDEX=0 \
RETENTION_DAYS=7 \
./scripts/capture_forever.sh
```

`capture_forever.sh` runs in the foreground by design. It restarts a failed
capture/classify iteration after the configured backoff and opens a fresh
`capture_runs` row for the next iteration.

## Stopping Capture

Press `Ctrl-C` in the terminal running `capture_forever.sh`.

The signal is forwarded to the child `continuous_capture_blocks.sh` process.
That child closes the current `capture_runs` row before exiting when it reaches
its normal cleanup path. If the host sleeps, WSL stops, or the process is killed
hard, the database remains usable; the affected run may simply have a null
`ended_utc`.

To close a run manually if needed:

```bash
.venv/bin/python -m radio_classifier runs end \
  --db-path data/store/broadcast.db \
  --run-id RUN_ID
```

List recent run ids with:

```bash
.venv/bin/python -m radio_classifier runs list --db-path data/store/broadcast.db
```

## One-Off Runs

Use `continuous_capture_blocks.sh` directly for bounded experiments:

```bash
./scripts/continuous_capture_blocks.sh 4
```

By default this writes a fresh per-run database under `data/eval/`. To append a
bounded run into the long-term store:

```bash
./scripts/continuous_capture_blocks.sh 24 --append-db data/store/broadcast.db
```

The script refuses to reuse an existing run artifact path unless append mode is
explicitly requested, which prevents accidentally mixing two unrelated runs.

## Backups

Use SQLite's online backup command rather than copying a live database file:

```bash
mkdir -p data/backups
sqlite3 data/store/broadcast.db \
  ".backup 'data/backups/broadcast-$(date -u +%Y%m%dT%H%M%SZ).db'"
```

For a quick integrity check after backup:

```bash
sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"
```

## WAV Retention

`scripts/prune_old_wavs.sh` keeps JSON sidecars indefinitely and deletes only
old WAV files whose sidecar reports `"complete": true`.

Default retention is 7 days:

```bash
./scripts/prune_old_wavs.sh data/captures
```

Override retention:

```bash
RETENTION_DAYS=3 ./scripts/prune_old_wavs.sh data/captures
```

The pruner is idempotent and safe to run repeatedly. `continuous_capture_blocks.sh`
invokes it after each successfully classified block when the script is present
and executable.

## Common Reports

Recent summary:

```bash
.venv/bin/python -m radio_classifier report summary --since 24h
```

Top songs this week:

```bash
.venv/bin/python -m radio_classifier report songs --since 7d --top 25
```

Songs added in a precise UTC window:

```bash
.venv/bin/python -m radio_classifier report songs-added \
  --from 2026-06-01T00:00:00Z \
  --to 2026-06-02T00:00:00Z
```

Capture-run provenance:

```bash
.venv/bin/python -m radio_classifier report runs --since 7d
```
