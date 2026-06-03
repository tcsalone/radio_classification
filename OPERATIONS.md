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

For a bounded long capture where the target is **actual captured audio**, use
the resilient wrapper from the repository root:

```bash
./scripts/capture_until_audio_hours.sh 20
```

This is the preferred overnight workflow. It appends to
`data/store/broadcast.db`, opens a fresh `capture_runs` row for each capture
iteration, and sums the sidecar `duration_seconds` values. If `rtl_fm` or the
USB/IP dongle stream drops early, the partial chunk is classified, the next
iteration starts after a short backoff, and the wrapper keeps going until the
requested number of audio hours has been captured.

Useful overrides:

```bash
DB_PATH=data/store/broadcast.db \
BLOCK_SECONDS=1800 \
RESTART_BACKOFF_SECONDS=30 \
RETENTION_DAYS=7 \
./scripts/capture_until_audio_hours.sh 20
```

For an unbounded foreground supervisor, use:

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
`capture_runs` row for the next iteration. It is intentionally unbounded; use
`capture_until_audio_hours.sh` when you want a specific amount of audio.

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

Use `continuous_capture_blocks.sh` directly for bounded experiments where a
single `rtl_fm` stream is acceptable and no restart-on-drop behavior is needed:

```bash
./scripts/continuous_capture_blocks.sh 4
```

By default this writes a fresh per-run database under `data/eval/`. To append a
bounded run into the long-term store:

```bash
./scripts/continuous_capture_blocks.sh 24 --append-db data/store/broadcast.db
```

The script refuses to reuse an existing run artifact path unless append mode is
explicitly requested, which prevents accidentally mixing two unrelated runs. For
overnight collection, prefer `capture_until_audio_hours.sh` so a transient USB/IP
drop does not silently shorten the requested audio duration.

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

## LLM / Tier 3 (GPU Ollama)

Tier 3 classification and the optional commercial brand backfill (`--llm`) call
a local Ollama server. **Do not use the snap-packaged Ollama for this** — snap
confinement cannot reach the WSL2 CUDA driver, so it silently runs the model
100% on CPU (~40s per call, observed 2026-06-02). The native binary uses the
RTX 2080 Ti and drops that to ~1–1.5s per call.

Native GPU Ollama runs on port **11435** (the idle snap stays on 11434):

```bash
# One-time install (no sudo): native binary + bundled CUDA libs in ~/.local/ollama
# (download ollama-linux-amd64.tar.zst from github.com/ollama/ollama/releases,
#  decompress with `python -c "import zstandard,sys; ..."`, untar into ~/.local/ollama)

# Start the GPU server (background) and pull the model once:
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama serve > /tmp/ollama_native.log 2>&1 &
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama pull llama3.2:latest

# Verify GPU placement (PROCESSOR must read "100% GPU", not "100% CPU"):
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama ps
```

Point the classifier at the GPU instance via the env var:

```bash
export RADIO_CLASSIFIER_OLLAMA_HOST=http://127.0.0.1:11435
```

## Commercial Brand Backfill

COMMERCIAL events the funnel could not attribute to a brand land in the
dashboard's "Unbranded / unidentified" bucket. `commercials backfill-brands`
recovers brands from the stored transcript in two tiers:

1. **Deterministic** (default, offline, high precision): a known-phrase table +
   curated URL/domain map. This is the reliable win — it never mints junk brand
   names that would fragment existing advertisers.
2. **`--llm`** (optional): re-classifies the transcript text with the GPU Ollama
   above for events the deterministic pass missed. Lower yield on short
   fragments (text-only re-classification of ad tails/disclaimers is weak), so
   treat it as a bonus, not the primary recovery path.

```bash
# Preview (read-only) over the whole DB:
.venv/bin/python -m radio_classifier commercials backfill-brands \
  --db-path data/store/broadcast.db --dry-run

# Apply the deterministic pass (back up the DB first):
cp data/store/broadcast.db data/store/broadcast.db.bak_$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python -m radio_classifier commercials backfill-brands \
  --db-path data/store/broadcast.db --apply

# Optional LLM tier (needs GPU Ollama on 11435):
RADIO_CLASSIFIER_OLLAMA_HOST=http://127.0.0.1:11435 \
  .venv/bin/python -m radio_classifier commercials backfill-brands \
  --db-path data/store/broadcast.db --llm --apply
```

Re-running is idempotent (only events still missing a brand are processed).

### Boundary merge (overlapping-window orphans)

The biggest source of unbranded commercials is *window overlap*: one ad lands
across consecutive events, the LLM extracts the brand on one window but returns
`null` on the heavily-overlapping neighbor, leaving an orphan fragment.
`commercials merge-boundaries` attributes each unbranded COMMERCIAL fragment to
its adjacent branded-commercial neighbor when their transcripts are similar
enough to be the same ad.

The `--min-similarity` gate (default `0.55`) is the safety mechanism: true
same-ad overlaps cluster at ≥0.55, while straddle/pod-boundary cases (the
fragment is a *different* ad than the neighbor's label) fall in the
low-similarity tail and are rejected. The pass picks the highest-similarity
branded neighbor, so a straddle fragment attaches to the ad it actually
overlaps most.

```bash
# Preview (read-only):
.venv/bin/python -m radio_classifier commercials merge-boundaries \
  --db-path data/store/broadcast.db --dry-run

# Apply (back up first):
cp data/store/broadcast.db data/store/broadcast.db.bak_$(date -u +%Y%m%dT%H%M%SZ)
.venv/bin/python -m radio_classifier commercials merge-boundaries \
  --db-path data/store/broadcast.db --apply

# Tune precision/recall (higher = safer, lower recall):
.venv/bin/python -m radio_classifier commercials merge-boundaries \
  --db-path data/store/broadcast.db --min-similarity 0.60 --dry-run
```

On the 2026-06-02 run this recovered 108 of 204 unbranded fragments (204 → 96).
Going forward the `SegmentReducer` applies the same relaxed-identity absorb at
capture time (`unbranded_absorb_similarity_threshold`, default `0.55`), so new
captures should produce far fewer orphans. Re-running is idempotent.
