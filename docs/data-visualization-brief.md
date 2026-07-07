# Data & Reporting Brief — Radio Classifier

**Purpose of this document.** This is a handoff/reference for building new data
visualizations on top of the radio-classifier dataset. It documents the
**current state only**: the reports that exist today, what the operator cares
about, and how to get a copy of the database to experiment with. It deliberately
does **not** propose a future visualization design — that work is being done
separately with other tooling.

Last updated: 2026-06-11. All figures below are a point-in-time snapshot of the
live database on that date.

---

## 1. What the system does (one paragraph)

The pipeline captures an over-the-air FM radio stream (KITS "Live 105", 105.3 MHz,
San Francisco) via an RTL-SDR dongle in WSL2 and classifies every segment of
airtime into one of five categories — `SONG`, `DJ`, `COMMERCIAL`, `STATION`,
`PSA_NEWS` — using a 3-tier funnel (Tier 1 audfprint fingerprinting → Tier 2
YAMNet + Shazam fallback → Tier 3 Whisper transcription + local LLM). Results are
appended to a single long-lived SQLite database so stats accumulate across runs.

---

## 2. The database

### Location and how to get a testing copy

| Item | Path |
| --- | --- |
| Live database | `data/store/broadcast.db` |
| Backups (auto, per cleanup run) | `data/backups/broadcast-<UTCstamp>.db` |
| Ad-hoc safety snapshots | `data/store/broadcast.pre-*-<UTCstamp>.db` |
| Schema definition | `db/schema.sql` |
| Git repo | `https://github.com/tcsalone/radio_classification.git` (branch `main`) |

**Make a consistent copy for testing** (do not open the live file directly while a
capture is running — use SQLite's backup API or copy a backup snapshot):

```bash
# Option A: consistent online backup (safe while pipeline is running)
python -c "import sqlite3; s=sqlite3.connect('data/store/broadcast.db'); b=sqlite3.connect('/tmp/broadcast-test.db'); s.backup(b); s.close(); b.close()"

# Option B: copy the newest cleanup backup (already a static file)
cp "$(ls -t data/backups/broadcast-*.db | head -1)" /tmp/broadcast-test.db
```

The database is plain SQLite (currently ~16 MB) with no extensions required.
`PRAGMA foreign_keys = ON` is used by the app; WAL journaling is enabled.

### Snapshot of contents (2026-06-11)

- **10,465** `broadcast_events` spanning **2026-05-31 → 2026-06-10** (~158 h of
  classified airtime, accumulated across multiple capture runs).
- Category mix (all-time): SONG 4,367 · COMMERCIAL 2,968 · DJ 1,788 ·
  STATION 964 · PSA_NEWS 378.
- **416** songs (416 distinct songs have been played; 384 have MusicBrainz
  release dates), **789** commercials, **1,360** brands.
- **22** `capture_runs` rows, **14** of which actually carry events (the rest are
  empty 3-second startup/probe rows — see §6).

### Schema (schema v4)

`broadcast_events` is the central fact table; everything else is a dimension it
references.

**`broadcast_events`** — one row per classified segment of airtime.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | INTEGER PK | |
| `timestamp_start` | TEXT | ISO-8601 UTC, e.g. `2026-06-10T17:58:22.069Z` |
| `timestamp_end` | TEXT | nullable |
| `duration` | REAL | seconds of airtime for the segment |
| `category` | TEXT | one of `SONG`,`DJ`,`COMMERCIAL`,`STATION`,`PSA_NEWS` |
| `song_id` | INTEGER FK→songs | NULL for unidentified music (see §6) |
| `commercial_id` | INTEGER FK→commercials | nullable |
| `brand_id` | INTEGER FK→brands | nullable |
| `artist` | TEXT | denormalized display copy for SONG rows |
| `track_title` | TEXT | denormalized display copy for SONG rows |
| `brand_name` | TEXT | denormalized display copy for COMMERCIAL rows |
| `transcript_excerpt` | TEXT | Tier-3 speech excerpt (DJ/commercial/etc.) |
| `confidence` | REAL | classifier confidence, nullable |
| `capture_run_id` | INTEGER FK→capture_runs | provenance of the segment |
| `created_at` | TEXT | insert time |

**`songs`** — canonical song identities. Columns: `id`, `audfprint_track_id`
(reference-audio path when fingerprinted; NULL for Shazam-only), `artist`,
`title`, `source` (`audfprint`|`shazam`|`manual`), `first_seen_utc`,
`release_date` (MusicBrainz, nullable). Unique on `(artist, title, source)`.

**`commercials`** — text-derived ad identities. Columns: `id`, `brand_id`
(FK→brands), `duration_bucket_seconds`, `minhash_hex`, `reference_transcript`,
`first_heard_utc`, `play_count`. Unique on `(brand_id, duration_bucket_seconds, minhash_hex)`.

**`brands`** — canonical advertiser names. Columns: `id`, `canonical_name`
(unique), `aliases_json`, `created_at`.

**`capture_runs`** — provenance. Columns: `id`, `run_id` (unique text), `started_utc`,
`ended_utc` (nullable), `pipeline_version`, `host`, `notes`, `created_at`.

**`brand_mentions`** — DJ shout-outs / tags. Columns: `id`, `segment_id`
(FK→broadcast_events, ON DELETE CASCADE), `brand_id`, `mention_type`
(`paid_ad`|`dj_shoutout`|`tag`), `heard_utc`.

Indexes exist on `broadcast_events` for `timestamp_start`, `category`, `brand_id`,
`song_id`, `commercial_id`, `capture_run_id`.

---

## 3. Reports produced so far

All reports are generated by the `radio_classifier` CLI. Every windowed report
accepts a window via either `--since <duration|ISO>` (relative to "now",
default `24h`) **or** an explicit `--from <ISO>` / `--to <ISO>` pair.

### 3a. HTML Dashboard — `report dashboard`

```bash
python -m radio_classifier report dashboard --db-path data/store/broadcast.db \
  --from 2026-06-07T19:00:00Z --to 2026-06-10T18:30:00Z --out data/reports/dashboard.html
```

Output: `data/reports/dashboard.html` (self-contained static HTML, dark theme).
Sections rendered:

- **Headline stats**: Classified Airtime, Segments (raw `broadcast_events` count),
  Music Share (SONG airtime ÷ classified airtime), Commercial Share.
- **Song Age**: mean / median / airtime-weighted age, and Catalog Coverage
  (songs with vs. without known release dates), computed from `songs.release_date`.
- **Category Airtime**: total airtime per category.
- **Top Artists**: spins, "promo" spins (short station-promo plays), distinct
  titles, segment count, total airtime.
- **Top Songs**: spins, promo spins, segments, airtime.

### 3b. HTML Artist Play Log — `report artist-plays`

```bash
python -m radio_classifier report artist-plays --db-path data/store/broadcast.db \
  --from 2026-06-07T19:00:00Z --to 2026-06-10T18:30:00Z --top 3 --out data/reports/artist-plays.html
```

Output: `data/reports/artist-plays.html`. For each of the top N artists (default 3),
a per-artist section listing **every play**: timestamp (UTC), title, and length.

### 3c. Shazam Discoveries — `songs discovered`

```bash
python -m radio_classifier songs discovered --db-path data/store/broadcast.db \
  --since 48h --top 20 --min-plays 1
```

Text table of songs found by the Shazam fallback (`source='shazam'`): artist,
title, play count, last-heard, whether the song is already in the local
fingerprint tracklist, and a manual-review flag for low-confidence rows
(< 3 plays). Used to decide which songs to "promote" into the Tier-1 index.

### 3d. Other CLI report subcommands (text output)

`report` also exposes: `commercials`, `brands`, `songs`, `songs-added`
(songs first added to the catalog in a window), `songs-timeline` (chronological
SONG-only log), `artists` (per-artist airtime rollup), `timeline`, `summary`,
and `runs` (capture-run provenance). These print to stdout and are the simplest
machine-readable starting points for a custom visualization.

### 3e. Cursor Canvas — analytical artifact

`canvases/radio-capture-campaign-analysis.canvas.tsx` is a live React "canvas"
(rendered inside Cursor) summarizing the most recent multi-run campaign:
headline stats, a category-mix donut, a capture-runs table, the duplicate-song
cleanup, manual-review items, and health signals. It is hand-authored per
analysis rather than generated by the CLI.

---

## 4. What is important to the operator

Derived from the working history of the project. These are the questions and
qualities the operator repeatedly cares about:

1. **Accurate song identity & no duplicate identities.** The same recording must
   not appear as multiple `songs` rows (remix/feature/collab spelling drift was a
   recurring problem). Top-songs and play counts must reflect one identity per
   song.
2. **Clean commercial data.** Minimize "Unknown commercial" rows; deduplicate
   repeated airings of the same ad; attribute each ad to a canonical brand. Ad
   duplication and unbranded ads are explicitly disliked.
3. **Top artists / top songs / full play logs.** Who and what is played most, and
   the exact play-by-play timeline for top artists.
4. **Category mix and airtime split** (music vs. commercial vs. talk), including
   commercial load.
5. **Song age / catalog freshness** via MusicBrainz release dates (how old is the
   music being played, on average and airtime-weighted).
6. **Tier-1 coverage growth.** Promoting frequently-heard Shazam discoveries into
   the audfprint fingerprint index so they get deterministic Tier-1 IDs later.
7. **Manual-review surfacing.** Low-confidence discoveries and genre outliers the
   operator should sanity-check.
8. **Continuity.** All runs append to the **same** database so there is one
   growing pool of data to analyze over time.
9. **Pipeline observability / health.** Confidence that a run actually classified
   speech tiers (a misconfigured LLM host once silently dropped all
   commercial/DJ/station data to "unknown").

---

## 5. Useful read-only starting queries

```sql
-- Category mix for a window
SELECT category, COUNT(*) n, ROUND(SUM(duration)/3600.0,2) hours
FROM broadcast_events
WHERE timestamp_start >= '2026-06-07T19:00:00Z'
GROUP BY category ORDER BY n DESC;

-- Top songs by plays (identified only)
SELECT s.artist, s.title, COUNT(*) plays
FROM broadcast_events e JOIN songs s ON s.id = e.song_id
WHERE e.category='SONG' AND e.song_id IS NOT NULL
GROUP BY e.song_id ORDER BY plays DESC LIMIT 20;

-- Top advertiser brands
SELECT COALESCE(NULLIF(brand_name,''),'(unbranded)') brand, COUNT(*) airings
FROM broadcast_events WHERE category='COMMERCIAL'
GROUP BY brand ORDER BY airings DESC LIMIT 20;

-- Hourly airtime by category (good for a heatmap/timeline)
SELECT substr(timestamp_start,1,13) hour_utc, category, COUNT(*) n
FROM broadcast_events GROUP BY hour_utc, category ORDER BY hour_utc;

-- Per-run provenance
SELECT cr.id, cr.run_id, cr.started_utc, cr.ended_utc, COUNT(e.id) events
FROM capture_runs cr LEFT JOIN broadcast_events e ON e.capture_run_id = cr.id
GROUP BY cr.id ORDER BY cr.id DESC;
```

---

## 6. Data-quality notes / gotchas (important for any viz)

- **Unidentified music is real and common.** ~20% of SONG events have
  `song_id IS NULL` (Tier 2 detected music but no Tier 1/Shazam identity). In the
  dashboard these collapse into a single "? — ?" top-song row. Decide explicitly
  whether to show, bucket, or exclude them.
- **`--since` is relative to "now".** Reports default to `--since 24h`; if you
  regenerate hours/days after a run, that window can be **empty** and the report
  looks blank. Use explicit `--from`/`--to` for historical windows.
- **"Promos" vs "spins".** The dashboard separates short station-promo plays of a
  song (e.g. a 20-second sting) from full spins. Don't double-count.
- **Empty `capture_runs` rows.** Several runs have 0 events and ~3-second
  durations (pipeline startup/probe artifacts). Filter to runs with events when
  charting per-run stats.
- **Denormalized display fields.** `broadcast_events.artist` / `track_title` /
  `brand_name` are display copies; the canonical identity lives in
  `songs` / `brands`. Join to the dimension tables for deduped identity; the
  denormalized text may carry older spellings.
- **Timestamps are UTC ISO-8601 strings** with millisecond precision and a `Z`
  suffix. The station is US/Pacific — convert for local-time-of-day analysis.
- **Airtime = `SUM(duration)`**, which is segment-duration based (not wall-clock
  gap based); brief un-classified gaps between segments are not represented as
  rows.
