# Sample data — for visualization & analysis work

This directory holds a **cleaned, point-in-time SQLite snapshot** of the
classified broadcast corpus so contributors can build visualizations without
needing the capture hardware (RTL-SDR), the model stack, or the multi-gigabyte
raw audio.

The `.db` file is stored with **Git LFS** (see `.gitattributes`). After cloning:

```bash
git lfs install        # once per machine
git lfs pull           # fetch the actual .db (clone only downloads a pointer)
```

## What's here

| File | Size | Contents |
|---|---|---|
| `broadcast_analysis.db` | ~36 MB | Cleaned SQLite DB: **52,441** classified events spanning **2026-07-15 → 2026-08-07** (~3.5 weeks, 105.3 MHz FM). |

The snapshot has the full post-run cleanup applied: boundary-commercial merge,
deterministic brand backfill, commercial + song dedupe, block-boundary song
stitch, and the brand-identity merge (`commercials merge-brands`) — so brand
and song rankings are already de-fragmented and safe to chart directly.

## Schema

Full schema: [`../db/schema.sql`](../db/schema.sql). The tables a visualization
will care about most:

- **`broadcast_events`** — one row per classified segment. `category` is one of
  `SONG | COMMERCIAL | DJ | STATION | PSA_NEWS`. Carries `timestamp_start`,
  `timestamp_end`, `duration`, and denormalized `artist` / `track_title` /
  `brand_name` plus FKs `song_id` / `commercial_id` / `brand_id`.
- **`songs`** — canonical song identities (`artist`, `title`, `release_date`).
  Aggregate plays by `song_id`, not by the free-text `artist`/`track_title`.
- **`brands`** — canonical advertisers (`canonical_name`, `aliases_json`).
- **`commercials`** — unique ad identities with `play_count`.
- **`capture_runs`** — capture sessions; `broadcast_events.capture_run_id`
  joins here (run 1 = 24h baseline, later ids = the weekly runs).

## Quick start

The repo already ships report generators — point them at this DB:

```bash
python -m radio_classifier report dashboard   --db-path sample-data/broadcast_analysis.db --since 14d --out /tmp/dashboard.html
python -m radio_classifier report artist-plays --db-path sample-data/broadcast_analysis.db --since 14d --top 5 --out /tmp/artists.html
```

Report query helpers live in [`../src/radio_classifier/reports/`](../src/radio_classifier/reports/)
(`queries.py` has `songs_top`, `brands_top`, timeline helpers, etc.) — a good
starting point for a richer front-end.

A couple of sanity queries:

```sql
-- airtime share by category
SELECT category, COUNT(*) events, ROUND(SUM(duration)/3600.0, 1) hours
FROM broadcast_events GROUP BY category ORDER BY hours DESC;

-- top songs by play (aggregate on song_id, never raw text)
SELECT s.artist, s.title, COUNT(*) plays
FROM broadcast_events e JOIN songs s ON s.id = e.song_id
WHERE e.category = 'SONG' GROUP BY e.song_id ORDER BY plays DESC LIMIT 20;
```

## What is deliberately NOT in git

- **Raw audio** — `data/reference/` (licensed music) and `data/captures/`
  (recorded FM broadcast) are copyrighted and multi-GB; they are git-ignored and
  backed up outside the repo.
- **The live master DB** — it is appended to continuously during capture; this
  frozen snapshot is the stable artifact to develop against.

Regenerate a fresh snapshot any time with a `sqlite3` backup of the master
followed by `scripts/post_run_cleanup.sh` and `commercials merge-brands`.
