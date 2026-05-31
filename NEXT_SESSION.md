# Next Session — Discussion Topics

Captured 2026-05-30. Topics to work through next time we sit down for planning.

## Priority items (carried over from previous session)

### 1. Parallel capture + classify
- Today the script captures a 30 min block, *then* classifies. Wall-clock = capture + classify.
- Goal: capture block N+1 while classifying block N → wall-clock = max(capture, classify).
- Open questions:
  - GPU contention: rtl_fm doesn't touch the GPU but ingest does some light DSP. Should be fine.
  - SDR USB contention: only one rtl_fm process can hold the device at a time, but capture is sequential anyway, so no conflict.
  - Disk I/O: classify reads ~150 MB WAV while next capture writes another. Trivial on local SSD.
  - DB locking: SQLite with WAL mode should be fine for two writers in different time ranges.
- Design choices:
  - Two-process pipeline with a simple file-based handoff queue, OR
  - Convert `morning_run_blocks.sh` into a Python orchestrator with `asyncio` and run capture/classify as concurrent tasks.

### 2. UI for metrics
- Today: CLI-only reports.
- Possible directions:
  - Static HTML dashboard generated at end of each run (no server, just open the file).
  - Lightweight Flask/FastAPI app reading from the SQLite DB live.
  - Streamlit if we want it to be cheap to build and OK with the Streamlit look.
  - Grafana with the SQLite plugin (heavier but professional).
- Want: spins-over-time charts, brand frequency heatmaps, daypart breakdowns, top artists/songs leaderboards.

### 3. Posting metrics to Twitter/Bluesky
- Daily/weekly summaries: "Most-played artist this week was X with N spins."
- Brand exposure quirks: "Station mentioned itself N times today."
- Outliers: "Today's longest commercial block was N minutes."
- API needs: posting credentials, rate limits, image generation for charts.

## New additions (2026-05-30)

### 4. Rewrite in C / Rust for performance?

**The question:** Would porting to a compiled language make this meaningfully faster?

**Where Python actually lives in the pipeline (and what's CPU-bound):**
- **rtl_fm (capture):** already native C. Not touching this.
- **audfprint (Tier 1):** Python wrapper around numpy/scipy FFTs. The hot loop is numpy/C, not Python.
- **YAMNet (Tier 2):** TensorFlow C++ backend. Python is just glue.
- **faster-whisper (Tier 3a):** CTranslate2 (C++) running on CUDA. Python is just glue.
- **Ollama (Tier 3b):** Ollama is Go, model runs in llama.cpp (C++). Python is just an HTTP client.
- **Segment reducer + DB writes:** pure Python, but trivial CPU compared to the above.
- **CLI/reports:** SQL + light Python aggregation.

**Honest assessment:** I suspect the answer is **mostly no, with one possible exception.**
- Tier 1/2/3 are already native code under the hood. Rewriting the glue won't move the needle.
- The one place Python *might* be the bottleneck is the segment reducer + ingestion pipeline orchestration if we go parallel — but profiling would tell us, and rewriting orchestration in Rust is a lot of complexity for marginal gain.
- A more productive angle: **profile first**, identify any genuine Python hotspots, port only those to Cython/Rust extensions if warranted.

**To discuss next session:**
- Run a profiling pass (`py-spy record` on a real classify) to find actual hotspots before considering a rewrite.
- Decide if the hotspots (if any) warrant a partial rewrite vs. a full one.
- Cost: a full rewrite is probably 4-8 weeks of focused effort with significant regression risk. A targeted Cython/Rust extension for a hot loop is more like 1-3 days.

### 5. Offload Tier 2 / Tier 3 to cloud LLM (Gemini, etc.)?

**The question:** Could we push acoustic classification or speech classification to a hosted LLM API and what would it cost?

**Current Tier 3 load (rough numbers from today's run so far):**
- 633 segments classified in ~14 hours of audio.
- Tier 3 only fires on non-SONG segments after Tier 1 + Tier 2 filtering. From the in-progress run: 194 COMMERCIAL + 95 DJ + 55 STATION + 17 PSA_NEWS = **361 Tier 3 LLM calls** across 14 hours.
- Each call sends a Whisper transcript (typically 50-300 tokens) and gets back a category (~10 tokens).

**Rough back-of-envelope cost projection (assume 200 input + 10 output tokens per call):**

| Provider / model | Input $/M | Output $/M | 361 calls cost | Per 24h |
| --- | --- | --- | --- | --- |
| Gemini 2.0 Flash (current pricing as of 2026-05) | $0.10 | $0.40 | <$0.01 | ~$0.01 |
| Claude Haiku 4 | $0.25 | $1.25 | ~$0.02 | ~$0.04 |
| GPT-5-mini | $0.30 | $1.20 | ~$0.02 | ~$0.05 |
| Local Ollama (today) | $0.00 | $0.00 | $0.00 | $0.00 (but uses GPU) |

*(Numbers above are rough — actual prices need to be verified at planning time since LLM pricing shifts often.)*

So **Tier 3 to Gemini Flash is essentially free** (sub-penny per day). The question isn't cost — it's whether we want the dependency on a cloud service for what's currently a fully local pipeline.

**Tier 2 (YAMNet) to a cloud LLM is a different story:**
- YAMNet runs on every 0.96s window of audio = ~3,750 windows per hour, ~90,000/day.
- LLMs can't classify audio directly without expensive audio-input models (Gemini 2.0 Flash has audio input but it's metered separately and much more expensive).
- A cloud audio classifier would be roughly $0.10-$0.50 per minute of audio processed. For 24h of continuous capture, that's $144-$720/day. **Way too expensive** for what YAMNet already does locally for free.
- **Verdict:** keep Tier 2 local. Move Tier 3 to cloud only if it reduces operational burden (no model download, no GPU memory pressure) — but the per-cost answer says it's negligible either way.

**To discuss next session:**
- Do we *want* to move off local? Pros: GPU freed for other work, no Ollama process management. Cons: external dependency, network outages = pipeline stalls, privacy (transcripts go to Google/Anthropic/OpenAI).
- If we keep Tier 3 local, can we get better accuracy from a larger model running on the same GPU (llama 3.3 70B fits at 4-bit)?
- Hybrid: local Ollama default, cloud fallback if local is unavailable?

### 6. Commercials report dedup / cleanup

**The question:** `report commercials` shows a lot of near-duplicate rows. Cleaning this up will dramatically improve the report's usefulness.

**Three distinct root causes, stacked together** (concrete examples from the 12h run DB):

**A. Single ad split across two adjacent 10s windows → two `commercials` rows.** This is the dominant cause.
- `id=130/131 Toyota` (00:26:43 + 00:26:53): transcript 130 ends "...hurry in whi-" and 131 begins "in while there's still time to save..." — clearly the same ad spliced at the window boundary.
- `id=167/168 Chase` (00:23:13 + 00:23:23): same pattern.
- `id=61/62 Smart & Final / Smart and Final`: transcripts overlap at "$25 minimum purchase at Smart and Final" — same ad, split.
- `id=89/90 Graton`: more subtle — 89's transcript is actually the *tail* of a Bay Alarm ad that happened to mention Live 105, then 90 is the Graton ad. The brand attached to the wrong window.
- **Fix vector:** add a commercial-merge pass to the reducer (analogous to the song bridge reducer): if two consecutive `COMMERCIAL` segments are ≤ 10s apart AND minhashes overlap OR brand canonicalizes the same, merge into one row.

**B. Brand-name canonicalization gaps.** Even when commercials are correctly separated, brand variants split rollups:
- `&` vs "and": `Smart & Final` vs `Smart and Final` (id 61 vs 62, 138).
- Parent-vs-subsidiary brand naming: `Xfinity` vs `Xfinity Mobile` (id 74/76/77 vs 75/94). Decide: collapse subsidiaries under parent, or keep separate and group at report time.
- Parent-qualifier variants: `Golden State Lumber` vs `Golden State Lumber and Showroom` (id 11/12/13/33).
- Whisper-mishearing variants: `Habes Law` / `Habus Law` / `Habas Law` / `Habes Law` (id 1/22/40/23); `Atco` / `Atko` / `ATCO` (id 30/31/18/32); `Ambutra` / `Mbutra` (id 45/46/47); `Big Lou` / `Big Lou Insurance` / `Big Lou's Life Insurance` (id 52/53/54/55); `PF Chang's` / `P.F. Chang's` (id 26/35/39); `WOC Fire` / `Wokfire` (id 25/34).
- **Fix vector:** extend `canonicalize_brand` to handle: `&` → `and`, strip trailing parent-qualifier phrases (` and Showroom`, ` Insurance`, ` Mobile` if parent is also seen), case-fold all comparisons, add a Whisper-mishearing alias table for the common Live 105 advertisers.

**C. `commercials` row aggregation isn't matching minhashes well.** Even with same brand + same airing time, two rows get created instead of incrementing `play_count` on one row.
- Need to revisit the minhash-similarity threshold in the upsert path. Currently it appears to be too strict (or comparing wrong fields).
- May benefit from a secondary signature: brand_id + duration_bucket + a fuzzy transcript-token overlap, used as a fallback when minhash doesn't quite match.

**Two-part deliverable when we work this:**
1. **Pre-write fix (prevents future duplicates):** ad-merge pass in reducer + tighter `canonicalize_brand` + looser minhash dedup threshold in upsert.
2. **Post-hoc admin command (`commercials dedupe`):** fold existing duplicates in any DB, similar to the `songs dedupe` we built. Re-points `brand_mentions` and `broadcast_events.commercial_id` references to survivors, deletes losers.

**Estimated effort:** ~1 focused session for the prevention path, ~half a session for the dedup admin command. Tests for both should follow the patterns we established in `tests/test_songs_discovery.py` and `tests/test_persistence_store.py`.

### 7. New report: chronological song log (`report songs-timeline`)

**The question:** "Show me every song detected in chronological order." Today this requires a hand-written SQLite query — would be useful as a built-in subcommand.

**Proposed shape:**

```bash
radio-classifier report songs-timeline --db-path "$DB" [--since 1d] [--limit 1000]
```

Output columns: `start_utc`, `duration`, `artist`, `title`, `confidence`, `detection_source` (audfprint/shazam/unknown).

**Implementation sketch (low effort):**
- New `songs_timeline()` query in `reports/queries.py` reusing the existing `TimelineRow`-style pattern, filtered to `category = 'SONG'`, with optional `--since` and `--limit`.
- New `format_songs_timeline()` in `reports/format.py`.
- Wire into `cli.py` `report` subparser as `songs-timeline`.
- Tests in `test_reports.py` mirroring the `songs_top` tests.

**Estimated effort:** ~30-45 min including tests. Smallest item on the list and a great warm-up before tackling the commercials dedup work.

**Optional extras to consider while we're in there:**
- `--collapse` flag to merge adjacent same-song segments (similar to spin logic).
- `--unknown-only` flag to list just the `?` rows for investigation.
- `--csv` flag to dump as CSV for spreadsheet analysis.

## Other items spotted during today's 12h run (lower priority)

- ~~**Julia Wolf "In My Room (Acoustic)" promo-vs-song detection:** 10 "spins" in 7m10s = ~43 sec per spin. That's a station promo, not a real play. Spin definition needs a heuristic to disambiguate (e.g., "if average spin duration < 50% of full track length, suppress as promo").~~ **DONE.** Spin tally now reports `full_spin_count` and `promo_spin_count` side by side (threshold `PROMO_MAX_SPIN_SECONDS = 90s`). `report songs` / `report artists` show `spins   promos` columns, the dashboard renders a `+N promo` pill, and the songs/artists sort order tiebreaks by non-promo airtime so teaser-heavy entries drop down the leaderboard. Julia Wolf went from a top-2 entry to #4 in the morning-DB songs report and out of the top-8 artists report.
- ~~**LINKIN PARK casing artifact:** artist display name shows ALL CAPS because most Shazam responses returned that way. Either normalize artist casing on write, or pick a different display-winner heuristic.~~ **DONE.** `BroadcastStore.upsert_song` now normalizes the known `LINKIN PARK` display artifact to `Linkin Park`, prefers mixed-case reference data over all-caps Shazam display text for the same song identity, and preserves legitimate acronyms like `AFI`. Existing rows in `data/eval/morning_20260530_24x30m.db` were normalized too.
- **Modest Mouse "Picking Dragons' Pockets" Shazam-vs-tracklist dedup miss:** showed up as a "new" Shazam discovery despite being on the tracklist. Likely unicode apostrophe difference (U+2019 vs U+0027) or audfprint missing it at min_count=60. Worth a 30-min investigation.
- ~~**audfprint min_count=60 floor tuning:** might be too strict for some reference tracks. Could revisit per-track score floors or fall back to a lower floor if Shazam confirms.~~ **DONE.** The audfprint candidate floor is now `45`, but scores below the strong `60` threshold require 3 adjacent same-track matches in the funnel before Tier 1 wins. This should recover weaker real matches without reopening the old one-off false-positive pile.
