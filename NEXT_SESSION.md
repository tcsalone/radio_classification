# Next Session — Discussion Topics

## Current Handoff (2026-06-01)

Always-on capture store work is through Phase 4 locally:

- Phase 1: schema v3, `capture_runs`, automatic v2 to v3 migration, and
  pipeline version provenance are implemented.
- Phase 2: `capture_forever.sh`, `continuous_capture_blocks.sh --append-db`,
  `runs` CLI commands, and sidecar-aware WAV retention are implemented.
- Phase 3: report windows now support `--from` / `--to`, with new
  `report songs-added` and `report runs` commands.
- Phase 4: operator guidance now lives in `OPERATIONS.md`.

Primary operating path:

```bash
./scripts/capture_forever.sh
```

Default persistent database:

```bash
data/store/broadcast.db
```

Useful first checks next session:

```bash
.venv/bin/python -m radio_classifier report runs --since 7d
.venv/bin/python -m radio_classifier report summary --since 24h
sqlite3 data/store/broadcast.db "SELECT value FROM schema_meta WHERE key = 'version';"
```

Backup before long or risky maintenance:

```bash
mkdir -p data/backups
sqlite3 data/store/broadcast.db \
  ".backup 'data/backups/broadcast-$(date -u +%Y%m%dT%H%M%SZ).db'"
```

To stop long-running capture cleanly, press `Ctrl-C` in the terminal running
`capture_forever.sh`. See `OPERATIONS.md` for retention and manual run-close
commands.

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
- ~~Need to revisit the minhash-similarity threshold in the upsert path. Currently it appears to be too strict (or comparing wrong fields).~~
- ~~May benefit from a secondary signature: brand_id + duration_bucket + a fuzzy transcript-token overlap, used as a fallback when minhash doesn't quite match.~~

**Status update 2026-06-01:** ~~`CommercialIdentityResolver`~~ now has a tertiary cosine-only fallback (`cosine_tertiary = 0.85`) that fires after the primary (Jaccard ≥ 0.70) and secondary (Jaccard ≥ 0.55 + cosine ≥ 0.85) paths. The new path catches the common failure mode where two ASR passes of the same ad share most tokens but reorder phrases (e.g., `Camry and Corolla` ↔ `Corolla and Camry`) — that reordering collapses 3-shingle Jaccard to near zero while leaving raw-token cosine high. Because the matcher is still gated by exact `(brand_id, duration_bucket_seconds)`, the false-merge risk is bounded; the regression suite adds a "different ad copy, same brand+duration must stay separate" case to lock this in. Three new tests in `tests/test_commercials_identity.py`: tertiary match on reordered phrases, no-merge on genuinely different ads of the same brand, and a kill-switch test (`cosine_tertiary=0.0` reverts to the legacy two-path behaviour).

**Two-part deliverable when we work this:**
1. **Pre-write fix (prevents future duplicates):** ad-merge pass in reducer + tighter `canonicalize_brand` + looser minhash dedup threshold in upsert.
2. **Post-hoc admin command (`commercials dedupe`):** fold existing duplicates in any DB, similar to the `songs dedupe` we built. Re-points `brand_mentions` and `broadcast_events.commercial_id` references to survivors, deletes losers.

**Estimated effort:** ~1 focused session for the prevention path, ~half a session for the dedup admin command. Tests for both should follow the patterns we established in `tests/test_songs_discovery.py` and `tests/test_persistence_store.py`.

**Status update 2026-06-01:**

- **Item B (brand canonicalization):** `canonicalize_brand` now applies a
  generic ``" & "`` → ``" and "`` rule before alias lookup so unmapped
  advertisers fold automatically; inline ``&`` in brands like ``AT&T``
  stays intact. Tests added in `tests/test_brands.py`.
- **Item A (adjacent-split folding):** `dedupe_commercials` now uses a
  10-second adjacent-gap default (was 2s), so a single ad split across two
  classifier windows with a brief silent gap folds correctly. Tests added
  in `tests/test_commercials_dedupe.py`.
- **Item 2 (admin command):** `radio-classifier commercials dedupe`
  already exists (`src/radio_classifier/commercials/dedupe.py`).
- ~~Still open: reducer-level pre-write merge pass (the dedupe runs post-hoc
  today) and Item C (minhash threshold loosening).~~ **Both shipped 2026-06-01.**
  Reducer-level pre-write merge is in `SegmentReducer.feed()` (merges adjacent
  commercials by brand + transcript similarity + combined-duration cap before
  emitting transitions). Item C minhash loosening is the tertiary cosine
  fallback documented above.

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

## Findings from the 2026-05-31 16h continuous run (manual validation)

### 8. Audfprint miss-rate problem on tracks already in the catalog

**The question:** Manual listening of "unknown SONG" segments from the 16h run on
2026-05-31 found that **almost every "unknown" was a song we already had
reference audio for**. Audfprint just didn't match.

**Concrete misses validated by ear:**

| Segment | UTC | Duration | Track | Catalog status |
| --- | --- | --- | --- | --- |
| block 21 @ 14:40 | 17:55:44Z | 100s | Oasis - Wonderwall | in tracklist + ref MP3 on disk |
| block 5  @ 13:40 | 09:54:54Z |  50s | Bad Omens - Dying To Love | in tracklist + ref on disk (songs row still `source=shazam`, never matched) |
| block 1  @ 15:20 | 07:56:34Z |  40s | The White Stripes - Fell In Love With a Girl | catalog + ref, matched fine on previous runs |
| block 2  @ 1:20  | 08:12:34Z |  40s | In Color - Headlights | catalog + ref (just promoted this session) |
| block 6  @ 6:40  | 10:17:54Z |  40s | Temper City - Self Aware (end fragment) | catalog + ref |
| block 7  @ 12:20 | 10:53:34Z |  40s | The Cranberries - Zombie (very end) | catalog + ref |

**Why this matters:** the 100s Wonderwall miss is the standout — that's three
windows long, well above any "fragment too short to fingerprint" excuse.

**Likely root causes (need to differentiate):**

1. **Tail/intro fragment hashing is sparse.** Self Aware end + Zombie end are
   classic short-tail cases. Less actionable.
2. **High harmonic density tracks** (Wonderwall acoustic, In Color) have
   uneven hash distributions across the song and audfprint may pick a region
   that doesn't share many hashes with the broadcast capture.
3. **Reference download quality.** The "best" stream we now grab without
   transcoding could be opus/webm at lower bitrate than the previous mp3
   192k. Worth confirming for the freshly-promoted In Color / Bad Omens
   files specifically.
4. **min_count = 45 floor + low-confidence-needs-3-adjacent rule** might be
   suppressing real hits below the 60 threshold for these tracks.

**Things to try next session:**

- Run `fingerprint eval` against the validated truth list above to measure
  current recall, then sweep `min_count` in `[30, 35, 40, 45]` and the
  `low_confidence_fingerprint_required_repeats` in `[2, 3]` and graph
  recall/precision.
- For each of the six tracks, run `audfprint match` directly against the
  broadcast clip to see the actual score the index reports. If scores are
  uniformly very low (single digits) the issue is reference-side
  fingerprint quality. If scores are 40-44 (just below the floor), the
  threshold is too strict.
- Inspect the codecs of the freshly-downloaded reference files
  (`In Color`, `Bad Omens`, the six manual-review promotions, M.I.A.
  Paper Planes). If any are sub-128k opus, force a transcode for those.

### 9. Tier 2 misclassifications: commercials/PSAs with music beds bypass YAMNet

**Status:** DONE 2026-06-01 with the conservative rescue-path approach. The
funnel now keeps the global MUSIC gate unchanged, but when audfprint misses,
Shazam is enabled and returns `no_match`, and YAMNet's MUSIC score is within
0.15 of SPEECH, it runs Tier 3 as a `tier3_rescue` attempt. Non-SONG Tier 3
classifications are persisted as speech-derived events; Tier 3 `SONG`,
errors, low-confidence Shazam, Shazam disabled, and pure-music windows remain
`tier2_unknown_song`.

**The question:** Three of the "unknown SONG" segments from the 16h run were
actually non-music content that YAMNet labeled as MUSIC, so they flowed into
the song path and audfprint/Shazam both correctly returned no match — leaving
them stuck as `category=SONG, song_id=NULL`.

**Concrete examples (16h run, 2026-05-31):**

- block 12 @ 18:20 — "Conversations" PSA (40s)
- block 13 @ 18:30 — Twisted Tea hard ice tea commercial (40s)
- block 14 @ 19:40 — Gatorade Lower Sugar commercial (40s)

**Why this matters:** these are now contaminating the unknown-song
investigation queue (people manually validate them thinking they're missing
songs). They're also subtracting from the COMMERCIAL airtime stat.

**Root cause hypothesis:** the Tier 2 MUSIC threshold treats any window with
a clear melodic backing track as MUSIC, even when the dominant signal is a
speech voice-over. Twisted Tea and Gatorade ads both have jingle/music beds
under the VO.

**Things to try next session:**

- **Stricter MUSIC gate**: require the MUSIC score to exceed SPEECH by some
  margin (e.g. `music_score - speech_score >= 0.15`), not just be the top
  class. Easy to A/B because we already log the per-window YAMNet scores.
- **Speech-energy lookahead**: when MUSIC wins but SPEECH is close, check if
  the next/previous 1-2 windows were SPEECH; if so, retain SPEECH.
- **Rescue path post-classify**: when a SONG segment matches neither
  audfprint nor Shazam over its entire duration, AND a Tier-3 transcript
  exists for adjacent windows mentioning a brand name, reclassify the
  segment as COMMERCIAL or DJ.

### 10. `Bad Omens - Dying To Love` and `In Color - Headlights` audfprint-after-promotion miss

**Specific instance of #8 worth its own bullet** because it has a clear
reproduction: both tracks were promoted via `songs promote` in earlier
sessions, the reference audio was downloaded and indexed, but the songs row
in the DB still shows `source=shazam` and `audfprint_track_id=NULL` after
multiple runs that should have triggered an audfprint match.

**Diagnostic steps:**

1. `audfprint match -d data/audfprint/songs.pklz <broadcast_clip.wav>` for a
   known timestamp where the track played. Compare score vs `min_count=45`.
2. Confirm the ref files are inside the rebuilt index:
   `audfprint list -d data/audfprint/songs.pklz | grep -i 'In Color'`.
3. If matches are below `min_count`, try re-downloading those refs with the
   explicit `--audio-format mp3` legacy path to rule out opus quality
   degradation. (We changed the default to "best" / no-transcode this
   session, so freshly promoted tracks may be webm/opus instead of mp3.)

### 11. Add M.I.A. - Paper Planes to local fingerprint catalog

**Status:** DONE this session. Track id 104 was Shazam-only with no
reference audio; promoted to tracklist, downloaded, and audfprint index
rebuilt with 151 files. Should resolve as Tier-1 on future runs.

### 12. Reference-codec audit + audfprint wrapper best-by-score fix (2026-05-31)

**Status:** DONE this session. Two large issues were uncovered during the
fingerprint eval audit and both have been fixed.

**Issue A — webm/opus refs poisoned the index.**
The eight tracks promoted under the new no-transcode `seed download`
default landed as `.webm` (opus). Their fingerprints came out
high-density and noisy (e.g. `Djo - Basic Being Basic.webm` had 500+
common hashes vs. ~50-300 for mp3 refs), which caused them to win
audfprint's `rank 0` slot across *unrelated* query clips with small
scores. Combined with our `--max-matches 1` setting that meant the real
higher-score match (sitting at `rank 1+`) was never reported.

Fix:

- Re-downloaded all eight `.webm` refs as MP3 192 kbps (`Coldplay - Fix
  You`, `Djo - Basic Being Basic`, `Fun. - We Are Young (feat. Janelle
  Monáe)`, `Green Day - When I Come Around`, `Linkin Park - Somewhere I
  Belong`, `M.I.A. - Paper Planes`, `Sum 41 - In Too Deep`, `The
  Cranberries - Zombie`). Old files quarantined under
  `data/reference/_quarantine_webm_20260531/`.
- Reverted `DownloadConfig` / CLI default from `--audio-format best`
  back to `--audio-format mp3 --audio-quality 192`. The no-transcode
  path remains available behind an explicit flag.
- Rebuilt `data/audfprint/songs.pklz` (151 mp3 files).

**Issue B — audfprint `rank 0` is not score-ordered.**
The wrapper used `--max-matches 1` and the parser returned the first
`Matched` line. Both assumptions are wrong: audfprint's `rank` field
groups hits by reference, and a noisy reference can sit at `rank 0`
with a low score even when the real best-by-score match is at `rank
1+`. This silently turned real recovery candidates into NOMATCHes.

Fix:

- `AudfprintConfig.max_matches` default raised from 1 to 5.
- `parse_audfprint_match_output` and `parse_audfprint_batch_output` now
  scan all `Matched` lines for the query and return the highest-score
  one. Regression tests `test_parse_picks_best_match_by_score` and
  `test_parse_batch_picks_best_match_per_query` cover both.

**Validated unknowns eval — before vs after (`data/eval/fingerprint_20260531_validated_unknowns/`):**

| min_count | recall (before repair) | recall (after repair) |
|----------:|-----------------------:|----------------------:|
| 45        | 0 / 7                  | 0 / 7                 |
| 30        | 0 / 7                  | 1 / 7 (Bad Omens 60)  |
| 20        | 0 / 7                  | 2 / 7 (+ White Stripes 22) |
| 10        | 0 / 7                  | 2 / 7                 |
| 5         | 0 / 7                  | 4 / 7 (+ In Color 10, + Cranberries 14) |

Three clips remain stubbornly unrecoverable even at the floor:

- **Wonderwall (block 21):** zero hash overlap with the Oasis ref at
  any threshold. The reference self-matches perfectly (690/806), so the
  ref is correct. Either the broadcast played a different mix (live /
  remaster variant) or the FM degradation that day stripped the
  high-frequency landmarks audfprint depends on. **Action:** try adding
  a second Wonderwall reference from a different YouTube source (e.g.
  the original 1995 master rather than the 2024 stereo remaster).
  **Enabler shipped 2026-06-01.** ``_split_track_id`` now strips
  alternate-reference markers (``(alt)``, ``(alt 2)``, ``(ref)``,
  ``(reference)``, ``(source)``, ``(v2)``, …) from titles so multiple
  reference recordings for the same song collapse onto one identity at
  upsert time. Workflow once an alternate source is identified:

  ```bash
  # 1. Manually download the alternate from a known URL (NOT a search
  #    template — controlled selection is the whole point of the
  #    experiment). Drop into the canonical reference directory with the
  #    (alt) suffix so the audfprint title parser recognises it.
  python -m yt_dlp -x --audio-format mp3 --audio-quality 192 \
    -o "data/reference/songs/Oasis - Wonderwall (alt).%(ext)s" \
    "<ALT_YT_URL>"

  # 2. Extend the audfprint index with just the new file.
  RADIO_CLASSIFIER_AUDFPRINT_INDEX_NCORES=1 \
    .venv/bin/radio-classifier fingerprint index \
      --dir data/reference/songs/ --extend \
      --glob "Oasis - Wonderwall (alt).mp3"

  # 3. Sanity-check both references self-match at high score.
  .venv/bin/radio-classifier fingerprint explain \
    -i "data/reference/songs/Oasis - Wonderwall.mp3"        --expected Wonderwall
  .venv/bin/radio-classifier fingerprint explain \
    -i "data/reference/songs/Oasis - Wonderwall (alt).mp3"  --expected Wonderwall

  # 4. Re-run the morning-capture block that previously had zero hash
  #    overlap and confirm at least one reference matches.
  .venv/bin/radio-classifier fingerprint explain \
    -i "<the block-21 wav>" --expected Wonderwall
  ```

  Diagnostic confirmed in this session: existing ``Oasis - Wonderwall.mp3``
  self-matches at count 6395 (rank 1), and the next 11 hash buckets are
  the same reference. So the reference file itself is fine — the audfprint
  miss is purely a mix-variance / FM-degradation issue, which is exactly
  what a second reference should mitigate.
- **M.I.A. Paper Planes (block 14):** best real score is 4. Genuine
  weak signal — clip likely contains heavy DJ talk or song bookends.
  Re-clipping a cleaner 30 s window from the body of the song may
  recover it.
- **Temper City Self Aware (block 6):** real score 16, but
  `Linkin Park - Somewhere I Belong.mp3` legitimately scores **67** on
  the same clip. The two songs share an identifiable mid-tempo
  alt-metal riff; this is a real fingerprint collision. The funnel's
  `low_confidence_fingerprint_required_repeats=3` defense should
  suppress it as long as adjacent windows disagree, but worth keeping
  on a watch list.

**Open follow-up (deferred):**

- ~~Consider lowering default `AudfprintConfig.min_count` from 45 to ~20.~~
  **DONE 2026-06-01.** Lowered to `30` (a compromise between the previous
  45 floor and the 20 the eval recommended). The Temper City / Linkin Park
  collision sits at 67, well above either threshold, and the
  `low_confidence_fingerprint_required_repeats` guard remains the
  false-positive backstop for scores in [30, 60). Re-run the morning-drive
  bimodal score check if any new collisions surface in long-running data.
- Wonderwall second-reference experiment (above). **Code enabler shipped
  2026-06-01** — alt-reference suffix parsing in ``_split_track_id``. The
  remaining step (download a specific alternate-source URL, extend the
  index, verify recall on the previously-zero-overlap block) needs an
  operator with network access and a chosen alternate source URL.
- ~~Build a CLI `fingerprint explain <clip> <expected-track>` diagnostic
  that prints the expected reference's score / rank even when another
  reference wins.~~ **DONE 2026-06-01.** New
  `radio-classifier fingerprint explain --input CLIP [--expected SUBSTR]`
  command surfaces every audfprint candidate (default top 20) with rank
  and score, and explicitly calls out whether ``--expected`` appears.

## Other items spotted during today's 12h run (lower priority)

- ~~**Julia Wolf "In My Room (Acoustic)" promo-vs-song detection:** 10 "spins" in 7m10s = ~43 sec per spin. That's a station promo, not a real play. Spin definition needs a heuristic to disambiguate (e.g., "if average spin duration < 50% of full track length, suppress as promo").~~ **DONE.** Spin tally now reports `full_spin_count` and `promo_spin_count` side by side (threshold `PROMO_MAX_SPIN_SECONDS = 90s`). `report songs` / `report artists` show `spins   promos` columns, the dashboard renders a `+N promo` pill, and the songs/artists sort order tiebreaks by non-promo airtime so teaser-heavy entries drop down the leaderboard. Julia Wolf went from a top-2 entry to #4 in the morning-DB songs report and out of the top-8 artists report.
- ~~**LINKIN PARK casing artifact:** artist display name shows ALL CAPS because most Shazam responses returned that way. Either normalize artist casing on write, or pick a different display-winner heuristic.~~ **DONE.** `BroadcastStore.upsert_song` now normalizes the known `LINKIN PARK` display artifact to `Linkin Park`, prefers mixed-case reference data over all-caps Shazam display text for the same song identity, and preserves legitimate acronyms like `AFI`. Existing rows in `data/eval/morning_20260530_24x30m.db` were normalized too.
- ~~**Modest Mouse "Picking Dragons' Pockets" Shazam-vs-tracklist dedup miss:** showed up as a "new" Shazam discovery despite being on the tracklist. Likely unicode apostrophe difference (U+2019 vs U+0027) or audfprint missing it at min_count=60. Worth a 30-min investigation.~~ **DONE (2026-06-01).** Root cause was two-sided: Shazam returned the typographic apostrophe (U+2019) while audfprint registered the reference recording with the filename-sanitizer underscore (`Picking Dragons_ Pockets.mp3` → title `Picking Dragons_ Pockets`). Three coordinated fixes: (1) `_display_key` now folds typographic apostrophes → ASCII and drops both `'` and `_` so all three variants collapse onto one identity; (2) `_prefer_display_value` ranks titles by punctuation quality (ASCII apostrophe > typographic > none > underscore artifact), and dedupe upgrades the survivor's display text in the same pass; (3) `safe_filename_stem` keeps apostrophes (instead of replacing with `_`) and pre-normalizes typographic → ASCII so new downloads never reintroduce the artifact. Running `dedupe_songs` on `data/store/broadcast.db` folded **10 duplicate pairs** (Modest Mouse, Staind, Royel Otis, Linkin Park, Blink-182, Cage The Elephant, Red Hot Chili Peppers, Sublime, Almost Monday, Dexter and The Moonrocks), repointing 43 broadcast events. 20 audfprint-only rows still show legacy underscores (no Shazam loser ever existed to upgrade them); these will only resolve when Shazam confirms the same song, or via a one-off operator backfill that reads titles from `tracklist.txt`. 9 new regression tests cover the identity collapse, display upgrade, promote-already-in-tracklist with underscore, and the safe_filename_stem behavior. Commercials Dedupe Item C (loosen minhash threshold) shipped in the same chat as part of this work — new tertiary token-cosine path catches same-ad reorderings that 3-shingle Jaccard misses.
- ~~**audfprint min_count=60 floor tuning:** might be too strict for some reference tracks. Could revisit per-track score floors or fall back to a lower floor if Shazam confirms.~~ **DONE.** The audfprint candidate floor is now `30` (lowered from 45 on 2026-06-01 after the validated-unknowns eval), and scores below the strong `60` threshold still require 3 adjacent same-track matches in the funnel before Tier 1 wins. This should recover weaker real matches without reopening the old one-off false-positive pile.
