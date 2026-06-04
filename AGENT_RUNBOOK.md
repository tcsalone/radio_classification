# Agent Runbook — Day-to-Day Operation & Tuning

**Audience:** an automated operator (a cheaper 3rd-party or local LLM — e.g. Gemini,
or a local model) that runs the radio-classifier collection + classification jobs
on this WSL2 host. **Not** for codebase changes — those escalate to Cursor (see
[§9 Escalation to Cursor](#9-escalation-to-cursor)).

This file is the contract. Every command is copy-pasteable from the repo root.
Where a decision is required, the **threshold and the action are stated explicitly**
so you do not need judgment beyond comparing a number to a rule.

> **Golden rules**
> 1. **Read-only first.** Every mutating command has a `--dry-run` (or preview)
>    form. Always run the preview, check it against the rule, *then* `--apply`.
> 2. **Back up before any `--apply`** that touches the DB (see §4.1).
> 3. **Never edit source files, schema, or scripts.** If a rule says "the fix is
>    a code change," STOP and produce an escalation report (§9). Do not improvise.
> 4. **Idempotent by design.** Cleanup passes can be re-run safely; if unsure
>    whether a step ran, run its `--dry-run` and check the count is 0.
> 5. **One capture at a time.** Never start a capture if one is already running
>    (§3.0 preflight check P4).

---

## 0. Key paths & constants

| Thing | Value |
| --- | --- |
| Repo root | `/home/eamon/dev/radio-classifier` |
| Python | `.venv/bin/python` (or `.venv/bin/radio-classifier`) |
| Long-term DB | `data/store/broadcast.db` |
| Tracklist (Tier-1 source of truth) | `data/reference/tracklist.txt` |
| Reference audio | `data/reference/songs/` (all `.mp3`) |
| Fingerprint index | `data/audfprint/songs.pklz` |
| DB backups | `data/backups/` |
| Capture WAVs + sidecars | `data/captures/` |
| Logs | `data/logs/` |
| Reports (HTML/txt) | `data/reports/` |
| Capture supervisor | `scripts/capture_until_audio_hours.sh N` (N = audio hours) |
| I/O sampler | `scripts/io_sampler.py` |
| GPU Ollama | port **11435** (NOT the snap on 11434) |

**Required environment for any LLM/Tier-3 work** (export once per shell):

```bash
export RADIO_CLASSIFIER_OLLAMA_HOST=http://127.0.0.1:11435
export RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE=-1   # keep model resident (HDD fix)
```

---

## 1. The operating loop (high level)

```
                ┌─────────────────────────────────────────────┐
                │ 0. PREFLIGHT  (env, GPU, disk, no dup runs)   │
                └───────────────┬─────────────────────────────┘
                                ▼
   2. MONITOR  ◄────────  1. START CAPTURE (N audio hours)
   (during run)                │
                                ▼ (capture target reached)
                3. POST-RUN CLEANUP  (backup → merge → backfill → dedupe → enrich)
                                │
                                ▼
                4. DISCOVERY LOOP  (promote → download → reindex)
                                │
                                ▼
                5. REPORTS  (dashboard, artist-plays, discovered)
                                │
                                ▼
                6. HEALTH CHECKS  → if a rule trips a CODE issue → 9. ESCALATE
```

Phases 0–2 happen while capturing. Phases 3–6 run **after** the capture target
is reached. Phase 9 (escalation) is triggered by checks in phases 3 and 6.

---

## 2. Glossary of the 5 classes

`SONG`, `DJ`, `COMMERCIAL`, `STATION`, `PSA_NEWS`. The funnel is 3 tiers:
Tier 1 = audfprint fingerprint (deterministic, cheap), Tier 2 = YAMNet
(music/speech), Shazam fallback (network), Tier 3 = Whisper transcription +
local LLM (Ollama). Adding songs to Tier 1 (§4) is the main lever to reduce
expensive Tier-3/Shazam fallthrough.

---

## 3. Phase 0–2: Preflight, start, monitor

### 3.0 Preflight checks (run ALL before starting a capture)

```bash
cd /home/eamon/dev/radio-classifier

# P1: venv + CLI works
.venv/bin/radio-classifier --help >/dev/null && echo "P1 ok" || echo "P1 FAIL"

# P2: GPU Ollama is up AND on GPU (not CPU)
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama ps

# P3: free disk on the WSL volume (rootfs holds DB + WAVs)
df -h / | awk 'NR==2{print "rootfs avail:", $4, "used%:", $5}'

# P4: NO capture already running (must print nothing / no PIDs)
pgrep -af 'capture_until_audio_hours|continuous_capture_blocks|rtl_fm' | cut -c1-100

# P5: DB integrity
sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"
```

**Pass criteria:**
- P1 prints `P1 ok`.
- P2 lists `llama3.2:latest` with `100% GPU` and `UNTIL = Forever`. If the table
  is **empty**, the model is just cold — that is fine, it loads on first call.
  If `PROCESSOR` shows `100% CPU` → **STOP, escalate (§9 trigger E5)**: the snap
  Ollama is being used instead of the native GPU one.
- P3: `avail` ≥ **20 GB**. If below → run §6.4 (prune WAVs) before starting.
- P4: prints **nothing**. If any PID prints, a capture is already running — **do
  not start another**.
- P5: prints `ok`. If not → escalate (§9 trigger E1).

### 3.1 Start a capture (preferred: bounded by audio hours)

```bash
# 20 audio-hours into the long-term DB, resilient to USB/IP drops.
./scripts/capture_until_audio_hours.sh 20
```

This appends to `data/store/broadcast.db`, opens a `capture_runs` row per
iteration, prunes WAVs older than 7 days, and resumes after drops until 20 hours
of *actual audio* are captured. Run it in a background/detached terminal.

Useful overrides (defaults are good; only change on instruction):

```bash
DB_PATH=data/store/broadcast.db BLOCK_SECONDS=1800 RETENTION_DAYS=7 \
  ./scripts/capture_until_audio_hours.sh 20
```

### 3.2 Monitor during the run (every 30–60 min)

```bash
# M1: is the supervisor still alive?
pgrep -af capture_until_audio_hours | cut -c1-80 || echo "supervisor NOT running"

# M2: capture progress (latest iteration log)
ls -t data/logs/capture_until_*.log | head -1 | xargs tail -n 5

# M3: recent rows landing in the DB (last 2h event counts by class)
.venv/bin/python -m radio_classifier report summary --since 2h

# M4: model still resident + on GPU?
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama ps

# M5 (only if host feels slow): sample WSL I/O + swap for 10 min
#   device is the rootfs disk; find it with: df / | awk 'NR==2{print $1}'
.venv/bin/python scripts/io_sampler.py --interval 60 --device sdd \
  --stop-pid "$(pgrep -f capture_until_audio_hours | head -1)" \
  --out data/logs/io_sample_$(date -u +%Y%m%dT%H%M%SZ).log
```

**Monitor rules:**
- M1 empty + capture target not reached → supervisor died. Re-run §3.1 once; it
  resumes from captured audio. If it dies again within 10 min → escalate (E3).
- M3: if `SONG` count is **0** over 2h while the radio is on → Tier 1/2 likely
  broken or dongle is silent → escalate (E2).
- M4: `UNTIL` countdown shrinking toward 0 instead of `Forever` → the keep-alive
  env var was not exported; re-export §0 and restart the capture between blocks.
- M5: in the log, sustained `swap: pages_out > 0` across many samples AND
  `llama-server` I/O spikes → memory pressure (known issue, see
  `data/reports/hdd-vmmem-investigation-run4.md`). Not an emergency; note it in
  the run report. The mitigations (RAM bump / pagefile move) are already applied.

---

## 4. Phase 3: Post-run cleanup (after target reached)

Run these **in order**. Each is preview-then-apply. Record the before/after
numbers from the previews — they go into the run report (§7).

### 4.1 Back up the DB (ALWAYS, before any --apply)

```bash
mkdir -p data/backups
sqlite3 data/store/broadcast.db \
  ".backup 'data/backups/broadcast-$(date -u +%Y%m%dT%H%M%SZ).db'"
sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"   # expect: ok
```

### 4.2 Merge boundary commercials (overlapping-window orphans)

```bash
# Preview
.venv/bin/python -m radio_classifier commercials merge-boundaries \
  --db-path data/store/broadcast.db --dry-run
# Apply (only if preview > 0 merges and they look like same-ad)
.venv/bin/python -m radio_classifier commercials merge-boundaries \
  --db-path data/store/broadcast.db --apply
```
**Rule:** default `--min-similarity 0.55` is correct. Do NOT lower it below
`0.50`. If preview merges 0, that's fine (new captures self-absorb at capture
time). Re-running is idempotent.

### 4.3 Backfill brands (deterministic; LLM optional)

```bash
# Deterministic preview
.venv/bin/python -m radio_classifier commercials backfill-brands \
  --db-path data/store/broadcast.db --dry-run
# Deterministic apply
.venv/bin/python -m radio_classifier commercials backfill-brands \
  --db-path data/store/broadcast.db --apply
```
**Rule:** the deterministic pass is the reliable win. The optional `--llm` pass
has **low yield (~5%) and occasionally mints junk brands**; only run it if
explicitly asked, and review its `--dry-run` output for nonsense brand names
before `--apply`.

### 4.4 Dedupe commercial rows (brand-variant folding)

```bash
.venv/bin/python -m radio_classifier commercials dedupe \
  --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier commercials dedupe \
  --db-path data/store/broadcast.db --apply
```

### 4.5 Dedupe songs (shazam vs audfprint, casing/whitespace)

```bash
.venv/bin/python -m radio_classifier songs dedupe \
  --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier songs dedupe \
  --db-path data/store/broadcast.db --apply
```

### 4.6 Enrich release dates (MusicBrainz, rate-limited, network)

```bash
.venv/bin/python -m radio_classifier songs enrich-releases \
  --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier songs enrich-releases \
  --db-path data/store/broadcast.db
```
**Rule:** this hits the public MusicBrainz API and self-rate-limits. If it errors
on network, skip it and note in the report; it is non-critical and idempotent.

---

## 5. Phase 4: Discovery loop — add missing songs to Tier 1

This is the highest-value tuning action: songs Shazam discovered but that are not
in the audfprint index keep falling through to expensive tiers. Promote them so
Tier 1 catches them next time.

### 5.1 List discoveries still missing from the index

```bash
.venv/bin/python -m radio_classifier songs discovered \
  --db-path data/store/broadcast.db --since 24h --top 200 --min-plays 1 \
  | tee data/reports/songs-discovered-$(date -u +%Y%m%dT%H%M%SZ).txt
```
The output has columns: `id artist title plays last_heard tracklist review`.
Rows with `tracklist=missing` are candidates.

**Selection rule (apply automatically, but flag for human/Cursor if unsure):**
- ✅ Promote rows with `plays ≥ 1` and a real artist/title.
- ⚠️ EXCLUDE likely mis-IDs: stock/production-music artists (e.g. "Gavin Luke"),
  and rows whose artist looks wrong vs a sibling row of the same title (e.g.
  "Tours — No Woman, No Cry" when "Bob Marley — No Woman, No Cry" also appears).
- ✅ Keep legit live/acoustic cuts (a fingerprint only matches that recording).
- If a row's legitimacy is genuinely ambiguous, **do not guess** — leave it out
  and list it under "deferred discoveries" in the run report.

### 5.2 Back up the index + tracklist, then promote

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
cp -v data/audfprint/songs.pklz       "data/audfprint/songs_${TS}.pklz.bak"
cp -v data/reference/tracklist.txt    "data/reference/tracklist_${TS}.bak"

# Promote selected ids (repeat --song-id per id). Promote is dedupe-safe:
# it skips ids already in the tracklist, non-shazam ids, and rows missing fields.
.venv/bin/python -m radio_classifier songs promote \
  --db-path data/store/broadcast.db \
  --song-id 268 --song-id 202 --song-id 338   # … one --song-id per chosen id
```

### 5.3 Download reference audio (yt-dlp, mp3) — skips what's on disk

```bash
.venv/bin/python -m radio_classifier seed download \
  --tracklist data/reference/tracklist.txt --out data/reference/songs/ \
  2>&1 | tee data/logs/seed_download_$(date -u +%Y%m%dT%H%M%SZ).log
```
Final line reports `downloaded=N skipped=M failed=K total=T`.
**Rule on failures:** isolated `failed` rows are usually transient yt-dlp/YouTube
extraction errors. Retry each failed track once via a one-line tracklist:

```bash
printf 'ARTIST | TITLE\n' > /tmp/retry.txt
.venv/bin/python -m radio_classifier seed download \
  --tracklist /tmp/retry.txt --out data/reference/songs/
```
If a track fails twice → leave it out, list it in the report (not a code bug).

### 5.4 Rebuild the fingerprint index

```bash
# Sanity: reference dir should be all .mp3, no orphan .webm/.m4a
ls data/reference/songs/ | sed 's/.*\.//' | sort | uniq -c

# Rebuild (default out = data/audfprint/songs.pklz; rebuild is safer than --extend)
.venv/bin/python -m radio_classifier fingerprint index \
  --dir data/reference/songs/ \
  2>&1 | tee data/logs/fp_index_$(date -u +%Y%m%dT%H%M%SZ).log
```
Final line: `rebuilt index at … with N files`. **Rule:** `N` must be ≥ the
previous file count and ≥ the `.mp3` count from the sanity check. If non-mp3
files appear in the count → STOP, do not rely on the index, escalate (E4).

---

## 6. Phase 5–6: Reports & health checks

### 6.1 Generate the standard reports

```bash
# HTML dashboard
.venv/bin/python -m radio_classifier report dashboard --since 24h \
  --out data/reports/dashboard.html
# Top-artist play log
.venv/bin/python -m radio_classifier report artist-plays --since 24h --top 3 \
  --out data/reports/artist-plays.html
# Text summaries
.venv/bin/python -m radio_classifier report summary  --since 24h
.venv/bin/python -m radio_classifier report songs    --since 24h --top 25
.venv/bin/python -m radio_classifier report commercials --since 24h --top 25
.venv/bin/python -m radio_classifier report runs     --since 7d
```

### 6.2 Health checks with explicit thresholds

Run these after cleanup. Each maps a number to an action.

```bash
# H1: unbranded commercial share (lower is better)
.venv/bin/python -m radio_classifier report commercials --since 24h --top 50
```
- **H1 rule:** if the "Unknown/unbranded" bucket is **> 25%** of total commercial
  plays *after* §4.2–4.4, that exceeds expectation → escalate (E6) with the
  commercials report attached. (Routine range is well under that.)

```bash
# H2: Tier-1 coverage proxy — how many SONG plays still resolve via shazam
.venv/bin/python -m radio_classifier report songs-added --since 24h --source shazam --top 50
.venv/bin/python -m radio_classifier report songs-added --since 24h --source audfprint --top 50
```
- **H2 rule:** a large, recurring shazam list of the *same* songs run-over-run
  means the discovery loop (§5) is not being applied — apply it. If songs you
  already promoted still come back as `shazam source` repeatedly → escalate (E4):
  the index may not be matching them.

```bash
# H3: DB growth + integrity
du -h data/store/broadcast.db
sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"
```
- **H3 rule:** integrity must be `ok`. Any other output → escalate (E1).

### 6.3 DB backup retention (keep last 14)

```bash
ls -1t data/backups/broadcast-*.db | tail -n +15 | xargs -r rm -v
```

### 6.4 WAV prune (free disk; safe — only deletes completed WAVs)

```bash
./scripts/prune_old_wavs.sh data/captures            # 7-day default
RETENTION_DAYS=3 ./scripts/prune_old_wavs.sh data/captures   # if disk tight
```

---

## 7. Run report template (produce after every run)

Emit this as `data/reports/run-report-<UTC>.md`. It is both your audit log and
the raw material an escalation (§9) is built from.

```markdown
# Run report <UTC date>

## Capture
- target_hours: 20   captured_seconds: <from supervisor final line>
- capture_runs: <runs list output, last N>
- supervisor restarts: <count of iterations / drops>

## Cleanup (before → after)
- merge-boundaries: merged <X> of <scanned>
- backfill-brands (deterministic): branded <X>
- commercials dedupe: folded <X> rows
- songs dedupe: folded <X> rows
- enrich-releases: enriched <X> (or skipped: network)

## Discovery loop
- discovered missing: <count>
- promoted: <count>   excluded (mis-ID): <list ids+names>
- deferred (ambiguous): <list>
- download: downloaded=<N> skipped=<M> failed=<K>
- index: rebuilt with <N> files (prev <P>)

## Health
- H1 unbranded commercial share: <pct>%   (threshold 25%)
- H2 shazam vs audfprint song adds: <s> / <a>
- H3 db size: <MB>  integrity: ok
- M5 memory/swap note: <none | pressure observed>

## Escalations raised
- <none | E# with link to escalation file>
```

---

## 8. Tuning knobs (only the safe, runtime ones)

These are **operational** knobs you may turn without code changes. Anything not
listed here is a code change → escalate.

| Knob | Where | Default | When to change |
| --- | --- | --- | --- |
| Audio hours per capture | `capture_until_audio_hours.sh N` | 20 | per request |
| Block length | `BLOCK_SECONDS` env | 1800 | rarely; smaller = more frequent DB writes |
| WAV retention | `RETENTION_DAYS` env / prune script | 7 | shrink if disk tight |
| Ollama keep-alive | `RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE` | `-1` | `30m` to free GPU RAM when idle |
| Ollama host/model | `RADIO_CLASSIFIER_OLLAMA_HOST` / `_MODEL` | 11435 / llama3.2 | only if server moves |
| merge-boundaries similarity | `--min-similarity` | 0.55 | 0.55–0.65 only; never < 0.50 |
| discovery min plays | `songs discovered --min-plays` | 1 | raise to 2–3 to promote only confident IDs |

**Forbidden without escalation:** editing thresholds *in code*, schema changes,
new CLI flags, changing the funnel tier order, prompt/few-shot edits, brand alias
table edits. These are all code and belong in Cursor.

---

## 9. Escalation to Cursor

When a rule below trips, **stop the affected workflow** and write a structured
escalation file `data/reports/escalation-<UTC>.md` using the template at the end
of this section. Then hand that file to Cursor (the expensive model) for a code
change. Do **not** attempt the fix yourself.

### 9.1 Escalation triggers

| ID | Trigger (observed) | Likely code-level cause |
| --- | --- | --- |
| **E1** | `PRAGMA integrity_check` ≠ `ok`; or DB locked errors | persistence / migration bug |
| **E2** | M3 shows `SONG`=0 for ≥2h with radio confirmed on | ingest/Tier-1/Tier-2 regression |
| **E3** | Supervisor dies repeatedly (<10 min) across restarts | capture script / USB-IP handling |
| **E4** | Promoted+indexed songs keep returning as `shazam`; or index file count includes non-mp3 | fingerprint matching / index build |
| **E5** | Ollama runs `100% CPU` despite native binary | GPU/Ollama config or client default |
| **E6** | Unbranded commercials > 25% after full cleanup | classifier prompt / brand extraction logic |
| **E7** | A cleanup `--dry-run` proposes obviously wrong changes (junk brands, wrong song folds) | dedupe/backfill heuristic |
| **E8** | Same yt-dlp failure mode on *many* tracks at once | downloader / yt-dlp version |
| **E9** | Any traceback/stack from a CLI command | bug — attach full traceback |

### 9.2 Data to collect for EVERY escalation (this is the feedback to Cursor)

Gather these into the escalation file so Cursor has runtime evidence without
re-running anything:

1. **The exact command(s)** you ran (copy-paste, with flags).
2. **Full stdout/stderr** of the failing command (or the last 100 lines of the
   relevant log under `data/logs/`).
3. **The check that tripped** (which H#/M#/E# rule, the number observed vs the
   threshold).
4. **DB facts** (run, do not interpret):
   ```bash
   sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"
   sqlite3 data/store/broadcast.db "SELECT category, COUNT(*) FROM broadcast_events GROUP BY category;"
   sqlite3 data/store/broadcast.db "SELECT source, COUNT(*) FROM songs GROUP BY source;"
   .venv/bin/python -m radio_classifier runs list --db-path data/store/broadcast.db --limit 5
   ```
5. **Environment snapshot:**
   ```bash
   .venv/bin/radio-classifier --help | head -1
   OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama ps
   df -h / | awk 'NR==2'
   git -C /home/eamon/dev/radio-classifier rev-parse --short HEAD
   git -C /home/eamon/dev/radio-classifier status --porcelain
   ```
6. **A representative sample** of the bad data (e.g. 5–10 offending rows):
   ```bash
   # example for E6/E7 — unbranded commercial transcripts
   sqlite3 -header -column data/store/broadcast.db \
     "SELECT id, substr(transcript_excerpt,1,80) FROM broadcast_events
      WHERE category='COMMERCIAL' AND brand_id IS NULL
        AND (brand_name IS NULL OR brand_name='') LIMIT 10;"
   ```
7. **A backup** taken at the moment of failure (§4.1) so Cursor can reproduce
   against the exact state. Note its filename in the report.
8. **What you already tried** (which dry-runs, which retries) and the result.

### 9.3 What NOT to send / do

- Do not paste the entire DB or multi-MB logs — send the **tail + a sample**.
- Do not attempt schema edits, migrations, or source edits to "work around" it.
- Do not delete or rewrite history; preserve the failing artifacts.

### 9.4 Escalation file template

```markdown
# Escalation <UTC> — <E#> <one-line title>

## Trigger
- Rule: <E# / H# / M#>. Observed: <value>. Threshold/expected: <value>.

## Command(s) run
```
<exact commands>
```

## Output (tail / full for short)
```
<stdout + stderr, or last 100 log lines>
```

## DB + env snapshot
<paste outputs from §9.2 items 4 and 5>

## Bad-data sample
<paste §9.2 item 6>

## State for reproduction
- DB backup: data/backups/broadcast-<UTC>.db
- git HEAD: <short sha>   dirty: <yes/no + porcelain>

## Already tried
- <dry-runs, retries, and their results>

## Requested change (operator's best guess, optional)
- <plain-language description of what seems wrong; Cursor decides the fix>
```

---

## 10. Quick command index (cheat sheet)

```bash
# preflight
OLLAMA_HOST=127.0.0.1:11435 ~/.local/ollama/bin/ollama ps
sqlite3 data/store/broadcast.db "PRAGMA integrity_check;"

# capture
./scripts/capture_until_audio_hours.sh 20

# post-run cleanup (each: --dry-run then --apply)
.venv/bin/python -m radio_classifier commercials merge-boundaries --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier commercials backfill-brands  --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier commercials dedupe           --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier songs dedupe                 --db-path data/store/broadcast.db --dry-run
.venv/bin/python -m radio_classifier songs enrich-releases        --db-path data/store/broadcast.db --dry-run

# discovery loop
.venv/bin/python -m radio_classifier songs discovered --db-path data/store/broadcast.db --since 24h --top 200
.venv/bin/python -m radio_classifier songs promote    --db-path data/store/broadcast.db --song-id <ID> …
.venv/bin/python -m radio_classifier seed download     --tracklist data/reference/tracklist.txt --out data/reference/songs/
.venv/bin/python -m radio_classifier fingerprint index --dir data/reference/songs/

# reports
.venv/bin/python -m radio_classifier report dashboard    --since 24h --out data/reports/dashboard.html
.venv/bin/python -m radio_classifier report artist-plays --since 24h --top 3 --out data/reports/artist-plays.html
.venv/bin/python -m radio_classifier report summary      --since 24h
```

---

*This runbook describes runtime operation only. The authoritative reference for
setup and rationale is `OPERATIONS.md`; the spec is `SPEC.md`. When in doubt,
prefer read-only commands and escalate to Cursor with evidence (§9).*
