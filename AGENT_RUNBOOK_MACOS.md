# Agent Runbook — macOS Standalone Operation

**Audience:** an automated operator running capture + classification on an Apple
Silicon Mac (M4 Pro, 24 GB). This is the **macOS parallel stack** under
`macos/`. The legacy WSL/NVIDIA runbook is unchanged:
[`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md).

Every command is copy-pasteable from the **repo root**.

> **Golden rules**
> 1. **Source `macos/env.defaults` every session** — Ollama is on **:11434**, not WSL's :11435.
> 2. **Never run `prereq-check --gpu` on Mac** — there is no NVIDIA CUDA stack.
> 3. **One capture at a time** — check for existing `rtl_fm` / capture processes first.
> 4. **Back up the DB** before any mutating maintenance (`data/backups/`).

---

## 0. Key paths & constants

| Thing | Value |
| --- | --- |
| Repo root | `/path/to/radio-classifier` |
| macOS entry point | [`macos/README.md`](macos/README.md) |
| Environment | `source macos/env.defaults` |
| Python | `.venv/bin/python` |
| Long-term DB | `data/store/broadcast.db` |
| Capture supervisor | `bash macos/scripts/capture_until_audio_hours.sh N` |
| Weekend master | `bash macos/scripts/weekend_capture.sh N` |
| Ollama | **port 11434** (Ollama.app / `brew install ollama`) |
| Whisper | `cpu` / `int8` (from `macos/env.defaults`) |

---

## 1. One-time install

```bash
cd /path/to/radio-classifier
bash macos/install.sh
```

Installs Homebrew deps (`librtlsdr`, `ffmpeg`), Python venv, and
`pip install -e ".[acoustic,shazam,seeding,dev]"` — **without** the Linux `[gpu]` extra.

Clone audfprint to `~/dev/audfprint` if not present. Pull `llama3.2:latest` via Ollama.

---

## 2. Every-session preflight

```bash
cd /path/to/radio-classifier
source macos/env.defaults
bash macos/scripts/preflight.sh
```

Manual checks:

```bash
# P1: venv
.venv/bin/radio-classifier --help >/dev/null && echo "P1 ok"

# P2: Ollama reachable (Metal)
curl -sf "${RADIO_CLASSIFIER_OLLAMA_HOST}/api/tags" | head -c 200; echo

# P3: RTL dongle
rtl_test -t

# P4: no duplicate capture
pgrep -af 'capture_until_audio_hours|continuous_capture_blocks|rtl_fm' || true

# P5: disk space (aim for ≥ 20 GB free on the data volume)
df -h .

# P6: prereq (Darwin — no --gpu)
.venv/bin/radio-classifier prereq-check --ollama
```

**Pass criteria:**
- P2 returns JSON from `/api/tags`.
- P3 completes without "No supported devices found".
- P4 prints nothing (no conflicting processes).
- P6 all lines show `ok`.

---

## 3. Smoke tests (before long runs)

Run on the MacBook in order:

### 3.1 RTL smoke (10 seconds)

```bash
source macos/env.defaults
timeout 10 rtl_fm -f 105.3M -M wbfm -s 48000 -r 48000 - 2>/dev/null | head -c 100000 >/dev/null
echo "rtl_fm smoke rc=$?"
```

### 3.2 Short capture chunk (2 minutes)

```bash
source macos/env.defaults
mkdir -p data/captures/mac_smoke
.venv/bin/python -m radio_classifier capture chunks \
  --frequency 105300000 \
  --device-index 0 \
  --sample-rate 48000 \
  --chunk-seconds 120 \
  --duration-limit 120 \
  --out-dir data/captures/mac_smoke \
  --run-id mac_smoke_$(date -u +%Y%m%dT%H%M%SZ)
```

### 3.3 Classify one chunk

```bash
source macos/env.defaults
WAV="$(ls -t data/captures/mac_smoke/*.wav | head -1)"
JSON="$(ls -t data/captures/mac_smoke/*.json | head -1)"
START="$(.venv/bin/python -c "import json; print(json.load(open('$JSON'))['capture_start_utc'])")"

.venv/bin/python -m radio_classifier classify \
  -i "$WAV" \
  --capture-start-utc "$START" \
  --whisper-device cpu \
  --whisper-compute-type int8 \
  --enable-shazam \
  --persist \
  --db-path data/store/broadcast.db \
  --progress
```

Confirm SONG and COMMERCIAL/DJ counts are non-zero in the output.

### 3.4 End-to-end supervisor (30 min audio-hours)

```bash
source macos/env.defaults
bash macos/scripts/capture_until_audio_hours.sh 0.5
```

---

## 4. Production capture

### 4.1 Bounded audio-hours (foreground)

```bash
source macos/env.defaults
bash macos/scripts/capture_until_audio_hours.sh 20
```

### 4.2 Long unattended run (detached)

```bash
source macos/env.defaults
nohup bash macos/scripts/weekend_capture.sh 88 > data/logs/mac_weekend_master.log 2>&1 &
echo $! > data/logs/mac_weekend_master.pid
```

Monitor:

```bash
tail -f data/logs/mac_weekend_master.log
pgrep -af 'weekend_capture|capture_until|continuous_capture|rtl_fm'
```

### 4.3 Stall watchdog

`macos/scripts/continuous_capture_blocks.sh` kills a wedged `rtl_fm` when the
in-progress WAV stops growing for 180s. The supervisor in
`capture_until_audio_hours.sh` restarts automatically.

---

## 5. Post-run cleanup

```bash
bash macos/scripts/post_run_cleanup.sh data/store/broadcast.db
```

Delegates to the shared `scripts/post_run_cleanup.sh` (backup, dedupe, enrich, reports).

---

## 6. Status & reports

```bash
source macos/env.defaults
DB=data/store/broadcast.db

.venv/bin/python -m radio_classifier runs list --db-path "$DB" --limit 10
.venv/bin/python -m radio_classifier report summary --db-path "$DB" --since 24h
.venv/bin/python -m radio_classifier report dashboard --db-path "$DB" --since 24h \
  --out data/reports/dashboard.html
```

For data older than 24h, use explicit `--from` / `--to` (not `--since 24h`).

---

## 7. Database portability

Copy `data/store/broadcast.db` from the WSL machine as-is. Schema:
[`db/schema.sql`](db/schema.sql). Query guide:
[`docs/data-visualization-brief.md`](docs/data-visualization-brief.md).

---

## 8. macOS vs WSL differences

| Setting | macOS | WSL legacy |
| --- | --- | --- |
| Scripts | `macos/scripts/` | `scripts/` |
| Ollama port | 11434 | 11435 |
| RTL attach | Direct USB | `usbipd attach --wsl` |
| pip extras | no `[gpu]` | `.[gpu]` |
| prereq-check | `--ollama` only | `--gpu --ollama` |
| Runbook | this file | `AGENT_RUNBOOK.md` |

**Never source WSL capture env on Mac** (port mismatch silently breaks Tier-3).

---

## 9. Troubleshooting

| Symptom | Action |
| --- | --- |
| All speech = `unknown` | Verify `echo $RADIO_CLASSIFIER_OLLAMA_HOST` is `:11434`; `curl .../api/tags` |
| `rtl_test -t` fails | Replug dongle; avoid unpowered USB-C hubs |
| Capture hangs, rtl_fm alive | Stall watchdog should kill within 3 min; supervisor restarts |
| TensorFlow/YAMNet install fails on arm64 | `pip install -e ".[acoustic]"` only; YAMNet forced CPU on Darwin |
| Whisper too slow | Expected on CPU; benchmark CoreML later (optional) |

---

## 10. Validation checklist (M4 hardware)

Complete before trusting multi-day runs:

1. `source macos/env.defaults && bash macos/scripts/preflight.sh`
2. `radio-classifier prereq-check --ollama`
3. `rtl_test -t` and 10s `rtl_fm` smoke
4. 2 min `capture chunks` → `classify` with non-zero events
5. `bash macos/scripts/capture_until_audio_hours.sh 0.5`
6. Copy existing `broadcast.db`; verify `report summary` against known data
