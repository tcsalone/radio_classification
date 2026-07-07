# macOS Standalone — radio-classifier

Run **capture + full 3-tier classification** natively on Apple Silicon (M4 Pro)
without WSL, `usbipd`, or NVIDIA CUDA wheels.

The legacy WSL/NVIDIA workflow lives under [`scripts/`](../scripts/) and
[`AGENT_RUNBOOK.md`](../AGENT_RUNBOOK.md). This tree is a **parallel operator
stack** — it does not replace or modify the WSL scripts.

## Quick start

```bash
cd /path/to/radio-classifier

# One-time install (Homebrew, venv, pip extras — no [gpu])
bash macos/install.sh

# Every session
source macos/env.defaults
bash macos/scripts/preflight.sh

# Short smoke capture (30 min of audio-hours)
bash macos/scripts/capture_until_audio_hours.sh 0.5

# Long run + cleanup (detached)
nohup bash macos/scripts/weekend_capture.sh 20 > data/logs/mac_weekend_master.log 2>&1 &
```

## Key differences from WSL

| Setting | macOS (`macos/env.defaults`) | WSL legacy (`scripts/`) |
|---------|------------------------------|-------------------------|
| Ollama host | `http://127.0.0.1:11434` | `http://127.0.0.1:11435` |
| Whisper | `cpu` / `int8` | `cpu` / `int8` (stable) |
| RTL attach | Plug dongle into USB | `usbipd attach --wsl` |
| pip extras | `.[acoustic,shazam,seeding,dev]` | `.[acoustic,gpu,...]` |
| GPU checks | Skip `prereq-check --gpu` | `prereq-check --gpu` |

## Directory layout

```
macos/
  README.md              ← you are here
  env.defaults           ← source before every run
  install.sh             ← one-time setup
  lib/file_size.sh       ← portable stat for stall watchdog
  scripts/
    preflight.sh
    continuous_capture_blocks.sh
    capture_until_audio_hours.sh
    weekend_capture.sh
    post_run_cleanup.sh  → delegates to scripts/post_run_cleanup.sh
    validate.sh          ← full M4 hardware checklist before long runs
```

## Operator manual

Full procedures: [`AGENT_RUNBOOK_MACOS.md`](../AGENT_RUNBOOK_MACOS.md).

## Database portability

Copy `data/store/broadcast.db` from the WSL machine as-is. See
[`docs/data-visualization-brief.md`](../docs/data-visualization-brief.md) for
schema and query starters.
