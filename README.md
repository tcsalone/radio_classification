# radio-classifier

Local terrestrial-radio classifier. RTL-SDR captures an FM stream, a
three-tier funnel labels every contiguous slice as one of **SONG, DJ,
COMMERCIAL, STATION, PSA_NEWS** (with brand attribution where
possible), and SQLite stores timestamps + duration for offline analysis.

Forked from [`live105sux`](../live105sux/). Authoritative design
document: **[`SPEC.md`](SPEC.md)**.

## The funnel

| Tier | Purpose | Cost | Backend |
|---|---|---|---|
| **1** | Identify known songs from the station's seeded rotation. | Lowest (CPU). | [`audfprint`](https://github.com/dpwe/audfprint), songs only — commercials are **not** fingerprinted. |
| **2** | If Tier 1 misses, decide MUSIC vs SPEECH vs OTHER. | Medium (GPU). | YAMNet (TensorFlow Hub) or PANNs CNN14. |
| **3** | Transcribe speech and 5-class it with a local LLM. | Heaviest (GPU). | `faster-whisper medium.en` + Ollama 3B/8B Q4. |

The optional `shazamio` fallback handles unknown songs when explicitly
enabled with `--enable-shazam` (off by default).

## Prerequisites

- **Windows 10 + WSL 2** (Ubuntu) with **CUDA Toolkit for WSL** and an
  **NVIDIA RTX 2080 Ti** (or better). `nvidia-smi` must work inside WSL.
- **RTL-SDR dongle** attached via [`usbipd-win`](https://github.com/dorssel/usbipd-win).
- **`rtl_fm`** on PATH (`apt install rtl-sdr`).
- **[Ollama](https://ollama.com/)** running locally; pull e.g.
  `ollama pull llama3.2`.

Run `radio-classifier prereq-check --gpu` after install to verify all
of the above before live capture.

## Install (development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[acoustic,gpu,fingerprint,dev]"
# Optional:
# pip install -e ".[shazam]"       # network song ID fallback
# pip install -e ".[seeding]"      # tracklist scrape + yt-dlp toolchain
```

The `[gpu]` extra installs `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, and
`nvidia-cuda-nvrtc-cu12` wheels. `radio-classifier` automatically
`dlopen()`s the shared libraries from those wheels before constructing
`WhisperModel`, so **no `LD_LIBRARY_PATH` configuration is required**.

If `prereq-check --gpu` still reports `Library libcublas.so.12 is not found`,
the wheels are missing — run `pip install -e ".[gpu]"`.

### Install audfprint

`audfprint` is not packaged for PyPI and the upstream GitHub repo does not
include `setup.py` / `pyproject.toml`, so pip cannot install it as a normal
dependency. Install it as an external CLI tool:

```bash
sudo apt-get install -y ffmpeg git
git clone https://github.com/dpwe/audfprint.git ~/dev/audfprint
cd ~/dev/audfprint
pip install -r requirements.txt
chmod +x audfprint.py
```

If you clone it to `~/dev/audfprint`, `radio-classifier` auto-discovers
`~/dev/audfprint/audfprint.py` from any terminal. You can also add that
directory to `PATH` or explicitly point the app at the script:

```bash
export RADIO_CLASSIFIER_AUDFPRINT_BIN="python ~/dev/audfprint/audfprint.py"
```

To make the explicit override permanent across terminals:

```bash
echo 'export RADIO_CLASSIFIER_AUDFPRINT_BIN="python ~/dev/audfprint/audfprint.py"' >> ~/.bashrc
```

`radio-classifier prereq-check` verifies that the tool is discoverable.

Run tests:

```bash
pytest -q
```

## Quickstart

```bash
# 1. Seed the song fingerprint database (one-time)
radio-classifier fingerprint index --dir data/reference/songs/ \
                                   --out data/audfprint/songs.pklz

# 2. Initialize the SQLite event log
radio-classifier db init

# 3. Verify GPU + Ollama
radio-classifier prereq-check --gpu

# 4. Live capture (105.3 MHz, persist segments to SQLite)
radio-classifier ingest --persist --duration-limit 3600 -v

# 5. Reports
radio-classifier report commercials --since 24h --top 10
radio-classifier report brands       --since 24h
radio-classifier report songs        --since 24h
radio-classifier report timeline     --since 1h
radio-classifier report summary      --since 24h
```

## Privacy & legal

- Transcripts may contain PII (caller names, contest mentions). Stored
  in plain text under `data/` by default.
- No outbound network in the default ingest path.
- `--enable-shazam` sends fingerprints to a third-party service.
- The optional seeding toolchain uses `yt-dlp`; operator is responsible
  for ToS compliance.

See [`SPEC.md`](SPEC.md) for the full architecture, schema, and
acceptance criteria.
