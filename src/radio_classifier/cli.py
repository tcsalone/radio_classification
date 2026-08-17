"""Command-line interface for radio-classifier.

Subcommands:

* ``prereq-check``  — diagnostics for GPU / CUDA / Ollama / rtl_fm / audfprint.
* ``db init``        — apply ``db/schema.sql`` (schema v3) to a SQLite file.
* ``db migrate-from-live105sux`` — port a live105sux v1 SQLite into the current schema.
* ``fingerprint index`` — build / extend the audfprint song index.
* ``fingerprint eval``  — run the recall harness against a truth CSV.
* ``fingerprint explain`` — show every audfprint candidate score for one clip.
* ``ingest``         — live RTL-SDR capture through the 3-tier funnel.
* ``capture chunks`` — continuous RTL-SDR capture into fixed WAV chunks.
* ``classify``       — offline 3-tier funnel on a WAV file.
* ``runs``           — open/close/list capture-run provenance rows.
* ``report``         — CLI-only reports (commercials / brands / songs / artists / timeline / summary).
* ``seed scrape``    — print a tracklist parsed from a station page.
* ``seed download``  — fetch reference audio via yt-dlp (``[seeding]``).
* ``songs discovered`` — list Shazam-discovered songs (and their tracklist status).
* ``songs promote``    — append selected Shazam discoveries to ``tracklist.txt``.
* ``songs dedupe``     — fold duplicate ``songs`` rows (e.g. shazam + audfprint
  pairs for the same track) and re-point ``broadcast_events`` at the survivor.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import wave
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from radio_classifier.ingest.rtl_fm import RtlFmExitedError, RtlFmStream
from radio_classifier.ingest.wav import read_mono_s16le_wav, write_mono_s16le_wav
from radio_classifier.ingest.windows import iter_overlapping_windows
from radio_classifier.persistence import BroadcastStore, persist_finalize, persist_input
from radio_classifier.segments import SegmentReducer
from radio_classifier.version import pipeline_version


def _project_root() -> Path:
    """``src/radio_classifier/cli.py`` → parents[2] = repo root."""
    return Path(__file__).resolve().parents[2]


def _default_db_path() -> Path:
    return _project_root() / "data" / "store" / "broadcast.db"


class _CliConfigError(ValueError):
    """Raised when a CLI argument is structurally invalid (e.g. empty path).

    Distinguished from generic ``ValueError`` so :func:`main` can convert it
    into a clean stderr message + non-zero exit code without printing a
    Python traceback.
    """


def _resolve_db_path(raw: Path | None) -> Path:
    """Resolve a ``--db-path`` argument into a concrete :class:`Path`.

    *  ``None`` → fall back to :func:`_default_db_path`.
    *  Empty string / ``"."`` / whitespace-only → :class:`_CliConfigError`.
       This catches the common ``--db-path "$DB"`` foot-gun where ``$DB``
       was never set in the operator's shell. Before this guard, an empty
       string ended up opening the current working directory as a SQLite
       file and surfaced as an opaque ``sqlite3.OperationalError`` traceback.
    *  Anything else passes through unchanged.
    """
    if raw is None:
        return _default_db_path()
    text = str(raw).strip()
    if not text or text == ".":
        raise _CliConfigError(
            "--db-path is empty. If you used a shell variable like `--db-path "
            '"$DB"`, make sure $DB is set in this shell (e.g. '
            "`export DB=data/eval/foo.db`), or pass the path literally."
        )
    return raw


def _default_index_path() -> Path:
    return _project_root() / "data" / "audfprint" / "songs.pklz"


def _default_tracklist_path() -> Path:
    return _project_root() / "data" / "reference" / "tracklist.txt"


def _resolve_capture_wav_path(user_path: Path, clock_start_ns: int) -> Path:
    """Resolve the ``--capture-wav`` argument to a concrete file path.

    Special value ``Path("auto")`` (case-insensitive) expands to
    ``data/captures/<UTC>.wav`` under the project root so operators have a
    predictable place to look for the human-validation WAV without having to
    invent paths.
    """
    if str(user_path).lower() == "auto":
        ts = (
            datetime.fromtimestamp(clock_start_ns / 1e9, tz=timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ")
        )
        return _project_root() / "data" / "captures" / f"{ts}.wav"
    return user_path


def _iso_from_time_ns(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1e9, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_time_ns(value: str) -> int:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _CliConfigError(f"invalid UTC timestamp: {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def _normalize_utc_text(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# --------------------------------------------------------------------- parser
def _add_persist_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--persist",
        action="store_true",
        help="Append segment rows to SQLite (default DB: data/store/broadcast.db)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite database file (default: <repo>/data/store/broadcast.db)",
    )
    p.add_argument(
        "--no-wal",
        action="store_true",
        help="Disable SQLite WAL journal mode",
    )


def _add_window_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--window-seconds", type=float, default=20.0, help="Analysis window length (s)")
    p.add_argument("--overlap-fraction", type=float, default=0.5, help="Window overlap (0..1)")


def _add_funnel_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--audfprint-index",
        type=Path,
        default=None,
        help="audfprint index file (default: data/audfprint/songs.pklz)",
    )
    p.add_argument(
        "--audfprint-min-count",
        type=int,
        default=30,
        help=(
            "Minimum audfprint common-hash count to surface a Tier-1 candidate "
            "(default: 30; low scores still require extra adjacent confirmation)"
        ),
    )
    p.add_argument(
        "--no-tier1",
        action="store_true",
        help="Skip Tier 1 audfprint matching (debug)",
    )
    p.add_argument(
        "--no-tier2",
        action="store_true",
        help="Skip Tier 2 acoustic gating (debug; goes straight to Tier 3)",
    )
    p.add_argument(
        "--no-tier3",
        action="store_true",
        help="Skip Tier 3 speech transcription + LLM (debug)",
    )
    p.add_argument(
        "--enable-shazam",
        action="store_true",
        help="Allow shazamio fallback when Tier 1 misses and Tier 2 says MUSIC",
    )
    p.add_argument(
        "--shazam-recheck-windows",
        type=int,
        default=4,
        help=(
            "When Shazam fallback is enabled, reuse a Shazam result for this "
            "many consecutive unknown-music windows before rechecking "
            "(default: 4; use 1 to call every window)"
        ),
    )
    p.add_argument(
        "--unknown-music-rescue-speech-margin",
        type=float,
        default=0.15,
        help=(
            "After audfprint and Shazam both miss, run Tier 3 rescue only when "
            "YAMNet MUSIC is within this margin of SPEECH (default: 0.15; "
            "use 0 to disable)"
        ),
    )
    p.add_argument(
        "--whisper-backend",
        type=str,
        choices=("faster-whisper", "mlx"),
        default=os.environ.get("WHISPER_BACKEND", "faster-whisper"),
        help=(
            "Speech backend: 'faster-whisper' (CPU/CUDA) or 'mlx' (Apple Metal, "
            "Apple Silicon only). Default from $WHISPER_BACKEND or faster-whisper."
        ),
    )
    p.add_argument(
        "--whisper-model",
        type=str,
        default=os.environ.get("WHISPER_MODEL", "medium.en"),
        help=(
            "Model size/path (faster-whisper) or HF repo/path (mlx, e.g. "
            "mlx-community/whisper-large-v3-turbo). Default from $WHISPER_MODEL "
            "or medium.en."
        ),
    )
    p.add_argument(
        "--whisper-device",
        type=str,
        default="cuda",
        help="Device for faster-whisper (default: cuda; ignored by mlx backend)",
    )
    p.add_argument(
        "--whisper-compute-type",
        type=str,
        default="float16",
        help="Compute type for faster-whisper (default: float16)",
    )
    p.add_argument(
        "--whisper-language",
        type=str,
        default="en",
        help='Language code, or "auto" for detection (default: en)',
    )
    p.add_argument(
        "--whisper-beam-size",
        type=int,
        default=None,
        help=(
            "Whisper decode beam size (default: faster-whisper default; "
            "try 1 only for speed/quality experiments)"
        ),
    )
    p.add_argument(
        "--whisper-vad-filter",
        action="store_true",
        help="Enable faster-whisper VAD filtering inside speech windows (default: off)",
    )
    p.add_argument(
        "--speech-min-rms",
        type=float,
        default=750.0,
        help=(
            "Skip Tier 3 when a window's PCM RMS is below this threshold "
            "(default: 750; use 0 to disable)"
        ),
    )
    p.add_argument(
        "--ollama-base-url",
        type=str,
        default=os.environ.get(
            "RADIO_CLASSIFIER_OLLAMA_HOST", "http://127.0.0.1:11434"
        ),
        help="Ollama base URL",
    )
    p.add_argument(
        "--ollama-model",
        type=str,
        default=os.environ.get(
            "RADIO_CLASSIFIER_OLLAMA_MODEL", "llama3.2:latest"
        ),
        help="Ollama model name (override with RADIO_CLASSIFIER_OLLAMA_MODEL)",
    )
    p.add_argument(
        "--tier2-min-prob",
        type=float,
        default=0.25,
        help="Tier-2 minimum bucket probability before deferring to SPEECH route",
    )
    p.add_argument(
        "--no-tier2-speech-bias",
        action="store_true",
        help="Disable Tier-2 DJ-over-music bias (prefer MUSIC when probs are close)",
    )


def _add_json_lines(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--json-lines",
        action="store_true",
        help="Emit one JSON object per window to stdout (human text still to stderr)",
    )


def _add_progress_arguments(p: argparse.ArgumentParser) -> None:
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=None,
        help="Show classification progress, ETA, and running category counts on stderr",
    )
    group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable classification progress output",
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="radio-classifier", description="Local terrestrial radio classifier")
    sub = p.add_subparsers(dest="command", required=True)

    # ---- prereq-check
    pc = sub.add_parser("prereq-check", help="Run runtime pre-flight checks")
    pc.add_argument("--gpu", action="store_true", help="Run GPU / CUDA / faster-whisper checks")
    pc.add_argument("--ollama", action="store_true", help="Also probe Ollama /api/tags")

    # ---- db
    db = sub.add_parser("db", help="Database management")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_init = db_sub.add_parser("init", help="Initialise SQLite schema v3")
    db_init.add_argument("--db-path", type=Path, default=None)

    mig = db_sub.add_parser(
        "migrate-from-live105sux", help="Migrate a live105sux v1 DB into a fresh radio-classifier DB"
    )
    mig.add_argument("--src", type=Path, required=True, help="Source live105sux SQLite path")
    mig.add_argument("--dst", type=Path, required=True, help="Destination radio-classifier SQLite path")

    # ---- fingerprint
    fp = sub.add_parser("fingerprint", help="Manage the Tier-1 song fingerprint index")
    fp_sub = fp.add_subparsers(dest="fp_command", required=True)
    fp_idx = fp_sub.add_parser(
        "index",
        help="Build (or rebuild) the audfprint song index from a directory of reference audio",
    )
    fp_idx.add_argument("--dir", type=Path, required=True, help="Directory of reference audio files")
    fp_idx.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Index file path (default: data/audfprint/songs.pklz). Will be REBUILT if it exists.",
    )
    fp_idx.add_argument(
        "--extend",
        action="store_true",
        help=(
            "Append the new files into the existing index instead of rebuilding. "
            "Default is rebuild because rebuild is safer (no stale entries, no audfprint "
            "extend-mode hangs on bad inputs)."
        ),
    )
    fp_idx.add_argument("--glob", type=str, default="**/*", help="File glob inside --dir (default: all files)")

    fp_eval = fp_sub.add_parser("eval", help="Recall harness against truth CSV")
    fp_eval.add_argument("--index", type=Path, default=None, help="Index file (default: data/audfprint/songs.pklz)")
    fp_eval.add_argument("--truth", type=Path, required=True, help="CSV: clip,song_id or clip,artist,title")
    fp_eval.add_argument(
        "--audfprint-min-count",
        type=int,
        default=30,
        help="Minimum audfprint common-hash count for eval candidates (default: 30)",
    )

    fp_explain = fp_sub.add_parser(
        "explain",
        help="Run audfprint match on one clip and print every candidate score",
    )
    fp_explain.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="WAV clip to match (a captured broadcast snippet, not a reference file)",
    )
    fp_explain.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Index file (default: data/audfprint/songs.pklz)",
    )
    fp_explain.add_argument(
        "--max-matches",
        type=int,
        default=20,
        help="Maximum candidates audfprint should consider (default: 20)",
    )
    fp_explain.add_argument(
        "--min-count",
        type=int,
        default=1,
        help=(
            "Floor on candidate hash counts (default: 1). The default surfaces "
            "very weak candidates the production funnel would reject so you "
            "can compare them against the expected track."
        ),
    )
    fp_explain.add_argument(
        "--expected",
        type=str,
        default=None,
        help=(
            "Substring (case-insensitive) of the expected reference name. "
            "When supplied the report calls out whether that reference "
            "appeared and at what rank/score."
        ),
    )

    # ---- classify (offline)
    cls = sub.add_parser("classify", help="3-tier funnel over a WAV file")
    cls.add_argument("-i", "--input", type=Path, required=True, help="Mono 16-bit WAV path")
    cls.add_argument("--sample-rate-override", type=int, default=None)
    cls.add_argument(
        "--capture-start-utc",
        type=str,
        default=None,
        help=(
            "UTC start timestamp for the input WAV. Use this for delayed "
            "classification of continuously captured chunks so DB events use "
            "broadcast time instead of classification time."
        ),
    )
    cls.add_argument(
        "--capture-run-id",
        type=int,
        default=None,
        help="capture_runs.id to stamp onto persisted broadcast_events",
    )
    _add_window_arguments(cls)
    _add_funnel_arguments(cls)
    _add_persist_arguments(cls)
    _add_json_lines(cls)
    _add_progress_arguments(cls)
    cls.add_argument(
        "--no-batch-tier1",
        action="store_true",
        help="Disable offline audfprint batch matching and use per-window Tier 1 calls",
    )
    cls.add_argument("-v", "--verbose", action="store_true")

    # ---- ingest (live RTL-SDR)
    ing = sub.add_parser("ingest", help="Live RTL-SDR capture + 3-tier funnel")
    ing.add_argument("--frequency", type=float, default=105.3e6, help="Center freq Hz")
    ing.add_argument("--device-index", type=int, default=0)
    ing.add_argument("--sample-rate", type=int, default=48_000)
    ing.add_argument(
        "--duration-limit",
        type=float,
        default=None,
        help="Stop after this many seconds of wall clock (for testing)",
    )
    ing.add_argument(
        "--capture-wav",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Also write the full captured PCM to a mono 16-bit WAV at PATH "
            "(for manual ground-truth listening). Pass 'auto' to write to "
            "data/captures/<UTC>.wav."
        ),
    )
    _add_window_arguments(ing)
    _add_funnel_arguments(ing)
    _add_persist_arguments(ing)
    _add_json_lines(ing)
    ing.add_argument("-v", "--verbose", action="store_true")

    # ---- capture (record-only helpers)
    cap = sub.add_parser("capture", help="Record RTL-SDR audio without classification")
    cap_sub = cap.add_subparsers(dest="capture_command", required=True)
    cap_chunks = cap_sub.add_parser(
        "chunks",
        help="Continuously capture one RTL-SDR stream into fixed-size WAV chunks",
    )
    cap_chunks.add_argument("--frequency", type=float, default=105.3e6, help="Center freq Hz")
    cap_chunks.add_argument("--device-index", type=int, default=0)
    cap_chunks.add_argument("--sample-rate", type=int, default=48_000)
    cap_chunks.add_argument("--chunk-seconds", type=float, default=1800.0)
    cap_chunks.add_argument("--duration-limit", type=float, default=None)
    cap_chunks.add_argument("--out-dir", type=Path, required=True)
    cap_chunks.add_argument("--run-id", type=str, default=None)

    # ---- runs (capture-run provenance)
    runs = sub.add_parser("runs", help="Manage capture run provenance rows")
    runs_sub = runs.add_subparsers(dest="runs_command", required=True)
    runs_start = runs_sub.add_parser("start", help="Open a capture run row")
    runs_start.add_argument("--db-path", type=Path, default=None)
    runs_start.add_argument("--run-id", type=str, required=True)
    runs_start.add_argument("--started-utc", type=str, default=None)
    runs_start.add_argument("--pipeline-version", type=str, default=None)
    runs_start.add_argument("--host", type=str, default=None)
    runs_start.add_argument("--notes", type=str, default=None)

    runs_end = runs_sub.add_parser("end", help="Close a capture run row")
    runs_end.add_argument("--db-path", type=Path, default=None)
    runs_end.add_argument("--run-id", type=str, required=True)
    runs_end.add_argument("--ended-utc", type=str, default=None)
    runs_end.add_argument("--notes", type=str, default=None)

    runs_list = runs_sub.add_parser("list", help="List recent capture runs")
    runs_list.add_argument("--db-path", type=Path, default=None)
    runs_list.add_argument("--limit", type=int, default=20)

    # ---- report
    rep = sub.add_parser("report", help="CLI reports against a radio-classifier SQLite DB")
    rep_sub = rep.add_subparsers(dest="report_command", required=True)
    for name in (
        "commercials",
        "brands",
        "songs",
        "songs-added",
        "songs-timeline",
        "artists",
        "artist-plays",
        "timeline",
        "summary",
        "dashboard",
        "runs",
    ):
        help_text = None
        if name == "artists":
            help_text = (
                "Per-artist airtime rollup (case-folded dedup, spins, distinct "
                "titles, total airtime)"
            )
        elif name == "artist-plays":
            help_text = (
                "Write a static HTML play log: every song play for the top N "
                "artists, grouped per artist (default top 3)"
            )
        elif name == "songs-timeline":
            help_text = "Chronological SONG-only listening log"
        elif name == "songs-added":
            help_text = "Songs first added to the catalog in a time window"
        elif name == "dashboard":
            help_text = "Write a static HTML metrics dashboard"
        elif name == "runs":
            help_text = "Capture run provenance summary"
        r = rep_sub.add_parser(name, help=help_text)
        r.add_argument("--db-path", type=Path, default=None)
        r.add_argument("--since", type=str, default=None, help="Window start as duration/ISO (default: 24h)")
        r.add_argument("--from", dest="from_utc", type=str, default=None, help="Explicit inclusive UTC window start")
        r.add_argument("--to", dest="until_utc", type=str, default=None, help="Explicit exclusive UTC window end")
        default_top = 3 if name == "artist-plays" else 10
        r.add_argument("--top", type=int, default=default_top, help=f"Limit (default {default_top})")
        if name == "commercials":
            r.add_argument("--brand", type=str, default=None, help="Filter to a single brand")
        if name == "songs-added":
            r.add_argument(
                "--source",
                type=str,
                choices=("audfprint", "shazam", "manual"),
                default=None,
                help="Filter by songs.source",
            )
        if name in {"timeline", "songs-timeline"}:
            r.add_argument("--limit", type=int, default=500)
        if name in {"dashboard", "artist-plays"}:
            default_out = "dashboard.html" if name == "dashboard" else "artist-plays.html"
            r.add_argument(
                "--out",
                type=Path,
                default=None,
                help=f"Output HTML path (default: data/reports/{default_out})",
            )

    # ---- seed
    seed = sub.add_parser("seed", help="Seeding toolchain (optional [seeding] extra)")
    seed_sub = seed.add_subparsers(dest="seed_command", required=True)
    seed_scrape = seed_sub.add_parser("scrape", help="Parse a station's recently-played page into a tracklist")
    seed_scrape.add_argument("--url", type=str, required=True)
    seed_scrape.add_argument("--row-selector", type=str, required=True)
    seed_scrape.add_argument("--artist-selector", type=str, required=True)
    seed_scrape.add_argument("--title-selector", type=str, required=True)

    seed_dl = seed_sub.add_parser("download", help="Download reference audio via yt-dlp")
    seed_dl.add_argument("--tracklist", type=Path, required=True, help="File with 'artist | title' per line")
    seed_dl.add_argument("--out", type=Path, default=None, help="Output dir (default: data/reference/songs/)")
    seed_dl.add_argument(
        "--audio-format",
        type=str,
        default="mp3",
        help=(
            "Audio format passed to yt-dlp. Default 'mp3' transcodes to a "
            "uniform codec/bitrate, which audfprint indexes more reliably "
            "than a mixed corpus of m4a/opus/webm originals. Pass 'best' "
            "to skip the transcode for speed (verify fingerprint recall "
            "before relying on it)."
        ),
    )
    seed_dl.add_argument(
        "--audio-quality",
        type=str,
        default="192",
        help="yt-dlp --audio-quality (bitrate in kbps for mp3/m4a/opus).",
    )
    seed_dl.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Parallel yt-dlp downloads (default: 4). Set to 1 for serial mode.",
    )

    # ---- songs (Shazam discovery workflow)
    songs = sub.add_parser(
        "songs",
        help="Inspect and act on songs the system has learned about",
    )
    songs_sub = songs.add_subparsers(dest="songs_command", required=True)

    songs_disc = songs_sub.add_parser(
        "discovered",
        help="List songs discovered via Shazam (source='shazam') with play stats",
    )
    songs_disc.add_argument("--db-path", type=Path, default=None)
    songs_disc.add_argument(
        "--since",
        type=str,
        default="24h",
        help="Play-count window (Nh / Nd / ISO-8601). Default: 24h",
    )
    songs_disc.add_argument(
        "--top",
        type=int,
        default=20,
        help="Max rows to print (default: 20)",
    )
    songs_disc.add_argument(
        "--min-plays",
        type=int,
        default=1,
        help="Only show discoveries heard at least this many times in --since (default: 1)",
    )
    songs_disc.add_argument(
        "--include-indexed",
        action="store_true",
        help="Also show discoveries already present in tracklist.txt",
    )
    songs_disc.add_argument(
        "--tracklist",
        type=Path,
        default=None,
        help="Tracklist file used for the 'tracklist' column (default: data/reference/tracklist.txt)",
    )

    songs_prom = songs_sub.add_parser(
        "promote",
        help="Append Shazam-discovered songs to tracklist.txt",
    )
    songs_prom.add_argument("--db-path", type=Path, default=None)
    songs_prom.add_argument(
        "--song-id",
        type=int,
        action="append",
        required=True,
        metavar="N",
        help="Repeat for each songs.id to promote (refuses non-Shazam ids)",
    )
    songs_prom.add_argument(
        "--tracklist",
        type=Path,
        default=None,
        help="Tracklist file to append to (default: data/reference/tracklist.txt)",
    )

    songs_dedupe = songs_sub.add_parser(
        "dedupe",
        help=(
            "Fold same-song rows that differ only by source (shazam vs audfprint) "
            "or by trivial casing/whitespace. Survivor keeps audfprint_track_id "
            "when any sibling has one."
        ),
    )
    songs_dedupe.add_argument("--db-path", type=Path, default=None)
    songs_enrich = songs_sub.add_parser(
        "enrich-releases",
        help="Fetch MusicBrainz release dates for songs (rate-limited)",
    )
    songs_enrich.add_argument("--db-path", type=Path, default=None)
    songs_enrich.add_argument("--since", type=str, default=None)
    songs_enrich.add_argument("--from", dest="from_utc", type=str, default=None)
    songs_enrich.add_argument("--to", dest="until_utc", type=str, default=None)
    songs_enrich.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max songs to query (default: all candidates)",
    )
    songs_enrich.add_argument(
        "--include-existing",
        action="store_true",
        help="Re-query even when release_date is already set",
    )
    songs_enrich.add_argument("--dry-run", action="store_true")

    songs_dedupe.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dedupe plan without modifying the DB.",
    )

    songs_stitch = songs_sub.add_parser(
        "stitch",
        help=(
            "Fold contiguous same-song SONG events split across capture-block "
            "boundaries into a single play (post-classification cleanup)."
        ),
    )
    songs_stitch.add_argument("--db-path", type=Path, default=None)
    songs_stitch.add_argument(
        "--since", type=str, default=None, help="Window start as duration/ISO (default: all)"
    )
    songs_stitch.add_argument(
        "--from", dest="from_utc", type=str, default=None, help="Explicit inclusive UTC window start"
    )
    songs_stitch.add_argument(
        "--to", dest="until_utc", type=str, default=None, help="Explicit exclusive UTC window end"
    )
    songs_stitch.add_argument(
        "--max-gap-seconds",
        type=float,
        default=2.0,
        help="Max gap between fragments to treat as contiguous (default: 2.0).",
    )
    songs_stitch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the stitch plan without modifying the DB (default).",
    )
    songs_stitch.add_argument(
        "--apply",
        action="store_true",
        help="Apply the stitch plan. Without this flag the command is read-only.",
    )

    # ---- commercials cleanup workflow
    commercials = sub.add_parser(
        "commercials",
        help="Inspect and clean up text-derived commercial identities",
    )
    commercials_sub = commercials.add_subparsers(dest="commercials_command", required=True)
    commercials_dedupe = commercials_sub.add_parser(
        "dedupe",
        help="Preview/fold duplicate commercial rows from brand variants or adjacent split windows.",
    )
    commercials_dedupe.add_argument("--db-path", type=Path, default=None)
    commercials_dedupe.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dedupe plan without modifying the DB (default).",
    )
    commercials_dedupe.add_argument(
        "--apply",
        action="store_true",
        help="Apply the dedupe plan. Without this flag the command is read-only.",
    )
    commercials_backfill = commercials_sub.add_parser(
        "backfill-brands",
        help="Recover brands for unbranded COMMERCIAL events from their transcripts.",
    )
    commercials_backfill.add_argument("--db-path", type=Path, default=None)
    commercials_backfill.add_argument(
        "--since", type=str, default=None, help="Window start as duration/ISO (default: all)"
    )
    commercials_backfill.add_argument(
        "--from", dest="from_utc", type=str, default=None, help="Explicit inclusive UTC window start"
    )
    commercials_backfill.add_argument(
        "--to", dest="until_utc", type=str, default=None, help="Explicit exclusive UTC window end"
    )
    commercials_backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview brand assignments without modifying the DB (default).",
    )
    commercials_backfill.add_argument(
        "--apply",
        action="store_true",
        help="Apply the backfill. Without this flag the command is read-only.",
    )
    commercials_backfill.add_argument(
        "--llm",
        action="store_true",
        help="Consult the local Ollama classifier for events the deterministic pass cannot brand.",
    )
    commercials_backfill.add_argument(
        "--ollama-model",
        type=str,
        default=None,
        help="Override the Ollama model for --llm (default: env / llama3.2).",
    )
    commercials_backfill.add_argument(
        "--limit", type=int, default=None, help="Cap the number of events processed."
    )
    commercials_merge = commercials_sub.add_parser(
        "merge-boundaries",
        help="Attribute unbranded commercial fragments to an adjacent branded ad (window-overlap orphans).",
    )
    commercials_merge.add_argument("--db-path", type=Path, default=None)
    commercials_merge.add_argument(
        "--since", type=str, default=None, help="Window start as duration/ISO (default: all)"
    )
    commercials_merge.add_argument(
        "--from", dest="from_utc", type=str, default=None, help="Explicit inclusive UTC window start"
    )
    commercials_merge.add_argument(
        "--to", dest="until_utc", type=str, default=None, help="Explicit exclusive UTC window end"
    )
    commercials_merge.add_argument(
        "--min-similarity",
        type=float,
        default=0.55,
        help="Minimum transcript similarity to a branded neighbor to merge (default: 0.55).",
    )
    commercials_merge.add_argument(
        "--max-gap-seconds",
        type=float,
        default=12.0,
        help="Maximum gap to the branded neighbor (default: 12s).",
    )
    commercials_merge.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview merges without modifying the DB (default).",
    )
    commercials_merge.add_argument(
        "--apply",
        action="store_true",
        help="Apply the merges. Without this flag the command is read-only.",
    )
    commercials_merge_brands = commercials_sub.add_parser(
        "merge-brands",
        help="Fold duplicate brand rows for the same advertiser (case/alias variants).",
    )
    commercials_merge_brands.add_argument("--db-path", type=Path, default=None)
    commercials_merge_brands.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the merge plan without modifying the DB (default).",
    )
    commercials_merge_brands.add_argument(
        "--apply",
        action="store_true",
        help="Apply the brand merge. Without this flag the command is read-only.",
    )

    return p


# ------------------------------------------------------------------ commands
def cmd_prereq(args: argparse.Namespace) -> int:
    from radio_classifier.prereq import run_checks

    results = run_checks(with_gpu=args.gpu, with_ollama=args.ollama or args.gpu)
    rc = 0
    for r in results:
        prefix = "ok  " if r.ok else "FAIL"
        line = f"[{prefix}] {r.name}"
        if r.detail:
            line += f" — {r.detail}"
        print(line, file=sys.stderr if not r.ok else sys.stdout)
        if not r.ok:
            rc = 1
    return rc


def cmd_db_init(args: argparse.Namespace) -> int:
    path = _resolve_db_path(args.db_path)
    with BroadcastStore(path):
        pass
    print(f"radio-classifier: initialized {path}", file=sys.stderr)
    return 0


def cmd_db_migrate(args: argparse.Namespace) -> int:
    from radio_classifier.persistence.migrate import migrate_from_live105sux

    report = migrate_from_live105sux(src_db=args.src, dst_db=args.dst)
    print(
        f"radio-classifier: migrated rows_read={report.rows_read} "
        f"inserted={report.rows_inserted} skipped={report.rows_skipped} "
        f"brands_created={report.brands_created}",
        file=sys.stderr,
    )
    return 0


def cmd_fp_index(args: argparse.Namespace) -> int:
    from radio_classifier.fingerprint import AudfprintIndex

    src_dir: Path = args.dir
    if not src_dir.is_dir():
        print(f"radio-classifier fingerprint index: not a directory: {src_dir}", file=sys.stderr)
        return 1
    out = args.out or _default_index_path()
    files = [p for p in sorted(src_dir.glob(args.glob)) if p.is_file()]
    if not files:
        print(f"radio-classifier fingerprint index: no files matching {args.glob} under {src_dir}", file=sys.stderr)
        return 1
    index = AudfprintIndex(index_path=out)
    will_extend = args.extend
    if will_extend and not out.exists():
        print(
            f"radio-classifier fingerprint index: --extend requested but index does not exist; "
            f"creating fresh index at {out}",
            file=sys.stderr,
        )
        will_extend = False
    if not will_extend and out.exists():
        out.unlink()
    index.build_or_extend(files, extend=will_extend)
    mode = "extended" if will_extend else "rebuilt"
    print(
        f"radio-classifier: {mode} index at {out} with {len(files)} files",
        file=sys.stderr,
    )
    return 0


def cmd_fp_eval(args: argparse.Namespace) -> int:
    from radio_classifier.fingerprint import AudfprintConfig, AudfprintIndex
    from radio_classifier.seeding.eval import evaluate, load_truth

    index_path = args.index or _default_index_path()
    index = AudfprintIndex(index_path=index_path, config=AudfprintConfig(min_count=args.audfprint_min_count))
    if not index.exists():
        print(f"radio-classifier fingerprint eval: index missing: {index_path}", file=sys.stderr)
        return 1
    truth = load_truth(args.truth)
    if not truth:
        print(f"radio-classifier fingerprint eval: empty truth CSV: {args.truth}", file=sys.stderr)
        return 1
    report = evaluate(index, truth)
    print(
        f"recall: {report.correct}/{report.total} = {report.recall:.1%}",
        file=sys.stderr,
    )
    for row in report.rows:
        ok = "ok " if row.correct else "MISS"
        matched = row.matched_track_id or "(no match)"
        print(
            f"  [{ok}] {row.clip_path.name}  truth={row.truth!r}  matched={matched!r}",
            file=sys.stderr,
        )
    return 0 if report.recall >= 0.9 else 2


def cmd_fp_explain(args: argparse.Namespace) -> int:
    from radio_classifier.fingerprint import AudfprintConfig, AudfprintIndex

    wav_path: Path = args.input
    if not wav_path.is_file():
        print(f"radio-classifier fingerprint explain: not a file: {wav_path}", file=sys.stderr)
        return 1
    index_path = args.index or _default_index_path()
    index = AudfprintIndex(index_path=index_path, config=AudfprintConfig())
    if not index.exists():
        print(
            f"radio-classifier fingerprint explain: index missing: {index_path}",
            file=sys.stderr,
        )
        return 1

    candidates = index.explain(
        wav_path,
        max_matches=max(1, int(args.max_matches)),
        min_count=max(1, int(args.min_count)),
    )
    print(f"clip:  {wav_path}")
    print(f"index: {index_path}")
    if not candidates:
        print("no candidates returned by audfprint (NOMATCH or empty output)")
        return 0

    print()
    print("rank  count  track")
    print("----  -----  -----")
    expected_lower = args.expected.lower() if args.expected else None
    expected_hits: list[tuple[int, int, str]] = []
    for rank, (track, count) in enumerate(candidates, start=1):
        marker = ""
        if expected_lower is not None and expected_lower in track.lower():
            marker = "  <-- expected"
            expected_hits.append((rank, count, track))
        print(f"{rank:>4}  {count:>5}  {track}{marker}")

    if expected_lower is not None:
        print()
        if expected_hits:
            best = expected_hits[0]
            print(
                f"expected match found: {best[2]!r} at rank {best[0]}, score {best[1]}"
            )
        else:
            print(
                f"expected {args.expected!r} NOT among the top {len(candidates)} "
                f"candidates (try a larger --max-matches or a lower --min-count)"
            )
    return 0


# -------------------------------------------------- funnel construction
def _build_funnel(args: argparse.Namespace) -> "FunnelBundle":
    from radio_classifier.commercials import CommercialIdentityResolver
    from radio_classifier.pipeline import FunnelOrchestrator

    tier1 = None
    if not args.no_tier1:
        from radio_classifier.fingerprint import AudfprintConfig, AudfprintIndex

        index_path = args.audfprint_index or _default_index_path()
        idx = AudfprintIndex(
            index_path=index_path,
            config=AudfprintConfig(min_count=args.audfprint_min_count),
        )
        if not idx.exists():
            print(
                f"radio-classifier: WARNING audfprint index missing at {index_path}; "
                "Tier 1 disabled for this run.",
                file=sys.stderr,
            )
            tier1 = None
        else:
            tier1 = idx

    # Tier 3 before Tier 2: faster-whisper must claim GPU VRAM first on 11 GB
    # cards. YAMNet will run on CPU when both tiers are active with CUDA Whisper.
    tier3 = None
    if not args.no_tier3:
        try:
            from radio_classifier.speech import OllamaSpeechClassifier, build_transcriber, run_speech_pipeline

            transcriber = build_transcriber(
                backend=args.whisper_backend,
                model_size=args.whisper_model,
                model=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                language=args.whisper_language,
                beam_size=args.whisper_beam_size,
                vad_filter=args.whisper_vad_filter,
            )
            llm = OllamaSpeechClassifier(
                base_url=args.ollama_base_url,
                model=args.ollama_model,
            )

            def _tier3(window):
                return run_speech_pipeline(
                    window,
                    transcriber=transcriber,
                    ollama_classifier=llm,
                    min_rms=args.speech_min_rms,
                )

            tier3 = _tier3
        except Exception as exc:  # noqa: BLE001
            print(
                f"radio-classifier: WARNING failed to load Tier 3 ({exc}); "
                "Tier 3 disabled.",
                file=sys.stderr,
            )

    tier2 = None
    if not args.no_tier2:
        try:
            from radio_classifier.acoustic import YamnetAcousticClassifier

            from radio_classifier.platform import is_macos

            force_yamnet_cpu = (tier3 is not None and args.whisper_device == "cuda") or (
                is_macos() and tier3 is not None
            )
            if force_yamnet_cpu:
                if is_macos():
                    print(
                        "radio-classifier: YAMNet on CPU (macOS standalone; "
                        "avoids TensorFlow Metal contention).",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "radio-classifier: YAMNet on CPU (Whisper uses CUDA; "
                        "keeps both tiers within GPU memory on 11 GB cards).",
                        file=sys.stderr,
                    )
            tier2 = YamnetAcousticClassifier(
                min_prob=args.tier2_min_prob,
                speech_bias=not args.no_tier2_speech_bias,
                force_cpu=force_yamnet_cpu,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"radio-classifier: WARNING failed to load YAMNet ({exc}); "
                "Tier 2 disabled for this run.",
                file=sys.stderr,
            )

    shazam_fn = None
    if args.enable_shazam:
        from radio_classifier.music import identify_window_sync

        def _shazam(window):
            return identify_window_sync(window)

        shazam_fn = _shazam

    store = None
    reducer = None
    if getattr(args, "persist", False):
        db_path = _resolve_db_path(args.db_path)
        store = BroadcastStore(db_path, use_wal=not getattr(args, "no_wal", False))
        reducer = SegmentReducer()

    resolver = (
        CommercialIdentityResolver(store=store) if store is not None else None
    )

    orchestrator = FunnelOrchestrator(
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
        resolver=resolver,
        store=store,
        shazam_fn=shazam_fn,
        window_seconds=args.window_seconds,
        shazam_recheck_windows=max(1, args.shazam_recheck_windows),
        unknown_music_rescue_speech_margin=max(
            0.0,
            args.unknown_music_rescue_speech_margin,
        ),
    )

    return FunnelBundle(
        orchestrator=orchestrator,
        store=store,
        reducer=reducer,
        capture_run_id=getattr(args, "capture_run_id", None),
    )


class FunnelBundle:
    def __init__(self, *, orchestrator, store, reducer, capture_run_id: int | None = None) -> None:
        self.orchestrator = orchestrator
        self.store = store
        self.reducer = reducer
        self.capture_run_id = capture_run_id

    def close(self, *, windows: list, window_seconds: float) -> None:
        try:
            if self.reducer is not None and self.store is not None and windows:
                persist_finalize(
                    self.reducer,
                    self.store,
                    last_window_start_utc=windows[-1].window_start_utc,
                    window_seconds=window_seconds,
                    capture_run_id=self.capture_run_id,
                )
        finally:
            if self.store is not None:
                self.store.close()


def _emit_funnel(args: argparse.Namespace, result, *, window_seconds: float) -> None:
    if getattr(args, "verbose", False):
        stage = result.stage.value
        details = []
        if result.fingerprint is not None:
            details.append(f"fp={result.fingerprint.status.value}")
            if result.fingerprint.track_id:
                details.append(f"track={result.fingerprint.track_id!r}")
            if result.fingerprint.match_score is not None:
                details.append(f"count={int(result.fingerprint.match_score)}")
            if result.fingerprint.message:
                fp_msg = result.fingerprint.message.replace("\n", " ").strip()
                if len(fp_msg) > 200:
                    fp_msg = fp_msg[:200] + "..."
                details.append(f"fp_msg={fp_msg!r}")
        if result.acoustic is not None:
            details.append(
                f"ac={result.acoustic.label.value} "
                f"m={result.acoustic.music_prob:.2f} s={result.acoustic.speech_prob:.2f}"
            )
        if result.shazam is not None:
            details.append(f"shazam={result.shazam.status.value}")
            if result.shazam.artist:
                details.append(f"sz_artist={result.shazam.artist!r}")
            if result.shazam.title:
                details.append(f"sz_title={result.shazam.title!r}")
            if result.shazam.confidence is not None:
                details.append(f"sz_conf={result.shazam.confidence:.2f}")
            if result.shazam.message:
                sz_msg = result.shazam.message.replace("\n", " ").strip()
                if len(sz_msg) > 200:
                    sz_msg = sz_msg[:200] + "..."
                details.append(f"sz_msg={sz_msg!r}")
        if result.speech is not None:
            transcript = result.speech.transcript or ""
            details.append(f"words={len(transcript.split())}")
            if result.speech.confidence is not None:
                details.append(f"conf={result.speech.confidence:.2f}")
            if result.speech.category is not None:
                details.append(f"llm={result.speech.category.value}")
                if result.speech.brand:
                    details.append(f"brand={result.speech.brand!r}")
            if transcript:
                excerpt = transcript.replace("\n", " ").strip()
                if len(excerpt) > 80:
                    excerpt = excerpt[:80] + "..."
                details.append(f"tx={excerpt!r}")
            if result.speech.category is None:
                details.append(f"sp_status={result.speech.status.value}")
                if result.speech.message:
                    msg = result.speech.message.replace("\n", " ").strip()
                    if len(msg) > 200:
                        msg = msg[:200] + "..."
                    details.append(f"sp_msg={msg!r}")
        if result.commercial_resolution is not None:
            details.append(f"commercial_id={result.commercial_resolution.commercial_id}")
        print(
            f"window start_utc={result.window_start_utc} stage={stage} "
            + " ".join(details),
            file=sys.stderr,
        )
    if args.json_lines:
        payload = {
            "window_start_utc": result.window_start_utc,
            "stage": result.stage.value,
            "fingerprint": _fp_to_dict(result.fingerprint),
            "acoustic": _ac_to_dict(result.acoustic),
            "speech": _sp_to_dict(result.speech),
            "shazam": _sz_to_dict(result.shazam),
            "commercial_resolution": _cr_to_dict(result.commercial_resolution),
        }
        print(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


def _fp_to_dict(fp):
    if fp is None:
        return None
    return {
        "status": fp.status.value,
        "track_id": fp.track_id,
        "artist": fp.artist,
        "title": fp.title,
        "match_score": fp.match_score,
        "message": fp.message,
    }


def _ac_to_dict(ac):
    if ac is None:
        return None
    return {
        "label": ac.label.value,
        "music_prob": ac.music_prob,
        "speech_prob": ac.speech_prob,
        "other_prob": ac.other_prob,
        "top": [{"name": n, "prob": p} for n, p in ac.top_classes],
    }


def _sp_to_dict(sp):
    if sp is None:
        return None
    return {
        "status": sp.status.value,
        "category": sp.category.value if sp.category else None,
        "brand": sp.brand,
        "brand_mentions": [
            {"name": m.name, "type": m.mention_type} for m in (sp.brand_mentions or [])
        ],
        "confidence": sp.confidence,
        "rationale": sp.rationale,
        "transcript": sp.transcript,
        "message": sp.message,
    }


def _sz_to_dict(sz):
    if sz is None:
        return None
    return {
        "status": sz.status.value,
        "artist": sz.artist,
        "title": sz.title,
        "confidence": sz.confidence,
        "message": sz.message,
    }


def _cr_to_dict(cr):
    if cr is None:
        return None
    return {
        "commercial_id": cr.commercial_id,
        "brand_id": cr.brand_id,
        "duration_bucket_seconds": cr.duration_bucket_seconds,
        "was_new": cr.was_new,
        "reason": cr.reason,
    }


def _persist_brand_mentions(
    store,
    *,
    event_id: int,
    mentions,
    heard_utc: str,
) -> None:
    from radio_classifier.brands import canonicalize_brand

    if store is None or event_id <= 0 or not mentions:
        return
    for m in mentions:
        brand_name = canonicalize_brand(m.name)
        if brand_name is None:
            continue
        brand_id = store.upsert_brand(brand_name)
        store.insert_brand_mention(
            segment_id=event_id,
            brand_id=brand_id,
            mention_type=m.mention_type,
            heard_utc=heard_utc,
        )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "?"
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes:d}m{sec:02d}s"
    return f"{sec:d}s"


class _ProgressReporter:
    """Small dependency-free progress reporter for offline classify."""

    def __init__(self, *, total: int, enabled: bool, verbose: bool) -> None:
        self.total = max(1, total)
        self.enabled = enabled
        self.verbose = verbose
        self.started = time.monotonic()
        self.counts: dict[str, int] = {
            "SONG": 0,
            "COMMERCIAL": 0,
            "DJ": 0,
            "STATION": 0,
            "PSA_NEWS": 0,
            "UNKNOWN": 0,
        }
        self._tty = sys.stderr.isatty() and not verbose
        self._last_len = 0
        # In non-TTY logs (Cursor terminals, redirected stderr), print at 5%
        # increments so progress is visible without one extra line per window.
        self._line_every = max(1, total // 20)

    def observe(self, idx: int, result) -> None:
        if not self.enabled:
            return
        category = None
        if result.segment_input is not None:
            category = result.segment_input.key.category.value
        self.counts[category or "UNKNOWN"] = self.counts.get(category or "UNKNOWN", 0) + 1
        if self._tty:
            self._write_tty(idx, result.stage.value)
            return
        if idx == 1 or idx == self.total or idx % self._line_every == 0:
            print(f"progress {self._line(idx, result.stage.value)}", file=sys.stderr)

    def finish(self) -> None:
        if self.enabled and self._tty:
            print(file=sys.stderr)

    def _line(self, idx: int, stage: str) -> str:
        elapsed = max(0.001, time.monotonic() - self.started)
        pct = min(100.0, idx / self.total * 100.0)
        rate = idx / elapsed
        remaining = (self.total - idx) / rate if rate > 0 else None
        categories = (
            f"song={self.counts['SONG']} "
            f"ad={self.counts['COMMERCIAL']} "
            f"dj={self.counts['DJ']} "
            f"station={self.counts['STATION']} "
            f"psa={self.counts['PSA_NEWS']} "
            f"unknown={self.counts['UNKNOWN']}"
        )
        return (
            f"{idx}/{self.total} {pct:5.1f}% "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)} "
            f"last={stage} {categories}"
        )

    def _write_tty(self, idx: int, stage: str) -> None:
        text = f"\rclassify {self._line(idx, stage)}"
        padding = " " * max(0, self._last_len - len(text))
        print(text + padding, end="", file=sys.stderr, flush=True)
        self._last_len = len(text)


def _progress_enabled(args: argparse.Namespace) -> bool:
    if args.progress is not None:
        return bool(args.progress)
    # Default to progress for interactive non-verbose runs. Verbose already
    # emits one line per window, and explicit --progress can still combine both.
    return sys.stderr.isatty() and not args.verbose


class _CachedTier1:
    """Tier-1 adapter backed by precomputed batch audfprint results."""

    def __init__(self, results) -> None:
        self._by_start = {r.window_start_utc: r for r in results}

    def match_window(self, window):
        result = self._by_start.get(window.window_start_utc)
        if result is None:
            from radio_classifier.fingerprint import FingerprintResult, FingerprintStatus

            return FingerprintResult(
                status=FingerprintStatus.skipped,
                window_start_utc=window.window_start_utc,
                message="batch Tier 1 result missing for window",
            )
        return replace(result, window_start_utc=window.window_start_utc)


def _maybe_precompute_tier1(
    args: argparse.Namespace,
    bundle: FunnelBundle,
    windows: list,
    *,
    progress_enabled: bool,
) -> None:
    if args.no_batch_tier1:
        return
    tier1 = getattr(bundle.orchestrator, "tier1", None)
    if tier1 is None or not hasattr(tier1, "match_windows"):
        return
    if progress_enabled:
        print(
            f"progress tier1_batch start windows={len(windows)} "
            "(one audfprint index load per batch)",
            file=sys.stderr,
        )
    started = time.monotonic()
    results = tier1.match_windows(windows)
    bundle.orchestrator.tier1 = _CachedTier1(results)
    if progress_enabled:
        matches = sum(1 for r in results if r.status.value == "match")
        errors = sum(1 for r in results if r.status.value == "error")
        print(
            "progress tier1_batch done "
            f"elapsed={_format_duration(time.monotonic() - started)} "
            f"matches={matches} errors={errors}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------- classify
def cmd_classify(args: argparse.Namespace) -> int:
    path: Path = args.input
    if not path.is_file():
        print(f"radio-classifier classify: not a file: {path}", file=sys.stderr)
        return 1
    try:
        pcm, rate = read_mono_s16le_wav(path)
    except (ValueError, OSError) as e:
        print(f"radio-classifier classify: {e}", file=sys.stderr)
        return 1

    effective_rate = args.sample_rate_override or rate
    clock_start_ns = (
        _parse_utc_time_ns(args.capture_start_utc)
        if args.capture_start_utc
        else time.time_ns()
    )
    windows = list(
        iter_overlapping_windows(
            pcm,
            effective_rate,
            window_seconds=args.window_seconds,
            overlap_fraction=args.overlap_fraction,
            clock_start_ns=clock_start_ns,
        )
    )
    if not windows:
        print("radio-classifier classify: no full windows", file=sys.stderr)
        return 0

    progress_enabled = _progress_enabled(args)
    bundle = _build_funnel(args)
    try:
        _maybe_precompute_tier1(
            args,
            bundle,
            windows,
            progress_enabled=progress_enabled,
        )
        progress = _ProgressReporter(
            total=len(windows),
            enabled=progress_enabled,
            verbose=args.verbose,
        )
        for idx, w in enumerate(windows, start=1):
            r = bundle.orchestrator.process(w)
            _emit_funnel(args, r, window_seconds=args.window_seconds)
            progress.observe(idx, r)
            if bundle.reducer is not None:
                new_ids = persist_input(
                    bundle.reducer,
                    bundle.store,
                    r.segment_input,
                    capture_run_id=bundle.capture_run_id,
                )
                if new_ids:
                    # mentions belong to the window that closed the *previous* segment
                    _persist_brand_mentions(
                        bundle.store,
                        event_id=new_ids[-1],
                        mentions=r.brand_mentions or [],
                        heard_utc=r.window_start_utc,
                    )
    finally:
        if "progress" in locals():
            progress.finish()
        bundle.close(windows=windows, window_seconds=args.window_seconds)
    return 0


# ------------------------------------------------------------------ ingest
def cmd_ingest(args: argparse.Namespace) -> int:
    stream = RtlFmStream(
        frequency_hz=args.frequency,
        device_index=args.device_index,
        sample_rate_hz=args.sample_rate,
    )
    stream.start()
    buf = bytearray()
    try:
        for chunk in stream.iter_stdout_bytes(max_wall_seconds=args.duration_limit):
            buf.extend(chunk)
    except RtlFmExitedError as e:
        print(f"radio-classifier ingest: rtl_fm failed: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"radio-classifier ingest: I/O error: {e}", file=sys.stderr)
        return 1

    if len(buf) % 2 != 0:
        buf = buf[:-1]
    if not buf:
        print("radio-classifier ingest: no PCM bytes captured", file=sys.stderr)
        return 1

    pcm = np.frombuffer(buf, dtype="<i2")
    clock_start_ns = time.time_ns()

    if args.capture_wav is not None:
        capture_path = _resolve_capture_wav_path(args.capture_wav, clock_start_ns)
        try:
            write_mono_s16le_wav(capture_path, pcm, args.sample_rate)
        except OSError as e:
            print(f"radio-classifier ingest: capture WAV write failed: {e}", file=sys.stderr)
            return 1
        capture_start_iso = (
            datetime.fromtimestamp(clock_start_ns / 1e9, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        print(
            f"radio-classifier ingest: wrote capture to {capture_path} "
            f"(starts at {capture_start_iso}; subtract this from any window "
            f"start_utc to seek inside the file)",
            file=sys.stderr,
        )

    windows = list(
        iter_overlapping_windows(
            pcm,
            args.sample_rate,
            window_seconds=args.window_seconds,
            overlap_fraction=args.overlap_fraction,
            clock_start_ns=clock_start_ns,
        )
    )
    if args.verbose:
        print(
            f"radio-classifier ingest: samples={pcm.shape[0]} rate={args.sample_rate} "
            f"windows={len(windows)}",
            file=sys.stderr,
        )
    if not windows:
        return 0

    bundle = _build_funnel(args)
    try:
        for w in windows:
            r = bundle.orchestrator.process(w)
            _emit_funnel(args, r, window_seconds=args.window_seconds)
            if bundle.reducer is not None:
                new_ids = persist_input(
                    bundle.reducer,
                    bundle.store,
                    r.segment_input,
                    capture_run_id=bundle.capture_run_id,
                )
                if new_ids:
                    _persist_brand_mentions(
                        bundle.store,
                        event_id=new_ids[-1],
                        mentions=r.brand_mentions or [],
                        heard_utc=r.window_start_utc,
                    )
    finally:
        bundle.close(windows=windows, window_seconds=args.window_seconds)
    return 0


def _chunk_base_name(run_id: str, index: int) -> str:
    return f"{run_id}_block{index:04d}"


def _write_chunk_sidecar(
    *,
    sidecar_path: Path,
    wav_path: Path,
    run_id: str,
    block_index: int,
    sample_rate_hz: int,
    start_ns: int,
    samples_written: int,
    complete: bool,
) -> None:
    duration_seconds = samples_written / sample_rate_hz if sample_rate_hz else 0.0
    end_ns = start_ns + int(duration_seconds * 1_000_000_000)
    payload = {
        "run_id": run_id,
        "block_index": block_index,
        "wav_path": str(wav_path),
        "capture_start_utc": _iso_from_time_ns(start_ns),
        "capture_end_utc": _iso_from_time_ns(end_ns),
        "sample_rate_hz": sample_rate_hz,
        "channels": 1,
        "sample_width_bytes": 2,
        "samples": samples_written,
        "duration_seconds": duration_seconds,
        "complete": complete,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_capture_chunks(args: argparse.Namespace) -> int:
    """Capture one uninterrupted RTL-SDR stream into fixed-size WAV chunks.

    This intentionally does **no** classification. It keeps RF capture cheap
    and continuous while a separate bounded worker classifies completed chunks
    as GPU/LLM resources allow.
    """

    if args.chunk_seconds <= 0:
        print("radio-classifier capture chunks: --chunk-seconds must be > 0", file=sys.stderr)
        return 1
    if args.duration_limit is not None and args.duration_limit <= 0:
        print("radio-classifier capture chunks: --duration-limit must be > 0", file=sys.stderr)
        return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(tz=timezone.utc).strftime("capture_%Y%m%dT%H%M%SZ")
    sample_rate = int(args.sample_rate)
    bytes_per_sample = 2
    target_samples = max(1, int(round(float(args.chunk_seconds) * sample_rate)))
    target_bytes = target_samples * bytes_per_sample

    stream = RtlFmStream(
        frequency_hz=args.frequency,
        device_index=args.device_index,
        sample_rate_hz=sample_rate,
    )
    stream.start()
    capture_start_ns = time.time_ns()
    block_index = 1
    current_bytes = 0
    current_samples = 0
    block_start_ns = capture_start_ns
    wav_path: Path | None = None
    sidecar_path: Path | None = None
    wav_file: wave.Wave_write | None = None

    def _open_next_chunk() -> None:
        nonlocal wav_path, sidecar_path, wav_file, current_bytes, current_samples, block_start_ns
        base = _chunk_base_name(run_id, block_index)
        wav_path = out_dir / f"{base}.wav"
        sidecar_path = out_dir / f"{base}.json"
        block_start_ns = capture_start_ns + int((block_index - 1) * target_samples / sample_rate * 1_000_000_000)
        wav_file = wave.open(str(wav_path), "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bytes_per_sample)
        wav_file.setframerate(sample_rate)
        current_bytes = 0
        current_samples = 0

    def _close_current_chunk(*, complete: bool) -> None:
        nonlocal wav_file
        if wav_file is None or wav_path is None or sidecar_path is None:
            return
        wav_file.close()
        wav_file = None
        _write_chunk_sidecar(
            sidecar_path=sidecar_path,
            wav_path=wav_path,
            run_id=run_id,
            block_index=block_index,
            sample_rate_hz=sample_rate,
            start_ns=block_start_ns,
            samples_written=current_samples,
            complete=complete,
        )
        status = "complete" if complete else "partial"
        print(
            f"radio-classifier capture chunks: wrote {status} block {block_index} "
            f"wav={wav_path} sidecar={sidecar_path}",
            file=sys.stderr,
        )

    _open_next_chunk()
    carry = b""
    try:
        for raw in stream.iter_stdout_bytes(max_wall_seconds=args.duration_limit):
            data = carry + raw
            if len(data) % bytes_per_sample:
                carry = data[-1:]
                data = data[:-1]
            else:
                carry = b""
            offset = 0
            while offset < len(data):
                assert wav_file is not None
                available = target_bytes - current_bytes
                piece = data[offset : offset + available]
                wav_file.writeframesraw(piece)
                wrote = len(piece)
                current_bytes += wrote
                current_samples += wrote // bytes_per_sample
                offset += wrote
                if current_bytes >= target_bytes:
                    _close_current_chunk(complete=True)
                    block_index += 1
                    _open_next_chunk()
    except RtlFmExitedError as e:
        print(f"radio-classifier capture chunks: rtl_fm failed: {e}", file=sys.stderr)
        if current_samples > 0:
            _close_current_chunk(complete=False)
        return 1
    except OSError as e:
        print(f"radio-classifier capture chunks: I/O error: {e}", file=sys.stderr)
        if current_samples > 0:
            _close_current_chunk(complete=False)
        return 1
    finally:
        if wav_file is not None:
            if current_samples > 0:
                _close_current_chunk(complete=False)
            else:
                wav_file.close()
                wav_file = None

    # If the capture ended exactly on a chunk boundary, ``_open_next_chunk``
    # has created an empty placeholder for the next block. Remove it rather
    # than advertising a zero-length partial chunk.
    if current_samples == 0 and wav_path is not None and wav_path.exists():
        wav_path.unlink()
    if current_samples == 0 and sidecar_path is not None and sidecar_path.exists():
        sidecar_path.unlink()
    return 0


# --------------------------------------------------------------------- runs
def cmd_runs(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args.db_path)
    if args.runs_command == "start":
        started_utc = args.started_utc or _iso_from_time_ns(time.time_ns())
        version = args.pipeline_version or pipeline_version()
        host = args.host or socket.gethostname()
        with BroadcastStore(db_path) as store:
            capture_run_id = store.open_capture_run(
                run_id=args.run_id,
                started_utc=started_utc,
                pipeline_version=version,
                host=host,
                notes=args.notes,
            )
        print(capture_run_id)
        return 0

    if args.runs_command == "end":
        ended_utc = args.ended_utc or _iso_from_time_ns(time.time_ns())
        with BroadcastStore(db_path) as store:
            store.close_capture_run(
                run_id=args.run_id,
                ended_utc=ended_utc,
                notes=args.notes,
            )
        print(f"radio-classifier: closed capture run {args.run_id}", file=sys.stderr)
        return 0

    if args.runs_command == "list":
        if not db_path.exists():
            print(f"radio-classifier runs list: db not found: {db_path}", file=sys.stderr)
            return 1
        with BroadcastStore(db_path) as store:
            rows = store.connection.execute(
                """
                SELECT
                    cr.id,
                    cr.run_id,
                    cr.started_utc,
                    cr.ended_utc,
                    cr.pipeline_version,
                    COUNT(be.id) AS event_count
                FROM capture_runs cr
                LEFT JOIN broadcast_events be ON be.capture_run_id = cr.id
                GROUP BY cr.id
                ORDER BY cr.started_utc DESC, cr.id DESC
                LIMIT ?
                """,
                (max(1, int(args.limit)),),
            ).fetchall()
        if not rows:
            print("(no rows)")
            return 0
        print("id  run_id                         started_utc               ended_utc                 events  pipeline")
        print("--  -----------------------------  ------------------------  ------------------------  ------  --------")
        for row in rows:
            print(
                f"{int(row[0]):<3} "
                f"{str(row[1])[:29]:<29}  "
                f"{str(row[2] or ''):<24}  "
                f"{str(row[3] or ''):<24}  "
                f"{int(row[5]):<6}  "
                f"{row[4]}"
            )
        return 0

    print("radio-classifier runs: unknown subcommand", file=sys.stderr)
    return 2


# ------------------------------------------------------------------ report
def _report_window(args: argparse.Namespace, parse_since_fn) -> tuple[str, str | None]:
    if (args.from_utc or args.until_utc) and args.since:
        raise _CliConfigError("report window accepts either --since or --from/--to, not both")
    if args.from_utc:
        since_utc = _normalize_utc_text(args.from_utc)
    else:
        since_utc = parse_since_fn(args.since or "24h")
    until_utc = _normalize_utc_text(args.until_utc) if args.until_utc else None
    if until_utc is not None and until_utc <= since_utc:
        raise _CliConfigError("--to must be later than the report window start")
    return since_utc, until_utc


def cmd_report(args: argparse.Namespace) -> int:
    from radio_classifier.reports import (
        artists_top,
        brands_top,
        commercials_top,
        format_artists,
        format_brands,
        format_commercials,
        format_runs,
        format_songs,
        format_songs_added,
        format_songs_timeline,
        format_summary,
        format_timeline,
        parse_since,
        runs_summary,
        songs_added,
        songs_timeline,
        songs_top,
        summary,
        timeline,
        write_artist_plays,
        write_dashboard,
    )

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier report: db not found: {db_path}", file=sys.stderr)
        return 1
    since_utc, until_utc = _report_window(args, parse_since)
    with BroadcastStore(db_path) as store:
        if args.report_command == "commercials":
            rows = commercials_top(
                store,
                since_utc=since_utc,
                until_utc=until_utc,
                top_n=args.top,
                brand=getattr(args, "brand", None),
            )
            print(format_commercials(rows))
        elif args.report_command == "brands":
            rows = brands_top(store, since_utc=since_utc, until_utc=until_utc, top_n=args.top)
            print(format_brands(rows))
        elif args.report_command == "songs":
            rows = songs_top(store, since_utc=since_utc, until_utc=until_utc, top_n=args.top)
            print(format_songs(rows))
        elif args.report_command == "songs-added":
            rows = songs_added(
                store,
                since_utc=since_utc,
                until_utc=until_utc,
                source=args.source,
                top_n=args.top,
            )
            print(format_songs_added(rows))
        elif args.report_command == "songs-timeline":
            rows = songs_timeline(store, since_utc=since_utc, until_utc=until_utc, limit=args.limit)
            print(format_songs_timeline(rows))
        elif args.report_command == "artists":
            rows = artists_top(store, since_utc=since_utc, until_utc=until_utc, top_n=args.top)
            print(format_artists(rows))
        elif args.report_command == "artist-plays":
            out_path = args.out or (_project_root() / "data" / "reports" / "artist-plays.html")
            written = write_artist_plays(
                store,
                since_utc=since_utc,
                until_utc=until_utc,
                out_path=out_path,
                top_n=args.top,
            )
            print(f"radio-classifier: wrote artist play log to {written}")
        elif args.report_command == "timeline":
            rows = timeline(store, since_utc=since_utc, until_utc=until_utc, limit=args.limit)
            print(format_timeline(rows))
        elif args.report_command == "summary":
            rows = summary(store, since_utc=since_utc, until_utc=until_utc)
            print(format_summary(rows))
        elif args.report_command == "runs":
            rows = runs_summary(store, since_utc=since_utc, until_utc=until_utc, top_n=args.top)
            print(format_runs(rows))
        elif args.report_command == "dashboard":
            out_path = args.out or (_project_root() / "data" / "reports" / "dashboard.html")
            written = write_dashboard(
                store,
                since_utc=since_utc,
                until_utc=until_utc,
                out_path=out_path,
                top_n=args.top,
            )
            print(f"radio-classifier: wrote dashboard to {written}")
        else:
            print(f"radio-classifier report: unknown subcommand", file=sys.stderr)
            return 2
    return 0


# ------------------------------------------------------------------- seed
def cmd_seed_scrape(args: argparse.Namespace) -> int:
    try:
        from radio_classifier.seeding.scrape import fetch_html, parse_tracklist
    except ImportError as exc:
        print(
            f"radio-classifier seed scrape: install the [seeding] extra ({exc})",
            file=sys.stderr,
        )
        return 1
    html = fetch_html(args.url)
    tracks = parse_tracklist(
        html,
        row_selector=args.row_selector,
        artist_selector=args.artist_selector,
        title_selector=args.title_selector,
    )
    for t in tracks:
        print(f"{t.artist} | {t.title}")
    print(f"radio-classifier: {len(tracks)} tracks parsed", file=sys.stderr)
    return 0


def cmd_seed_download(args: argparse.Namespace) -> int:
    try:
        from radio_classifier.seeding.download import DownloadConfig, download_tracks
        from radio_classifier.seeding.scrape import Track, dedupe_tracks
    except ImportError as exc:
        print(
            f"radio-classifier seed download: install the [seeding] extra ({exc})",
            file=sys.stderr,
        )
        return 1
    if not args.tracklist.exists():
        print(f"radio-classifier seed download: missing tracklist: {args.tracklist}", file=sys.stderr)
        return 1
    out = args.out or (_project_root() / "data" / "reference" / "songs")
    raw_tracks: list[Track] = []
    for raw in args.tracklist.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        artist, _, title = line.partition("|")
        raw_tracks.append(Track(artist=artist.strip(), title=title.strip()))
    if not raw_tracks:
        print("radio-classifier seed download: no tracks parsed from tracklist", file=sys.stderr)
        return 1
    tracks = dedupe_tracks(raw_tracks)
    skipped = len(raw_tracks) - len(tracks)
    if skipped:
        print(
            f"radio-classifier: deduped tracklist {len(raw_tracks)} -> {len(tracks)} "
            f"(skipped {skipped} duplicate{'s' if skipped != 1 else ''})",
            file=sys.stderr,
        )
    cfg = DownloadConfig(
        output_dir=out,
        audio_format=args.audio_format,
        audio_quality=args.audio_quality,
        concurrency=args.concurrency,
    )
    results = download_tracks(tracks, cfg)
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    downloaded = sum(1 for r in results if r.ok and not r.skipped)
    failed = sum(1 for r in results if not r.ok)
    print(
        f"radio-classifier: downloaded={downloaded} skipped={skipped} "
        f"failed={failed} total={total}",
        file=sys.stderr,
    )
    for r in results:
        if not r.ok:
            print(f"  FAIL {r.track.artist} - {r.track.title}: {r.message}", file=sys.stderr)
    return 0 if (downloaded + skipped) > 0 else 1


# ------------------------------------------------------------------- songs
def cmd_songs_discovered(args: argparse.Namespace) -> int:
    from radio_classifier.discovery import list_shazam_discoveries
    from radio_classifier.reports import format_discoveries, parse_since

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier songs discovered: db not found: {db_path}", file=sys.stderr)
        return 1
    tracklist = args.tracklist or _default_tracklist_path()
    since_utc = parse_since(args.since)
    with BroadcastStore(db_path) as store:
        rows = list_shazam_discoveries(
            store,
            since_utc=since_utc,
            top_n=args.top,
            min_plays=args.min_plays,
            include_indexed=args.include_indexed,
            tracklist_path=tracklist,
        )
    print(format_discoveries(rows))

    if not rows:
        return 0
    missing_ids = [r.song_id for r in rows if not r.in_tracklist]
    review_ids = [r.song_id for r in rows if r.needs_review]
    print(
        f"\n{len(rows)} Shazam discoveries "
        f"({len(missing_ids)} not yet indexed in {tracklist}).",
        file=sys.stderr,
    )
    if review_ids:
        ids = ", ".join(str(sid) for sid in review_ids)
        print(
            f"{len(review_ids)} low-confidence discovery row(s) flagged for manual review "
            f"(plays < 3): {ids}",
            file=sys.stderr,
        )
    if missing_ids:
        ids = " ".join(f"--song-id {sid}" for sid in missing_ids)
        print(
            "\nTo add the missing ones to the local fingerprint index, run:\n"
            f"  radio-classifier songs promote {ids}\n"
            f"  radio-classifier seed download --tracklist {tracklist} "
            "--out data/reference/songs/\n"
            "  radio-classifier fingerprint index --dir data/reference/songs/",
            file=sys.stderr,
        )
    return 0


def cmd_songs_promote(args: argparse.Namespace) -> int:
    from radio_classifier.discovery import promote_to_tracklist

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier songs promote: db not found: {db_path}", file=sys.stderr)
        return 1
    tracklist = args.tracklist or _default_tracklist_path()
    if not tracklist.exists():
        print(
            f"radio-classifier songs promote: tracklist not found: {tracklist}",
            file=sys.stderr,
        )
        return 1

    with BroadcastStore(db_path) as store:
        result = promote_to_tracklist(
            store,
            song_ids=list(args.song_id),
            tracklist_path=tracklist,
        )

    appended = [p for p in result.promoted if p.appended]
    skipped = [p for p in result.promoted if not p.appended]

    if appended:
        print(
            f"radio-classifier: appended {len(appended)} track(s) to {tracklist}:",
            file=sys.stderr,
        )
        for p in appended:
            print(f"  {p.artist} | {p.title}", file=sys.stderr)
    if skipped:
        print(
            f"radio-classifier: skipped {len(skipped)} song-id(s):",
            file=sys.stderr,
        )
        for p in skipped:
            print(f"  song_id={p.song_id}: {p.reason}", file=sys.stderr)
    if appended:
        print(
            "\nNext steps (run when ready to populate the audfprint index):\n"
            f"  radio-classifier seed download --tracklist {tracklist} "
            "--out data/reference/songs/\n"
            "  radio-classifier fingerprint index --dir data/reference/songs/",
            file=sys.stderr,
        )
    return 0 if appended or not skipped else 1


def cmd_songs_enrich_releases(args: argparse.Namespace) -> int:
    from radio_classifier.discovery.releases import enrich_song_releases
    from radio_classifier.reports.queries import parse_since

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier songs enrich-releases: db not found: {db_path}", file=sys.stderr)
        return 1

    since_utc: str | None = None
    until_utc: str | None = None
    if args.from_utc or args.until_utc or args.since:
        since_utc, until_utc = _report_window(args, parse_since)

    with BroadcastStore(db_path) as store:
        report = enrich_song_releases(
            store,
            since_utc=since_utc,
            until_utc=until_utc,
            only_missing=not args.include_existing,
            limit=args.limit,
            dry_run=args.dry_run,
        )

    verb = "would update" if args.dry_run else "updated"
    print(
        f"radio-classifier songs enrich-releases: examined={report.examined} "
        f"{verb}={report.updated} not_found={report.not_found} "
        f"errors={report.errors} skipped_existing={report.skipped_existing}",
        file=sys.stderr,
    )
    return 0 if report.errors == 0 else 1


def cmd_songs_dedupe(args: argparse.Namespace) -> int:
    from radio_classifier.discovery import dedupe_songs

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier songs dedupe: db not found: {db_path}", file=sys.stderr)
        return 1

    with BroadcastStore(db_path) as store:
        report = dedupe_songs(store, dry_run=args.dry_run)

    if not report.groups:
        print("radio-classifier: no duplicate song groups found", file=sys.stderr)
        return 0

    verb = "would fold" if report.dry_run else "folded"
    print(
        f"radio-classifier: {verb} {report.collapsed_pairs} duplicate row(s) "
        f"across {len(report.groups)} group(s)",
        file=sys.stderr,
    )
    for group in report.groups:
        survivor = group.survivor
        survivor_note = "audfprint" if survivor.audfprint_track_id else survivor.source
        print(
            f"  {group.key[0]!r} / {group.key[1]!r} → keep song_id={survivor.song_id} "
            f"({survivor_note})",
            file=sys.stderr,
        )
        for loser in group.losers:
            loser_note = "audfprint" if loser.audfprint_track_id else loser.source
            print(
                f"    drop song_id={loser.song_id} ({loser_note}, "
                f"{loser.event_count} event(s))",
                file=sys.stderr,
            )

    if not report.dry_run:
        print(
            f"\nradio-classifier: re-pointed {report.events_repointed} "
            f"broadcast_events row(s); deleted {report.rows_deleted} song row(s)",
            file=sys.stderr,
        )
    return 0


def cmd_songs_stitch(args: argparse.Namespace) -> int:
    from radio_classifier.discovery import stitch_song_plays
    from radio_classifier.reports import parse_since

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier songs stitch: db not found: {db_path}", file=sys.stderr)
        return 1

    if (args.from_utc or args.until_utc) and args.since:
        raise _CliConfigError("stitch window accepts either --since or --from/--to, not both")
    if args.from_utc:
        since_utc: str | None = _normalize_utc_text(args.from_utc)
    elif args.since:
        since_utc = parse_since(args.since)
    else:
        since_utc = None
    until_utc = _normalize_utc_text(args.until_utc) if args.until_utc else None

    dry_run = True if args.dry_run else not args.apply
    with BroadcastStore(db_path) as store:
        report = stitch_song_plays(
            store,
            since_utc=since_utc,
            until_utc=until_utc,
            max_gap_seconds=args.max_gap_seconds,
            dry_run=dry_run,
        )

    verb = "would stitch" if report.dry_run else "stitched"
    print(
        f"radio-classifier: {verb} {report.events_absorbed} fragment(s) into "
        f"{len(report.groups)} song play(s) (scanned {report.events_scanned} SONG event(s))",
        file=sys.stderr,
    )
    for g in report.groups[:25]:
        cross = " [cross-run]" if g.spanned_capture_runs else ""
        print(
            f"  song_id={g.song_id} {g.artist} - {g.title}: "
            f"keep event {g.survivor_event_id}, absorb {len(g.absorbed_event_ids)} "
            f"[{g.start_utc} -> {g.end_utc}]{cross}",
            file=sys.stderr,
        )
    if len(report.groups) > 25:
        print(f"  ... and {len(report.groups) - 25} more", file=sys.stderr)
    return 0


def cmd_commercials_dedupe(args: argparse.Namespace) -> int:
    from radio_classifier.commercials import dedupe_commercials

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(f"radio-classifier commercials dedupe: db not found: {db_path}", file=sys.stderr)
        return 1

    dry_run = True if args.dry_run else not args.apply
    with BroadcastStore(db_path) as store:
        report = dedupe_commercials(store, dry_run=dry_run)

    if not report.groups:
        print("radio-classifier: no duplicate commercial groups found", file=sys.stderr)
        return 0

    verb = "would fold" if report.dry_run else "folded"
    print(
        f"radio-classifier: {verb} {report.collapsed_pairs} duplicate commercial row(s) "
        f"across {len(report.groups)} group(s)",
        file=sys.stderr,
    )
    for group in report.groups:
        survivor = group.survivor
        print(
            f"  {survivor.canonical_brand!r} [{group.reason}] → "
            f"keep commercial_id={survivor.commercial_id} ({survivor.event_count} event(s))",
            file=sys.stderr,
        )
        for loser in group.losers:
            print(
                f"    drop commercial_id={loser.commercial_id} "
                f"({loser.brand!r}, {loser.event_count} event(s))",
                file=sys.stderr,
            )

    if not report.dry_run:
        print(
            f"\nradio-classifier: re-pointed {report.events_repointed} broadcast_events row(s); "
            f"re-pointed {report.brand_mentions_repointed} paid brand mention(s); "
            f"deleted {report.rows_deleted} commercial row(s)",
            file=sys.stderr,
        )
    return 0


def cmd_commercials_merge_brands(args: argparse.Namespace) -> int:
    from radio_classifier.commercials import merge_brands

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(
            f"radio-classifier commercials merge-brands: db not found: {db_path}",
            file=sys.stderr,
        )
        return 1

    dry_run = True if args.dry_run else not args.apply
    with BroadcastStore(db_path) as store:
        report = merge_brands(store, dry_run=dry_run)

    if not report.groups:
        print("radio-classifier: no duplicate brand groups found", file=sys.stderr)
        return 0

    verb = "would fold" if report.dry_run else "folded"
    print(
        f"radio-classifier: {verb} {report.collapsed_pairs} duplicate brand row(s) "
        f"across {len(report.groups)} advertiser(s)",
        file=sys.stderr,
    )
    for group in report.groups:
        survivor = group.survivor
        print(
            f"  → {group.display_name!r} (keep brand_id={survivor.brand_id}, "
            f"{survivor.total_refs} ref(s))",
            file=sys.stderr,
        )
        for loser in group.losers:
            print(
                f"    fold brand_id={loser.brand_id} {loser.canonical_name!r} "
                f"({loser.event_count} event(s), {loser.commercial_count} ad(s), "
                f"{loser.mention_count} mention(s))",
                file=sys.stderr,
            )

    if not report.dry_run:
        print(
            f"\nradio-classifier: re-pointed {report.events_repointed} broadcast_events row(s), "
            f"{report.mentions_repointed} brand mention(s), "
            f"{report.commercials_repointed} commercial(s) "
            f"(merged {report.commercials_merged} duplicate ad(s)); "
            f"deleted {report.brands_deleted} brand row(s)",
            file=sys.stderr,
        )
    return 0


def cmd_commercials_backfill_brands(args: argparse.Namespace) -> int:
    from radio_classifier.commercials import backfill_unbranded_commercials
    from radio_classifier.reports import parse_since

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(
            f"radio-classifier commercials backfill-brands: db not found: {db_path}",
            file=sys.stderr,
        )
        return 1

    # Window is optional here (default: whole DB), unlike report commands.
    if (args.from_utc or args.until_utc) and args.since:
        raise _CliConfigError(
            "backfill window accepts either --since or --from/--to, not both"
        )
    if args.from_utc:
        since_utc: str | None = _normalize_utc_text(args.from_utc)
    elif args.since:
        since_utc = parse_since(args.since)
    else:
        since_utc = None
    until_utc = _normalize_utc_text(args.until_utc) if args.until_utc else None

    classifier = None
    if args.llm:
        from radio_classifier.speech.ollama import OllamaSpeechClassifier

        classifier = OllamaSpeechClassifier(model=args.ollama_model)

    dry_run = True if args.dry_run else not args.apply
    with BroadcastStore(db_path) as store:
        report = backfill_unbranded_commercials(
            store,
            since_utc=since_utc,
            until_utc=until_utc,
            dry_run=dry_run,
            classifier=classifier,
            limit=args.limit,
        )

    verb = "would brand" if report.dry_run else "branded"
    print(
        f"radio-classifier: {verb} {report.events_branded} of {report.events_scanned} "
        f"unbranded commercial event(s) "
        f"(deterministic={report.deterministic_hits}, llm={report.llm_hits}"
        + (f", llm_errors={report.llm_errors}" if report.llm_errors else "")
        + ")",
        file=sys.stderr,
    )
    from collections import Counter

    by_brand = Counter(item.brand for item in report.items)
    for brand, count in sorted(by_brand.items(), key=lambda kv: (-kv[1], kv[0])):
        sources = {item.source for item in report.items if item.brand == brand}
        print(f"  {brand!r}: {count} event(s) [{'+'.join(sorted(sources))}]", file=sys.stderr)

    if report.events_scanned and report.events_branded == 0:
        print(
            "radio-classifier: no brands recovered "
            "(try --llm with a running Ollama server for higher recall)",
            file=sys.stderr,
        )
    return 0


def cmd_commercials_merge_boundaries(args: argparse.Namespace) -> int:
    from radio_classifier.commercials import merge_boundary_commercials
    from radio_classifier.reports import parse_since

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        print(
            f"radio-classifier commercials merge-boundaries: db not found: {db_path}",
            file=sys.stderr,
        )
        return 1

    if (args.from_utc or args.until_utc) and args.since:
        raise _CliConfigError(
            "merge-boundaries window accepts either --since or --from/--to, not both"
        )
    if args.from_utc:
        since_utc: str | None = _normalize_utc_text(args.from_utc)
    elif args.since:
        since_utc = parse_since(args.since)
    else:
        since_utc = None
    until_utc = _normalize_utc_text(args.until_utc) if args.until_utc else None

    dry_run = True if args.dry_run else not args.apply
    with BroadcastStore(db_path) as store:
        report = merge_boundary_commercials(
            store,
            since_utc=since_utc,
            until_utc=until_utc,
            min_similarity=args.min_similarity,
            max_gap_seconds=args.max_gap_seconds,
            dry_run=dry_run,
        )

    verb = "would merge" if report.dry_run else "merged"
    print(
        f"radio-classifier: {verb} {report.events_merged} of {report.events_scanned} "
        f"unbranded commercial fragment(s) into a branded neighbor "
        f"(min-similarity={args.min_similarity})",
        file=sys.stderr,
    )
    from collections import Counter

    by_brand = Counter(item.brand for item in report.items)
    for brand, count in sorted(by_brand.items(), key=lambda kv: (-kv[1], kv[0])):
        sims = [item.similarity for item in report.items if item.brand == brand]
        print(
            f"  {brand!r}: {count} fragment(s) [sim {min(sims):.2f}-{max(sims):.2f}]",
            file=sys.stderr,
        )
    return 0


# -------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "prereq-check":
            return cmd_prereq(args)
        if args.command == "db":
            if args.db_command == "init":
                return cmd_db_init(args)
            if args.db_command == "migrate-from-live105sux":
                return cmd_db_migrate(args)
            return 2
        if args.command == "fingerprint":
            if args.fp_command == "index":
                return cmd_fp_index(args)
            if args.fp_command == "eval":
                return cmd_fp_eval(args)
            if args.fp_command == "explain":
                return cmd_fp_explain(args)
            return 2
        if args.command == "classify":
            return cmd_classify(args)
        if args.command == "ingest":
            return cmd_ingest(args)
        if args.command == "capture":
            if args.capture_command == "chunks":
                return cmd_capture_chunks(args)
            return 2
        if args.command == "runs":
            return cmd_runs(args)
        if args.command == "report":
            return cmd_report(args)
        if args.command == "seed":
            if args.seed_command == "scrape":
                return cmd_seed_scrape(args)
            if args.seed_command == "download":
                return cmd_seed_download(args)
            return 2
        if args.command == "songs":
            if args.songs_command == "discovered":
                return cmd_songs_discovered(args)
            if args.songs_command == "promote":
                return cmd_songs_promote(args)
            if args.songs_command == "dedupe":
                return cmd_songs_dedupe(args)
            if args.songs_command == "stitch":
                return cmd_songs_stitch(args)
            if args.songs_command == "enrich-releases":
                return cmd_songs_enrich_releases(args)
            return 2
        if args.command == "commercials":
            if args.commercials_command == "dedupe":
                return cmd_commercials_dedupe(args)
            if args.commercials_command == "backfill-brands":
                return cmd_commercials_backfill_brands(args)
            if args.commercials_command == "merge-boundaries":
                return cmd_commercials_merge_boundaries(args)
            if args.commercials_command == "merge-brands":
                return cmd_commercials_merge_brands(args)
            return 2
        return 1
    except _CliConfigError as exc:
        print(f"radio-classifier: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
