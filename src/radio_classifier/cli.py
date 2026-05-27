"""Command-line interface for radio-classifier.

Subcommands:

* ``prereq-check``  — diagnostics for GPU / CUDA / Ollama / rtl_fm / audfprint.
* ``db init``        — apply ``db/schema.sql`` (schema v2) to a SQLite file.
* ``db migrate-from-live105sux`` — port a live105sux v1 SQLite into v2.
* ``fingerprint index`` — build / extend the audfprint song index.
* ``fingerprint eval``  — run the recall harness against a truth CSV.
* ``ingest``         — live RTL-SDR capture through the 3-tier funnel.
* ``classify``       — offline 3-tier funnel on a WAV file.
* ``report``         — CLI-only reports (commercials / brands / songs / timeline / summary).
* ``seed scrape``    — print a tracklist parsed from a station page.
* ``seed download``  — fetch reference audio via yt-dlp (``[seeding]``).
* ``songs discovered`` — list Shazam-discovered songs (and their tracklist status).
* ``songs promote``    — append selected Shazam discoveries to ``tracklist.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from radio_classifier.ingest.rtl_fm import RtlFmExitedError, RtlFmStream
from radio_classifier.ingest.wav import read_mono_s16le_wav, write_mono_s16le_wav
from radio_classifier.ingest.windows import iter_overlapping_windows
from radio_classifier.persistence import BroadcastStore, persist_finalize, persist_input
from radio_classifier.segments import SegmentReducer


def _project_root() -> Path:
    """``src/radio_classifier/cli.py`` → parents[2] = repo root."""
    return Path(__file__).resolve().parents[2]


def _default_db_path() -> Path:
    return _project_root() / "data" / "radio_classifier.db"


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


# --------------------------------------------------------------------- parser
def _add_persist_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--persist",
        action="store_true",
        help="Append segment rows to SQLite (default DB: data/radio_classifier.db)",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite database file (default: <repo>/data/radio_classifier.db)",
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
        "--whisper-model",
        type=str,
        default="medium.en",
        help="faster-whisper model size or path (default: medium.en)",
    )
    p.add_argument(
        "--whisper-device",
        type=str,
        default="cuda",
        help="Device for faster-whisper (default: cuda)",
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
    db_init = db_sub.add_parser("init", help="Initialise SQLite schema v2")
    db_init.add_argument("--db-path", type=Path, default=None)

    mig = db_sub.add_parser(
        "migrate-from-live105sux", help="Migrate a live105sux v1 DB into a fresh v2 DB"
    )
    mig.add_argument("--src", type=Path, required=True, help="Source live105sux SQLite path")
    mig.add_argument("--dst", type=Path, required=True, help="Destination radio-classifier SQLite path")

    # ---- fingerprint
    fp = sub.add_parser("fingerprint", help="Manage the Tier-1 song fingerprint index")
    fp_sub = fp.add_subparsers(dest="fp_command", required=True)
    fp_idx = fp_sub.add_parser("index", help="Build / extend the audfprint index")
    fp_idx.add_argument("--dir", type=Path, required=True, help="Directory of reference audio files")
    fp_idx.add_argument("--out", type=Path, default=None, help="Index file path (default: data/audfprint/songs.pklz)")
    fp_idx.add_argument("--extend", action="store_true", help="Add to an existing index instead of overwriting")
    fp_idx.add_argument("--glob", type=str, default="**/*", help="File glob inside --dir (default: all files)")

    fp_eval = fp_sub.add_parser("eval", help="Recall harness against truth CSV")
    fp_eval.add_argument("--index", type=Path, default=None, help="Index file (default: data/audfprint/songs.pklz)")
    fp_eval.add_argument("--truth", type=Path, required=True, help="CSV: clip,song_id or clip,artist,title")

    # ---- classify (offline)
    cls = sub.add_parser("classify", help="3-tier funnel over a WAV file")
    cls.add_argument("-i", "--input", type=Path, required=True, help="Mono 16-bit WAV path")
    cls.add_argument("--sample-rate-override", type=int, default=None)
    _add_window_arguments(cls)
    _add_funnel_arguments(cls)
    _add_persist_arguments(cls)
    _add_json_lines(cls)
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

    # ---- report
    rep = sub.add_parser("report", help="CLI reports against a v2 SQLite DB")
    rep_sub = rep.add_subparsers(dest="report_command", required=True)
    for name in ("commercials", "brands", "songs", "timeline", "summary"):
        r = rep_sub.add_parser(name)
        r.add_argument("--db-path", type=Path, default=None)
        r.add_argument("--since", type=str, default="24h")
        r.add_argument("--top", type=int, default=10, help="Limit (default 10)")
        if name == "commercials":
            r.add_argument("--brand", type=str, default=None, help="Filter to a single brand")
        if name == "timeline":
            r.add_argument("--limit", type=int, default=500)

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
    seed_dl.add_argument("--audio-format", type=str, default="mp3")
    seed_dl.add_argument("--audio-quality", type=str, default="192")

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
    path = args.db_path or _default_db_path()
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
    index.build_or_extend(files, extend=args.extend or out.exists())
    print(
        f"radio-classifier: indexed {len(files)} files into {out}",
        file=sys.stderr,
    )
    return 0


def cmd_fp_eval(args: argparse.Namespace) -> int:
    from radio_classifier.fingerprint import AudfprintIndex
    from radio_classifier.seeding.eval import evaluate, load_truth

    index_path = args.index or _default_index_path()
    index = AudfprintIndex(index_path=index_path)
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


# -------------------------------------------------- funnel construction
def _build_funnel(args: argparse.Namespace) -> "FunnelBundle":
    from radio_classifier.commercials import CommercialIdentityResolver
    from radio_classifier.pipeline import FunnelOrchestrator

    tier1 = None
    if not args.no_tier1:
        from radio_classifier.fingerprint import AudfprintIndex

        index_path = args.audfprint_index or _default_index_path()
        idx = AudfprintIndex(index_path=index_path)
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
            from radio_classifier.speech import OllamaSpeechClassifier, WhisperTranscriber, run_speech_pipeline

            transcriber = WhisperTranscriber(
                model_size=args.whisper_model,
                device=args.whisper_device,
                compute_type=args.whisper_compute_type,
                language=args.whisper_language,
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

            force_yamnet_cpu = tier3 is not None and args.whisper_device == "cuda"
            if force_yamnet_cpu:
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
        db_path = args.db_path if args.db_path is not None else _default_db_path()
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
    )

    return FunnelBundle(orchestrator=orchestrator, store=store, reducer=reducer)


class FunnelBundle:
    def __init__(self, *, orchestrator, store, reducer) -> None:
        self.orchestrator = orchestrator
        self.store = store
        self.reducer = reducer

    def close(self, *, windows: list, window_seconds: float) -> None:
        try:
            if self.reducer is not None and self.store is not None and windows:
                persist_finalize(
                    self.reducer,
                    self.store,
                    last_window_start_utc=windows[-1].window_start_utc,
                    window_seconds=window_seconds,
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
    clock_start_ns = time.time_ns()
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

    bundle = _build_funnel(args)
    try:
        for w in windows:
            r = bundle.orchestrator.process(w)
            _emit_funnel(args, r, window_seconds=args.window_seconds)
            if bundle.reducer is not None:
                new_ids = persist_input(bundle.reducer, bundle.store, r.segment_input)
                if new_ids:
                    # mentions belong to the window that closed the *previous* segment
                    _persist_brand_mentions(
                        bundle.store,
                        event_id=new_ids[-1],
                        mentions=r.brand_mentions or [],
                        heard_utc=r.window_start_utc,
                    )
    finally:
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
                new_ids = persist_input(bundle.reducer, bundle.store, r.segment_input)
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


# ------------------------------------------------------------------ report
def cmd_report(args: argparse.Namespace) -> int:
    from radio_classifier.reports import (
        brands_top,
        commercials_top,
        format_brands,
        format_commercials,
        format_songs,
        format_summary,
        format_timeline,
        parse_since,
        songs_top,
        summary,
        timeline,
    )

    db_path = args.db_path or _default_db_path()
    if not db_path.exists():
        print(f"radio-classifier report: db not found: {db_path}", file=sys.stderr)
        return 1
    since_utc = parse_since(args.since)
    with BroadcastStore(db_path) as store:
        if args.report_command == "commercials":
            rows = commercials_top(
                store,
                since_utc=since_utc,
                top_n=args.top,
                brand=getattr(args, "brand", None),
            )
            print(format_commercials(rows))
        elif args.report_command == "brands":
            rows = brands_top(store, since_utc=since_utc, top_n=args.top)
            print(format_brands(rows))
        elif args.report_command == "songs":
            rows = songs_top(store, since_utc=since_utc, top_n=args.top)
            print(format_songs(rows))
        elif args.report_command == "timeline":
            rows = timeline(store, since_utc=since_utc, limit=args.limit)
            print(format_timeline(rows))
        elif args.report_command == "summary":
            rows = summary(store, since_utc=since_utc)
            print(format_summary(rows))
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

    db_path = args.db_path or _default_db_path()
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
    print(
        f"\n{len(rows)} Shazam discoveries "
        f"({len(missing_ids)} not yet indexed in {tracklist}).",
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

    db_path = args.db_path or _default_db_path()
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


# -------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
        return 2
    if args.command == "classify":
        return cmd_classify(args)
    if args.command == "ingest":
        return cmd_ingest(args)
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
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
