"""yt-dlp-driven reference audio downloader.

Optional (``[seeding]`` extra). Never imported by runtime.

Operator responsibility: comply with YouTube ToS for their use case.
"""

from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from radio_classifier.seeding.scrape import Track


_SAFE = re.compile(r"[^A-Za-z0-9._\- ]+")

# Any of these extensions on disk for a track's stem means we already have
# usable reference audio and should not re-invoke yt-dlp. This deliberately
# includes formats yt-dlp may fall back to (opus/webm) when its preferred
# format is unavailable for a given source.
_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".m4a", ".aac", ".opus", ".webm", ".ogg", ".oga", ".flac", ".wav", ".mp4"}
)

# Sentinel ``audio_format`` values that mean "do not re-encode; keep whatever
# stream yt-dlp downloaded". Anything else (e.g. ``"mp3"``) triggers the
# ffmpeg extract+transcode pass via ``-x --audio-format``.
_NO_TRANSCODE_FORMATS: frozenset[str] = frozenset({"", "best", "bestaudio", "native"})


def safe_filename_stem(track: Track) -> str:
    """Build a sanitized filename stem like ``Artist - Title``."""
    raw = f"{track.artist} - {track.title}"
    return _SAFE.sub("_", raw).strip("_ ")


def existing_audio_for_stem(output_dir: Path, stem: str) -> Path | None:
    """Return an existing audio file matching ``stem.*`` if one is on disk.

    Used by the downloader to skip tracks whose reference audio was already
    fetched in a previous run. Returns ``None`` if no matching file exists.
    """
    if not output_dir.is_dir():
        return None
    for candidate in sorted(output_dir.glob(f"{stem}.*")):
        if candidate.is_file() and candidate.suffix.lower() in _AUDIO_EXTENSIONS:
            return candidate
    return None


@dataclass
class DownloadConfig:
    output_dir: Path
    # We default to ``mp3`` at 192 kbps because the audfprint fingerprint
    # quality is markedly more consistent on uniformly-encoded mp3 refs
    # than on a mix of mp3/m4a/opus. A 2026-05-31 validation pass found
    # freshly-downloaded ``.webm/opus`` refs producing high-density,
    # spurious matches that consistently won audfprint's ``rank 0`` slot
    # across unrelated query clips, crowding out the real higher-score
    # matches. Set ``audio_format`` to ``"best"`` to bypass transcoding
    # when speed dominates over recall (and you have validated quality).
    audio_format: str = "mp3"
    audio_quality: str = "192"
    min_quality_kbps: int = 96
    search_template: str = "ytsearch1:{artist} {title} audio"
    # Number of concurrent yt-dlp subprocesses for ``download_tracks``.
    # YouTube tolerates a handful of parallel downloads from one host; keep
    # it modest so we do not get rate-limited mid-batch.
    concurrency: int = 4


@dataclass
class DownloadResult:
    track: Track
    output_path: Path | None
    ok: bool
    message: str = ""
    skipped: bool = False


def _build_yt_dlp_cmd(stem: str, query: str, config: DownloadConfig) -> list[str]:
    """Compose the yt-dlp invocation for one track.

    Branches on ``audio_format``:

    * ``best`` / ``bestaudio`` / ``native`` / empty: ``-f bestaudio/best``
      with no ``-x`` re-encode. yt-dlp writes the stream as downloaded
      (extension follows the source container — typically ``.m4a`` or
      ``.webm``). This is the fast path and the new default.
    * any other value (``mp3``, ``opus``, ...): legacy ``-x --audio-format
      <fmt>`` pipeline that re-encodes via ffmpeg.
    """

    out_template = str(config.output_dir / (stem + ".%(ext)s"))
    fmt = (config.audio_format or "").strip().lower()
    cmd: list[str] = [sys.executable, "-m", "yt_dlp"]
    if fmt in _NO_TRANSCODE_FORMATS:
        cmd += ["-f", "bestaudio/best", "--no-keep-video"]
    else:
        cmd += ["-x", "--audio-format", fmt, "--audio-quality", config.audio_quality]
    cmd += ["-o", out_template, "--no-playlist", query]
    return cmd


def download_track(track: Track, config: DownloadConfig) -> DownloadResult:
    """Download one track via yt-dlp, or skip if reference audio already exists.

    Returns ``DownloadResult(ok=True, skipped=True, ...)`` if a file matching
    the safe stem with any common audio extension is already present in
    ``config.output_dir`` — this lets ``seed download`` be re-run cheaply
    after appending new entries to the tracklist.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_filename_stem(track)
    existing = existing_audio_for_stem(config.output_dir, stem)
    if existing is not None:
        return DownloadResult(
            track=track,
            output_path=existing,
            ok=True,
            message=f"already on disk: {existing.name}",
            skipped=True,
        )
    query = config.search_template.format(artist=track.artist, title=track.title)
    cmd = _build_yt_dlp_cmd(stem, query, config)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return DownloadResult(
            track=track,
            output_path=None,
            ok=False,
            message=(proc.stderr or proc.stdout).strip()[:400],
        )
    saved = existing_audio_for_stem(config.output_dir, stem)
    if saved is None:
        return DownloadResult(
            track=track,
            output_path=None,
            ok=False,
            message="yt-dlp returned 0 but no matching audio file appeared on disk",
        )
    return DownloadResult(track=track, output_path=saved, ok=True)


def download_tracks(tracks: list[Track], config: DownloadConfig) -> list[DownloadResult]:
    """Download all tracks, optionally in parallel.

    Each ``yt-dlp`` invocation spends most of its wall time waiting on the
    network and on a single-threaded ffmpeg pass, so we get near-linear
    speedup running a handful in parallel. ``config.concurrency`` caps the
    pool; falling back to a serial loop when only one worker is requested
    keeps tracebacks simple in the common case.
    """
    workers = max(1, int(config.concurrency))
    if workers == 1 or len(tracks) <= 1:
        return [download_track(t, config) for t in tracks]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: download_track(t, config), tracks))
