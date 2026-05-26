"""yt-dlp-driven reference audio downloader.

Optional (``[seeding]`` extra). Never imported by runtime.

Operator responsibility: comply with YouTube ToS for their use case.
"""

from __future__ import annotations

import re
import subprocess
import sys
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
    audio_format: str = "mp3"
    audio_quality: str = "192"  # kbps for mp3
    min_quality_kbps: int = 96
    search_template: str = "ytsearch1:{artist} {title} audio"


@dataclass
class DownloadResult:
    track: Track
    output_path: Path | None
    ok: bool
    message: str = ""
    skipped: bool = False


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
    out_template = str(config.output_dir / (stem + ".%(ext)s"))
    query = config.search_template.format(artist=track.artist, title=track.title)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-x",
        "--audio-format",
        config.audio_format,
        "--audio-quality",
        config.audio_quality,
        "-o",
        out_template,
        "--no-playlist",
        query,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return DownloadResult(
            track=track,
            output_path=None,
            ok=False,
            message=(proc.stderr or proc.stdout).strip()[:400],
        )
    expected = config.output_dir / f"{stem}.{config.audio_format}"
    if not expected.exists():
        return DownloadResult(
            track=track,
            output_path=None,
            ok=False,
            message="yt-dlp returned 0 but expected output file is missing",
        )
    return DownloadResult(track=track, output_path=expected, ok=True)


def download_tracks(tracks: list[Track], config: DownloadConfig) -> list[DownloadResult]:
    """Download all tracks sequentially."""
    return [download_track(t, config) for t in tracks]
