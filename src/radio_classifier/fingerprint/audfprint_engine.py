"""Thin wrapper around the ``audfprint`` CLI for indexing + matching.

We invoke ``audfprint`` as a subprocess rather than importing it as a library
because the upstream API is unstable across forks and the CLI is the documented
public surface. The wrapper:

* keeps **one** persistent index file (``data/audfprint/songs.pklz`` by
  default);
* speaks the ``audfprint match`` ``--list`` text protocol for one-shot lookups;
* parses ``audfprint``'s ``Matched`` stdout lines and turns them into
  :class:`FingerprintResult` rows.

Network is never touched; everything is local file I/O.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from radio_classifier.fingerprint.types import FingerprintResult, FingerprintStatus
from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.speech.wav_temp import temp_wav_for_window


_AUDFPRINT_BIN_ENV = "RADIO_CLASSIFIER_AUDFPRINT_BIN"
_MATCH_LINE_RE = re.compile(
    r"^Matched\s+(?P<query>[^\s]+).*?as\s+(?P<track>.+?)"
    r"(?:\s+at\s+-?\d+(?:\.\d+)?\s*s)?"
    r"\s+with\s+(?P<count>\d+)\s+",
    re.IGNORECASE,
)


def _audfprint_argv() -> list[str]:
    """Resolve the ``audfprint`` invocation.

    Order of preference:
      1. ``RADIO_CLASSIFIER_AUDFPRINT_BIN`` env var (full command, may be quoted).
         ``~`` and ``$VAR`` references in each token are expanded so operators
         can write e.g. ``python ~/dev/audfprint/audfprint.py``.
      2. ``audfprint`` on PATH.
      3. ``python -m audfprint`` (works if a packaged fork ever lands on PyPI).
    """
    import os
    import shlex

    override = os.environ.get(_AUDFPRINT_BIN_ENV)
    if override:
        return [os.path.expanduser(os.path.expandvars(t)) for t in shlex.split(override)]
    if shutil.which("audfprint"):
        return ["audfprint"]
    local_clone = Path("~/dev/audfprint/audfprint.py").expanduser()
    if local_clone.is_file():
        return [sys.executable, str(local_clone)]
    return [sys.executable, "-m", "audfprint"]


@dataclass
class AudfprintConfig:
    """Match-time tunables. Defaults tuned for FM (validated by Phase J eval).

    ``min_count`` is the minimum number of common hashes required to accept a
    match. Higher values reduce false positives at the cost of recall under
    noisy reception. Empirically with a small seeded index, 5 was too permissive
    on live FM (sister-song / similar-key false matches); 12 is the new floor.
    Tune via :class:`AudfprintIndex` per deployment if needed.
    """

    min_count: int = 12
    match_win: int = 2
    density: int = 20
    max_matches: int = 1


@dataclass
class AudfprintIndex:
    """A handle to a single ``audfprint`` hash table file."""

    index_path: Path
    config: AudfprintConfig = field(default_factory=AudfprintConfig)

    def exists(self) -> bool:
        return self.index_path.exists() and self.index_path.is_file()

    # ------------------------------------------------------------------- index
    def build_or_extend(self, audio_files: list[Path], *, extend: bool = False) -> None:
        """Build the index from a list of reference WAV / MP3 files.

        ``extend=False`` creates a fresh table (overwrites). ``extend=True``
        adds tracks to the existing table.
        """
        if not audio_files:
            raise ValueError("audio_files is empty")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = list(_audfprint_argv())
        cmd.append("add" if extend else "new")
        cmd += [
            "-d",
            str(self.index_path),
            "--density",
            str(self.config.density),
        ]
        cmd += [str(p) for p in audio_files]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                "audfprint indexing failed "
                f"(rc={proc.returncode}): {proc.stderr.strip()[:400]}"
            )

    # ------------------------------------------------------------------ match
    def match_window(self, window: AudioWindow) -> FingerprintResult:
        """Match a single :class:`AudioWindow` against the seeded index."""
        if not self.exists():
            return FingerprintResult(
                status=FingerprintStatus.skipped,
                window_start_utc=window.window_start_utc,
                message=f"audfprint index missing: {self.index_path}",
            )
        with temp_wav_for_window(window) as wav_path:
            return self._match_file(wav_path, window.window_start_utc)

    def match_file(self, wav_path: Path) -> FingerprintResult:
        """Match a pre-existing WAV file (used by the eval harness)."""
        if not self.exists():
            return FingerprintResult(
                status=FingerprintStatus.skipped,
                window_start_utc="",
                message=f"audfprint index missing: {self.index_path}",
            )
        return self._match_file(wav_path, "")

    def _match_file(self, wav_path: Path, window_start_utc: str) -> FingerprintResult:
        cmd = list(_audfprint_argv())
        cmd += [
            "match",
            "-d",
            str(self.index_path),
            "--min-count",
            str(self.config.min_count),
            "--match-win",
            str(self.config.match_win),
            "--max-matches",
            str(self.config.max_matches),
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return FingerprintResult(
                status=FingerprintStatus.error,
                window_start_utc=window_start_utc,
                message=(proc.stderr or proc.stdout).strip()[:400],
            )
        return parse_audfprint_match_output(proc.stdout, window_start_utc)


def parse_audfprint_match_output(stdout: str, window_start_utc: str) -> FingerprintResult:
    """Parse ``audfprint match`` stdout into a :class:`FingerprintResult`.

    Recognized formats (across upstream versions):

      ``Matched   query.wav 5.0 sec   ... as REF.wav at  3.4 with    13 ...``
      ``Matched query.wav as Foo Bar - Title.mp3 with 13 of ...``
      ``NOMATCH query.wav``

    Anything else falls through to ``no_match``.
    """
    if not stdout:
        return FingerprintResult(
            status=FingerprintStatus.no_match,
            window_start_utc=window_start_utc,
        )

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("NOMATCH"):
            return FingerprintResult(
                status=FingerprintStatus.no_match,
                window_start_utc=window_start_utc,
            )
        m = _MATCH_LINE_RE.search(line)
        if not m:
            continue
        track = m.group("track").strip()
        count = int(m.group("count"))
        artist, title = _split_track_id(track)
        return FingerprintResult(
            status=FingerprintStatus.match,
            window_start_utc=window_start_utc,
            track_id=track,
            artist=artist,
            title=title,
            match_score=float(count),
        )

    return FingerprintResult(
        status=FingerprintStatus.no_match,
        window_start_utc=window_start_utc,
    )


def _split_track_id(track: str) -> tuple[str | None, str | None]:
    """Heuristic split of an ``audfprint`` track label into (artist, title).

    Conventions we honour:

    * ``"Artist - Title.mp3"`` → ``("Artist", "Title")``.
    * ``"Artist_-_Title.wav"`` (underscored — some audfprint builds) → ``("Artist", "Title")``.
    * ``"Title.wav"`` → ``(None, "Title")``.
    * Path-like strings → take basename then strip extension.
    """
    raw = track
    if "/" in raw or "\\" in raw:
        raw = raw.replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".webm"):
        if raw.lower().endswith(ext):
            raw = raw[: -len(ext)]
            break
    for sep in (" - ", "_-_"):
        if sep in raw:
            artist, title = raw.split(sep, 1)
            return artist.strip() or None, title.strip() or None
    return None, raw.strip() or None
