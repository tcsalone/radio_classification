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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from radio_classifier.fingerprint.types import FingerprintResult, FingerprintStatus
from radio_classifier.ingest.wav import write_mono_s16le_wav
from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.speech.wav_temp import temp_wav_for_window


_AUDFPRINT_BIN_ENV = "RADIO_CLASSIFIER_AUDFPRINT_BIN"
_AUDFPRINT_INDEX_NCORES_ENV = "RADIO_CLASSIFIER_AUDFPRINT_INDEX_NCORES"
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


def _default_index_ncores() -> int:
    """Pick a sensible worker count for ``audfprint new --ncores``.

    Hashing reference files is embarrassingly parallel and CPU-bound. Upstream
    defaults to 1, which leaves a multi-core box mostly idle and turns a
    ~150-file rebuild into a 15+ minute wait. We default to a small fraction
    of the cores so a concurrent ``classify`` (which already uses CPU for
    Whisper/YAMNet/audfprint match) is not starved, capped at 6 because
    audfprint's marginal speedup flattens out beyond that.

    Operators can override with ``RADIO_CLASSIFIER_AUDFPRINT_INDEX_NCORES``.
    """

    import os

    override = os.environ.get(_AUDFPRINT_INDEX_NCORES_ENV)
    if override:
        try:
            value = int(override)
        except ValueError:
            value = 0
        if value > 0:
            return value
    cpu = os.cpu_count() or 1
    return max(1, min(6, cpu))


@dataclass
class AudfprintConfig:
    """Match-time tunables. Defaults tuned for FM (validated by Phase J eval).

    ``min_count`` is the minimum number of common hashes required to accept a
    match. Higher values reduce false positives at the cost of recall under
    noisy reception. Empirical history:

    *  5 (upstream default) — far too permissive on live FM.
    * 12 — first tightening; still produced clusters of 10-40s false matches
      with scores in the 20-50 range (e.g. SLTS / Otherside / Clocks ghosts).
    * 60 — conservative floor. A 2026-05-28 morning-drive validation run showed a
      bimodal score distribution with a clean valley at [50, 70): every
      manually-confirmed real match scored >=84, every confirmed false
      positive scored <=49. Setting the floor at 60 drops the noise pile
      without losing any verified real hit.
    * 45 — first relaxation. Borderline matches in the 45-59 range are
      allowed out of audfprint so the funnel can recover weak real hits, but
      :class:`FunnelOrchestrator` requires additional adjacent same-track
      confirmation before accepting them.
    * 30 — current candidate floor. The 2026-05-31 validated-unknowns eval
      showed real recall going from 0/7 at 45 to 1/7 at 30 (Bad Omens
      recovered at score 60), with the known Linkin Park / Temper City
      collision still scoring 67 — well above either threshold. The
      ``low_confidence_fingerprint_required_repeats`` guard in the funnel
      remains the false-positive backstop for scores in [30, 60).

    Tune via :class:`AudfprintIndex` per deployment if recall degrades on a
    different station / index.
    """

    min_count: int = 30
    match_win: int = 2
    density: int = 20
    # NOTE: audfprint's ``rank 0`` is not always the highest-scoring match. A
    # noisy reference (e.g. a poor-quality opus rip) can win ``rank 0`` with a
    # low score and crowd out the real higher-score match at ``rank 1``+. We
    # request the top 5 candidates and pick the best score on our side.
    max_matches: int = 5
    batch_size: int = 200
    index_ncores: int = field(default_factory=_default_index_ncores)


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
        ncores = max(1, int(self.config.index_ncores))
        if ncores > 1:
            cmd += ["--ncores", str(ncores)]
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

    def match_windows(self, windows: list[AudioWindow]) -> list[FingerprintResult]:
        """Batch-match windows with far fewer audfprint subprocess/index loads.

        Offline ``classify`` can have hundreds of overlapping windows. Calling
        ``audfprint match`` once per window is slow because every invocation
        starts Python and reloads the index. This method writes a chunk of
        temporary WAVs and passes them to one ``audfprint match`` call, reducing
        N subprocess/index loads to roughly ``ceil(N / batch_size)``.
        """
        if not windows:
            return []
        if not self.exists():
            return [
                FingerprintResult(
                    status=FingerprintStatus.skipped,
                    window_start_utc=w.window_start_utc,
                    message=f"audfprint index missing: {self.index_path}",
                )
                for w in windows
            ]

        results: list[FingerprintResult] = []
        batch_size = max(1, int(self.config.batch_size))
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            results.extend(self._match_window_chunk(chunk, chunk_offset=start))
        return results

    def match_file(self, wav_path: Path) -> FingerprintResult:
        """Match a pre-existing WAV file (used by the eval harness)."""
        if not self.exists():
            return FingerprintResult(
                status=FingerprintStatus.skipped,
                window_start_utc="",
                message=f"audfprint index missing: {self.index_path}",
            )
        return self._match_file(wav_path, "")

    def explain(
        self,
        wav_path: Path,
        *,
        max_matches: int = 20,
        min_count: int = 1,
    ) -> list[tuple[str, int]]:
        """Return every reference candidate ``(track_id, count)`` for one clip.

        Unlike :meth:`match_file` (which returns the best-scoring match), this
        surfaces the full ranked list so an operator can see whether an
        expected track shows up at all and at what score. Used by the
        ``radio-classifier fingerprint explain`` CLI when debugging why a
        known broadcast clip is being missed.

        ``min_count`` defaults to ``1`` so the diagnostic still surfaces real
        but very weak candidates that the production funnel would reject.
        Candidates are returned sorted by descending ``count``.
        """
        if not self.exists():
            return []
        cmd = list(_audfprint_argv())
        cmd += [
            "match",
            "-d",
            str(self.index_path),
            "--min-count",
            str(int(min_count)),
            "--match-win",
            str(self.config.match_win),
            "--max-matches",
            str(int(max_matches)),
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return []
        return parse_audfprint_candidates(proc.stdout)

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

    def _match_window_chunk(
        self,
        windows: list[AudioWindow],
        *,
        chunk_offset: int,
    ) -> list[FingerprintResult]:
        with tempfile.TemporaryDirectory(prefix="radio-classifier-audfprint-") as tmp:
            tmp_dir = Path(tmp)
            wav_paths: list[Path] = []
            starts: list[str] = []
            for i, window in enumerate(windows, start=chunk_offset):
                wav_path = tmp_dir / f"window_{i:06d}.wav"
                write_mono_s16le_wav(wav_path, window.samples, window.sample_rate_hz)
                wav_paths.append(wav_path)
                starts.append(window.window_start_utc)

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
            ]
            cmd += [str(p) for p in wav_paths]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                message = (proc.stderr or proc.stdout).strip()[:400]
                return [
                    FingerprintResult(
                        status=FingerprintStatus.error,
                        window_start_utc=s,
                        message=message,
                    )
                    for s in starts
                ]
            return parse_audfprint_batch_output(proc.stdout, wav_paths, starts)


def parse_audfprint_match_output(stdout: str, window_start_utc: str) -> FingerprintResult:
    """Parse ``audfprint match`` stdout into a :class:`FingerprintResult`.

    Recognized formats (across upstream versions):

      ``Matched   query.wav 5.0 sec   ... as REF.wav at  3.4 with    13 ...``
      ``Matched query.wav as Foo Bar - Title.mp3 with 13 of ...``
      ``NOMATCH query.wav``

    When multiple ``Matched`` lines are present (because we ask audfprint
    for the top-N candidates) we choose the line with the highest hash
    count, not the first. Audfprint's ``rank`` field can put a noisy
    reference with a low score at ``rank 0`` and the real high-score match
    at ``rank 1+`` — picking the best by score side-steps that.

    Anything else falls through to ``no_match``.
    """
    if not stdout:
        return FingerprintResult(
            status=FingerprintStatus.no_match,
            window_start_utc=window_start_utc,
        )

    best_track: str | None = None
    best_count: int = -1

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
        if count > best_count:
            best_count = count
            best_track = track

    if best_track is None:
        return FingerprintResult(
            status=FingerprintStatus.no_match,
            window_start_utc=window_start_utc,
        )

    artist, title = _split_track_id(best_track)
    return FingerprintResult(
        status=FingerprintStatus.match,
        window_start_utc=window_start_utc,
        track_id=best_track,
        artist=artist,
        title=title,
        match_score=float(best_count),
    )


def parse_audfprint_candidates(stdout: str) -> list[tuple[str, int]]:
    """Return every ``(track_id, count)`` candidate emitted by ``audfprint match``.

    Mirrors the parsing logic of :func:`parse_audfprint_match_output` but
    preserves the full list instead of collapsing to one winner. Results are
    sorted by descending ``count``; tied counts preserve audfprint's emission
    order. Empty input or no ``Matched`` lines returns ``[]``.
    """
    if not stdout:
        return []
    candidates: list[tuple[str, int, int]] = []
    for emit_index, raw in enumerate(stdout.splitlines()):
        line = raw.strip()
        if not line or line.upper().startswith("NOMATCH"):
            continue
        m = _MATCH_LINE_RE.search(line)
        if not m:
            continue
        track = m.group("track").strip()
        count = int(m.group("count"))
        candidates.append((track, count, emit_index))
    candidates.sort(key=lambda c: (-c[1], c[2]))
    return [(track, count) for track, count, _ in candidates]


def parse_audfprint_batch_output(
    stdout: str,
    query_paths: list[Path],
    window_start_utc: list[str],
) -> list[FingerprintResult]:
    """Parse one ``audfprint match`` output containing many query files.

    When audfprint is invoked with ``--max-matches > 1`` it emits multiple
    ``Matched`` lines per query. For each query we keep the line with the
    highest hash count, mirroring :func:`parse_audfprint_match_output`.
    """
    results = [
        FingerprintResult(
            status=FingerprintStatus.no_match,
            window_start_utc=ts,
        )
        for ts in window_start_utc
    ]
    best_count: list[int] = [-1] * len(window_start_utc)
    query_to_index: dict[str, int] = {}
    for i, path in enumerate(query_paths):
        query_to_index[str(path)] = i
        query_to_index[path.name] = i

    def lookup(query: str) -> int | None:
        if query in query_to_index:
            return query_to_index[query]
        return query_to_index.get(Path(query).name)

    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("NOMATCH"):
            parts = line.split()
            if len(parts) >= 2:
                idx = lookup(parts[1])
                if idx is not None and results[idx].status is not FingerprintStatus.match:
                    results[idx] = FingerprintResult(
                        status=FingerprintStatus.no_match,
                        window_start_utc=window_start_utc[idx],
                    )
            continue

        m = _MATCH_LINE_RE.search(line)
        if not m:
            continue
        idx = lookup(m.group("query").strip())
        if idx is None:
            continue
        count = int(m.group("count"))
        if count <= best_count[idx]:
            continue
        track = m.group("track").strip()
        artist, title = _split_track_id(track)
        results[idx] = FingerprintResult(
            status=FingerprintStatus.match,
            window_start_utc=window_start_utc[idx],
            track_id=track,
            artist=artist,
            title=title,
            match_score=float(count),
        )
        best_count[idx] = count

    return results


# Tail suffix that marks an alternate reference recording for the same song
# identity. Lets the operator drop multiple references for a single track
# into ``data/reference/songs/`` — e.g. ``Oasis - Wonderwall.mp3`` plus
# ``Oasis - Wonderwall (alt).mp3`` (a remaster or live cut from a different
# YouTube source) — and have both refs resolve to one ``songs`` row.
#
# Matched keywords are deliberately narrow: ``alt``, ``alt 2``, ``ref``,
# ``ref 2``, ``reference``, ``source``, ``v2`` etc. We do not strip generic
# parentheticals so legitimate variants like ``Wonderwall (MTV Unplugged)``
# stay distinguishable from the studio recording.
_REF_VARIANT_SUFFIX_RE = re.compile(
    r"\s*\(\s*(?:alt(?:ernate|ernative)?|ref(?:erence)?|source|src|v)\s*\d*\s*\)\s*$",
    re.IGNORECASE,
)


def _split_track_id(track: str) -> tuple[str | None, str | None]:
    """Heuristic split of an ``audfprint`` track label into (artist, title).

    Conventions we honour:

    * ``"Artist - Title.mp3"`` → ``("Artist", "Title")``.
    * ``"Artist_-_Title.wav"`` (underscored — some audfprint builds) → ``("Artist", "Title")``.
    * ``"Title.wav"`` → ``(None, "Title")``.
    * Path-like strings → take basename then strip extension.
    * Trailing alternate-reference markers (``(alt)``, ``(alt 2)``, ``(ref)``,
      ``(reference)``, ``(source)``, ``(v2)``, …) are stripped so multiple
      reference recordings for the same song share one identity. The check
      runs **after** path/extension stripping so a filename like
      ``Oasis - Wonderwall (alt 2).mp3`` produces ``("Oasis", "Wonderwall")``.
      Genuinely-different variants such as ``Wonderwall (MTV Unplugged)`` are
      left intact — only narrow operator-supplied disambiguators are folded.
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
            title = _REF_VARIANT_SUFFIX_RE.sub("", title).strip()
            return artist.strip() or None, title or None
    raw = _REF_VARIANT_SUFFIX_RE.sub("", raw).strip()
    return None, raw or None
