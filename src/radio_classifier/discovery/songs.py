"""Shazam discovery + tracklist promotion helpers.

The runtime funnel already records every Shazam hit into the ``songs`` table
with ``source='shazam'``. These helpers let the operator:

* See which songs Shazam has discovered, how often they've been heard, and
  whether they're already represented in ``data/reference/tracklist.txt`` so
  Tier 1 can match them in the future.
* Append selected discoveries to the tracklist with a provenance comment
  block, so a follow-up ``seed download`` + ``fingerprint index`` populates
  the audfprint database.

Both helpers are pure: ``list_shazam_discoveries`` is read-only against an
open :class:`BroadcastStore`, and ``promote_to_tracklist`` only appends to a
caller-supplied text file. No network, no model loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.normalize import normalize_token


# ---------------------------------------------------------------------- types
@dataclass
class DiscoveryRow:
    """One Shazam-sourced ``songs`` row enriched with play stats + tracklist status."""

    song_id: int
    artist: str | None
    title: str | None
    first_seen_utc: str
    play_count: int
    last_heard_utc: str | None
    in_tracklist: bool


@dataclass
class PromotedTrack:
    """One row appended (or skipped) by :func:`promote_to_tracklist`."""

    song_id: int
    artist: str
    title: str
    appended: bool
    reason: str = ""  # populated when appended is False


@dataclass
class PromoteResult:
    """Aggregated outcome of one :func:`promote_to_tracklist` call."""

    tracklist_path: Path
    promoted: list[PromotedTrack]

    @property
    def appended_count(self) -> int:
        return sum(1 for p in self.promoted if p.appended)

    @property
    def skipped_count(self) -> int:
        return sum(1 for p in self.promoted if not p.appended)


# -------------------------------------------------------------------- queries
def list_shazam_discoveries(
    store: BroadcastStore,
    *,
    since_utc: str | None = None,
    top_n: int = 20,
    min_plays: int = 1,
    include_indexed: bool = False,
    tracklist_path: Path | None = None,
) -> list[DiscoveryRow]:
    """Return Shazam-sourced songs ranked by recent play count.

    ``since_utc`` filters ``broadcast_events.timestamp_start >= since_utc`` for
    the play stats but never filters ``songs`` itself, so a song discovered
    before the window still shows up if it played inside the window.

    ``min_plays`` drops rows whose computed ``play_count`` is below the floor.
    ``include_indexed=False`` (default) hides rows already represented in the
    tracklist file so the operator sees only actionable discoveries.
    """
    rows = store.connection.execute(
        """
        SELECT
            s.id,
            s.artist,
            s.title,
            s.first_seen_utc,
            COUNT(e.id) AS plays,
            MAX(e.timestamp_start) AS last_heard
        FROM songs s
        LEFT JOIN broadcast_events e
            ON e.song_id = s.id
            AND e.category = 'SONG'
            AND (? IS NULL OR e.timestamp_start >= ?)
        WHERE s.source = 'shazam'
        GROUP BY s.id, s.artist, s.title, s.first_seen_utc
        ORDER BY plays DESC, last_heard DESC, s.first_seen_utc DESC
        LIMIT ?
        """,
        (since_utc, since_utc, max(1, top_n)),
    ).fetchall()

    tracklist_entries = _read_tracklist_entries(tracklist_path)

    results: list[DiscoveryRow] = []
    for r in rows:
        plays = int(r[4] or 0)
        if plays < min_plays:
            continue
        artist, title = r[1], r[2]
        in_tracklist = _tracklist_contains(tracklist_entries, artist, title)
        if in_tracklist and not include_indexed:
            continue
        results.append(
            DiscoveryRow(
                song_id=int(r[0]),
                artist=artist,
                title=title,
                first_seen_utc=str(r[3]),
                play_count=plays,
                last_heard_utc=r[5],
                in_tracklist=in_tracklist,
            )
        )
    return results


# --------------------------------------------------------------------- writer
def promote_to_tracklist(
    store: BroadcastStore,
    *,
    song_ids: list[int],
    tracklist_path: Path,
    now_iso_date: str | None = None,
) -> PromoteResult:
    """Append selected Shazam-discovered songs to ``tracklist_path``.

    Refuses to promote a ``song_id`` that:

    * does not exist in the ``songs`` table;
    * has ``source != 'shazam'`` (we don't relabel audfprint or manual rows);
    * has a ``NULL`` artist or title (no usable line to write);
    * is already present in the tracklist (exact or normalised match).

    Appends a ``# Batch N — promoted from Shazam discoveries`` comment block
    in front of the new entries so the provenance is visible in the file.
    """
    if not song_ids:
        return PromoteResult(tracklist_path=tracklist_path, promoted=[])

    existing_entries = _read_tracklist_entries(tracklist_path)
    existing_keys = {(_norm(a), _norm(t)) for a, t in existing_entries}

    promoted: list[PromotedTrack] = []
    appended_lines: list[str] = []
    seen_in_this_call: set[tuple[str, str]] = set()

    for song_id in song_ids:
        row = store.connection.execute(
            "SELECT id, artist, title, source FROM songs WHERE id = ?",
            (song_id,),
        ).fetchone()
        if row is None:
            promoted.append(
                PromotedTrack(
                    song_id=song_id,
                    artist="",
                    title="",
                    appended=False,
                    reason=f"song_id {song_id} not found",
                )
            )
            continue
        _, artist, title, source = row[0], row[1], row[2], row[3]
        if source != "shazam":
            promoted.append(
                PromotedTrack(
                    song_id=song_id,
                    artist=artist or "",
                    title=title or "",
                    appended=False,
                    reason=f"song_id {song_id} has source={source!r}; nothing to promote",
                )
            )
            continue
        if not artist or not title:
            promoted.append(
                PromotedTrack(
                    song_id=song_id,
                    artist=artist or "",
                    title=title or "",
                    appended=False,
                    reason=f"song_id {song_id} missing artist or title; skipping",
                )
            )
            continue

        key = (_norm(artist), _norm(title))
        if key in existing_keys:
            promoted.append(
                PromotedTrack(
                    song_id=song_id,
                    artist=artist,
                    title=title,
                    appended=False,
                    reason="already in tracklist",
                )
            )
            continue
        if key in seen_in_this_call:
            promoted.append(
                PromotedTrack(
                    song_id=song_id,
                    artist=artist,
                    title=title,
                    appended=False,
                    reason="duplicate in this promote call",
                )
            )
            continue
        seen_in_this_call.add(key)

        appended_lines.append(f"{artist} | {title}")
        promoted.append(
            PromotedTrack(
                song_id=song_id,
                artist=artist,
                title=title,
                appended=True,
            )
        )

    if appended_lines:
        _append_batch(tracklist_path, appended_lines, now_iso_date=now_iso_date)

    return PromoteResult(tracklist_path=tracklist_path, promoted=promoted)


# -------------------------------------------------------------------- helpers
def _read_tracklist_entries(path: Path | None) -> list[tuple[str, str]]:
    """Return ``(artist, title)`` pairs from a tracklist file.

    Skips blank lines and ``#`` comments. Returns ``[]`` if ``path`` is ``None``
    or missing.
    """
    if path is None or not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            continue
        artist, _, title = line.partition("|")
        entries.append((artist.strip(), title.strip()))
    return entries


def _tracklist_contains(
    entries: list[tuple[str, str]], artist: str | None, title: str | None
) -> bool:
    target = (_norm(artist), _norm(title))
    if target == (None, None):
        return False
    for a, t in entries:
        if (_norm(a), _norm(t)) == target:
            return True
    return False


def _norm(value: str | None) -> str | None:
    """Wrapper around :func:`normalize_token` so ``None`` flows through cleanly."""
    return normalize_token(value) if value else None


def _append_batch(
    tracklist_path: Path,
    lines: list[str],
    *,
    now_iso_date: str | None,
) -> None:
    """Append a comment-headed batch to ``tracklist_path``.

    The file must already exist; we don't create new tracklist files implicitly
    because the operator-curated layout (header comments, batch comments) is
    something we want to inherit, not stomp.
    """
    if not tracklist_path.exists():
        raise FileNotFoundError(
            f"tracklist file does not exist; create it first: {tracklist_path}"
        )
    iso_date = now_iso_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    existing = tracklist_path.read_text(encoding="utf-8")
    prefix = "" if existing.endswith("\n") or existing == "" else "\n"
    block = [
        f"{prefix}",
        f"# Batch — promoted from Shazam discoveries on {iso_date}",
    ]
    block.extend(lines)
    block.append("")  # trailing newline
    with tracklist_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(block))
