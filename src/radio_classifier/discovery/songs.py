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
import re
from typing import Iterator

from radio_classifier.persistence.broadcast_store import (
    BroadcastStore,
    _display_key,
    _prefer_display_value,
)
from radio_classifier.segments.normalize import normalize_token


LOW_CONFIDENCE_PLAY_THRESHOLD = 3


# ---------------------------------------------------------------- dedupe types
@dataclass
class _DedupeMember:
    """One ``songs`` row in a dedupe group."""

    song_id: int
    artist: str | None
    title: str | None
    source: str
    audfprint_track_id: str | None
    event_count: int


@dataclass
class DedupeGroup:
    """Same-song rows that should fold into a single survivor.

    Members are sorted: the **first** member is the survivor (preferred when
    folding), every other member is a loser whose ``broadcast_events`` will
    be re-pointed at the survivor before the loser row is deleted.

    Survivor selection rules, in priority order:

    1.  A row with a non-NULL ``audfprint_track_id`` always beats one without
        (audfprint matches are deterministic; Shazam matches are best-effort).
    2.  Within that tier, ``source = 'audfprint'`` beats ``'shazam'`` /
        anything else.
    3.  Within that tier, the row with the **most** existing ``event_count``
        wins (preserving the most history with the fewest rewrites).
    4.  Final tie-breaker: lowest ``song_id`` (earliest discovery).
    """

    key: tuple[str, str]
    members: list[_DedupeMember]

    @property
    def survivor(self) -> _DedupeMember:
        return self.members[0]

    @property
    def losers(self) -> list[_DedupeMember]:
        return self.members[1:]


@dataclass
class DedupeReport:
    """Result of one :func:`dedupe_songs` invocation."""

    groups: list[DedupeGroup]
    events_repointed: int
    rows_deleted: int
    dry_run: bool

    @property
    def collapsed_pairs(self) -> int:
        """How many ``songs`` rows would disappear (sum of group sizes - 1)."""
        return sum(len(g.losers) for g in self.groups)


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
    needs_review: bool


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
                needs_review=plays < LOW_CONFIDENCE_PLAY_THRESHOLD,
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


_FEATURE_SUFFIX_RE = re.compile(
    r"\s*(?:\(|\[)\s*(?:feat\.?|featuring|ft\.?|with)\b.*?(?:\)|\])\s*$",
    re.IGNORECASE,
)


def _norm(value: str | None) -> str | None:
    """Normalize a tracklist/discovery field for containment checks.

    Shazam often returns typographic apostrophes or a featured-artist suffix
    while our curated tracklist stores simpler titles. For "already indexed?"
    checks those should compare equal, so ``Who's`` == ``who’s`` and
    ``Go Away (feat. Best Coast)`` == ``Go Away``.

    Feature suffixes are stripped here and not in ``_display_key``: the
    broadcast store's identity matching deliberately treats ``Go Away`` and
    ``Go Away (feat. Best Coast)`` as the same song only at the tracklist
    layer, not in the broadcast event store.
    """
    if not value:
        return None
    cleaned = _FEATURE_SUFFIX_RE.sub("", value)
    key = _display_key(cleaned)
    if not key:
        return None
    return normalize_token(key)


def _normalize_dedupe_key(artist: str | None, title: str | None) -> tuple[str, str]:
    """Same identity rule used by ``BroadcastStore.upsert_song``.

    Delegates to :func:`broadcast_store._display_key` so dedupe folds the
    same set of songs the upsert path would have collapsed if it had seen
    them in a single pass (typographic-vs-ASCII apostrophes, filename
    underscore artifacts, casing/whitespace drift).

    Returning a 2-tuple of strings (not Optional) lets the caller use the
    value directly as a dict key without worrying about Nones — empty
    strings are still distinguishable from real content.
    """
    return (_display_key(artist), _display_key(title))


def _iter_dedupe_groups(store: BroadcastStore) -> Iterator[DedupeGroup]:
    """Yield every dedupe group with at least two members.

    The query joins ``broadcast_events`` so the survivor-picking heuristic can
    factor in how many existing rows depend on each candidate (the row with
    the most events is cheapest to keep, requiring the fewest re-points).
    """
    rows = store.connection.execute(
        """
        SELECT
            s.id,
            s.artist,
            s.title,
            s.source,
            s.audfprint_track_id,
            (SELECT COUNT(*) FROM broadcast_events e WHERE e.song_id = s.id) AS event_count
        FROM songs s
        ORDER BY s.id ASC
        """
    ).fetchall()

    buckets: dict[tuple[str, str], list[_DedupeMember]] = {}
    for r in rows:
        member = _DedupeMember(
            song_id=int(r[0]),
            artist=r[1],
            title=r[2],
            source=str(r[3] or ""),
            audfprint_track_id=r[4],
            event_count=int(r[5] or 0),
        )
        buckets.setdefault(_normalize_dedupe_key(member.artist, member.title), []).append(member)

    for key, members in buckets.items():
        if len(members) < 2:
            continue
        if not key[0] or not key[1]:
            # Don't fold rows that have no artist/title — those are useless
            # records that should be deleted by hand, not auto-merged.
            continue
        ranked = sorted(
            members,
            key=lambda m: (
                m.audfprint_track_id is None,        # has track_id → first
                0 if m.source == "audfprint" else 1, # audfprint source → first
                -m.event_count,                       # most events → first
                m.song_id,                            # earliest id → first
            ),
        )
        yield DedupeGroup(key=key, members=ranked)


def dedupe_songs(store: BroadcastStore, *, dry_run: bool = False) -> DedupeReport:
    """Fold ``songs`` rows that share a normalized (artist, title) identity.

    See :class:`DedupeGroup` for survivor-selection rules. The actual work,
    in transactional order:

    1.  Re-point ``broadcast_events.song_id`` from each loser to the
        survivor. (The :class:`DedupeGroup` ordering guarantees the survivor
        is the row that already has an ``audfprint_track_id`` if any of the
        siblings did, so future Tier 1 matches stay resolvable.)
    2.  If the survivor's ``audfprint_track_id`` is still NULL but one of the
        losers had one, copy it over. (Belt-and-braces — the survivor-pick
        usually already satisfies this, but we double-check so the dedupe
        is idempotent and order-of-operations-safe.)
    3.  Upgrade the survivor's displayed artist/title to the cleanest variant
        in the group. Audfprint reference filenames sanitize ``'`` to ``_``
        and that artifact propagates into ``songs.title``; if a Shazam or
        curated row in the same group carries a cleaner spelling we adopt
        it for display while keeping the audfprint identity (track id,
        source) intact.
    4.  Delete the loser ``songs`` rows.

    Pass ``dry_run=True`` to compute and return the report without writing.
    """
    groups = list(_iter_dedupe_groups(store))
    repointed = 0
    deleted = 0
    if not dry_run and groups:
        conn = store.connection
        conn.execute("BEGIN")
        try:
            for group in groups:
                survivor = group.survivor
                loser_ids = [m.song_id for m in group.losers]
                placeholders = ",".join("?" * len(loser_ids))
                best_artist = survivor.artist
                best_title = survivor.title
                for loser in group.losers:
                    best_artist = _prefer_display_value(best_artist, loser.artist)
                    best_title = _prefer_display_value(best_title, loser.title)
                cur = conn.execute(
                    f"UPDATE broadcast_events SET song_id = ?, artist = COALESCE(artist, ?), "
                    f"track_title = COALESCE(track_title, ?) "
                    f"WHERE song_id IN ({placeholders})",
                    [survivor.song_id, best_artist, best_title, *loser_ids],
                )
                repointed += cur.rowcount or 0
                if survivor.audfprint_track_id is None:
                    rescue = next(
                        (m for m in group.losers if m.audfprint_track_id is not None),
                        None,
                    )
                    if rescue is not None:
                        conn.execute(
                            "UPDATE songs SET audfprint_track_id = ?, source = ? WHERE id = ?",
                            (rescue.audfprint_track_id, "audfprint", survivor.song_id),
                        )
                if best_artist != survivor.artist or best_title != survivor.title:
                    conn.execute(
                        "UPDATE songs SET artist = ?, title = ? WHERE id = ?",
                        (best_artist, best_title, survivor.song_id),
                    )
                cur = conn.execute(
                    f"DELETE FROM songs WHERE id IN ({placeholders})",
                    loser_ids,
                )
                deleted += cur.rowcount or 0
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return DedupeReport(
        groups=groups,
        events_repointed=repointed,
        rows_deleted=deleted,
        dry_run=dry_run,
    )


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
