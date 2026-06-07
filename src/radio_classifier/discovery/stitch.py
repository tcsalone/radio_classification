"""Stitch SONG events split across capture-block boundaries.

Each 30-minute capture block is classified by a *separate* process with its
own :class:`SegmentReducer`, so a song that straddles a block boundary is
emitted as two adjacent ``broadcast_events`` rows: the first block's tail and
the next block's head. They share a ``song_id`` and meet exactly
(``A.timestamp_end == B.timestamp_start``), but no single reducer ever saw
both windows, so they never merged.

This pass runs *after* classification and folds those contiguous same-song
fragments back into one event. It is deliberately conservative:

* only ``SONG`` rows with a non-null ``song_id`` are eligible (unknown-music
  ``song_id IS NULL`` rows are left alone — we can't prove two are the same
  song),
* fragments must be contiguous within ``max_gap_seconds`` (default ~2s; block
  boundaries meet exactly, the tolerance only absorbs sub-second rounding),
* a real second airing of the same song is separated by other
  events/airtime, so its rows are *not* contiguous and are correctly kept
  distinct.

The earliest row in a contiguous run survives; its ``timestamp_end`` is
extended to the last fragment's end, ``duration`` is recomputed, and the
highest available ``confidence`` is kept. The absorbed rows are deleted
(``brand_mentions`` cascade, though SONG rows rarely carry any).
"""

from __future__ import annotations

from dataclasses import dataclass

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.reducer import _parse_utc, duration_seconds

# Block boundaries meet exactly (block N end == block N+1 start), so 0 would
# suffice; a small tolerance absorbs sub-second timestamp rounding without
# bridging genuine gaps (a re-airing is separated by minutes of other airtime).
_DEFAULT_MAX_GAP_SECONDS = 2.0


@dataclass
class StitchGroup:
    """One contiguous run of same-song fragments folded into the survivor."""

    song_id: int
    artist: str | None
    title: str | None
    survivor_event_id: int
    absorbed_event_ids: list[int]
    start_utc: str
    end_utc: str
    spanned_capture_runs: bool


@dataclass
class StitchReport:
    groups: list[StitchGroup]
    events_scanned: int
    dry_run: bool

    @property
    def events_absorbed(self) -> int:
        return sum(len(g.absorbed_event_ids) for g in self.groups)


@dataclass
class _Event:
    rowid: int
    song_id: int
    start: str
    end: str | None
    confidence: float | None
    artist: str | None
    title: str | None
    run_id: int | None


def _gap_seconds(prev_end: str | None, next_start: str | None) -> float:
    if not prev_end or not next_start:
        return 0.0
    return (_parse_utc(next_start) - _parse_utc(prev_end)).total_seconds()


def _load_song_events(
    store: BroadcastStore, since_utc: str | None, until_utc: str | None
) -> list[_Event]:
    clauses = ["category = 'SONG'", "song_id IS NOT NULL", "timestamp_end IS NOT NULL"]
    args: list[object] = []
    if since_utc is not None:
        clauses.append("timestamp_start >= ?")
        args.append(since_utc)
    if until_utc is not None:
        clauses.append("timestamp_start < ?")
        args.append(until_utc)
    rows = store.connection.execute(
        "SELECT id, song_id, timestamp_start, timestamp_end, confidence, artist, "
        "track_title, capture_run_id FROM broadcast_events WHERE "
        + " AND ".join(clauses)
        + " ORDER BY timestamp_start ASC, id ASC",
        args,
    ).fetchall()
    return [
        _Event(
            rowid=int(r[0]),
            song_id=int(r[1]),
            start=str(r[2]),
            end=r[3],
            confidence=r[4],
            artist=r[5],
            title=r[6],
            run_id=r[7],
        )
        for r in rows
    ]


def stitch_song_plays(
    store: BroadcastStore,
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    max_gap_seconds: float = _DEFAULT_MAX_GAP_SECONDS,
    dry_run: bool = True,
) -> StitchReport:
    """Fold contiguous same-song SONG fragments into one event.

    Walks the SONG events in chronological order and groups maximal runs where
    each row's ``song_id`` matches the previous and the inter-row gap is within
    ``max_gap_seconds``. Each run of length > 1 collapses into its earliest row.
    """
    events = _load_song_events(store, since_utc, until_utc)
    groups: list[StitchGroup] = []

    i = 0
    n = len(events)
    while i < n:
        run = [events[i]]
        j = i + 1
        while j < n:
            prev = events[j - 1]
            cur = events[j]
            if cur.song_id != prev.song_id:
                break
            if prev.end is None:
                break
            gap = _gap_seconds(prev.end, cur.start)
            if gap < 0 or gap > max_gap_seconds:
                break
            run.append(cur)
            j += 1

        if len(run) > 1:
            survivor = run[0]
            absorbed = run[1:]
            last = run[-1]
            confidences = [e.confidence for e in run if e.confidence is not None]
            best_conf = max(confidences) if confidences else None
            group = StitchGroup(
                song_id=survivor.song_id,
                artist=survivor.artist,
                title=survivor.title,
                survivor_event_id=survivor.rowid,
                absorbed_event_ids=[e.rowid for e in absorbed],
                start_utc=survivor.start,
                end_utc=str(last.end),
                spanned_capture_runs=len({e.run_id for e in run}) > 1,
            )
            groups.append(group)

            if not dry_run:
                new_duration = duration_seconds(survivor.start, str(last.end))
                store.connection.execute(
                    "UPDATE broadcast_events SET timestamp_end = ?, duration = ?, "
                    "confidence = ? WHERE id = ?",
                    (str(last.end), new_duration, best_conf, survivor.rowid),
                )
                store.connection.executemany(
                    "DELETE FROM broadcast_events WHERE id = ?",
                    [(e.rowid,) for e in absorbed],
                )
                store.connection.commit()

        i = j

    return StitchReport(groups=groups, events_scanned=n, dry_run=dry_run)
