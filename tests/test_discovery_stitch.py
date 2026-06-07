"""Cross-block song stitching: contiguous same-song fragments fold into one."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from radio_classifier.discovery import stitch_song_plays
from radio_classifier.persistence import BroadcastStore
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _song_event(
    store: BroadcastStore,
    *,
    start: datetime,
    seconds: int,
    song_id: int,
    artist: str,
    title: str,
    confidence: float | None = None,
    run_id: int | None = None,
) -> int:
    return store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(start + timedelta(seconds=seconds)),
            category=BroadcastCategory.SONG,
            song_id=song_id,
            artist=artist,
            track_title=title,
            confidence=confidence,
        ),
        capture_run_id=run_id,
    )


def _seed_song(store: BroadcastStore, song_id: int, artist: str, title: str) -> None:
    store.connection.execute(
        "INSERT INTO songs (id, artist, title, source) VALUES (?, ?, ?, 'audfprint')",
        (song_id, artist, title),
    )
    store.connection.commit()


def _seed_run(store: BroadcastStore, run_pk: int) -> None:
    store.connection.execute(
        "INSERT INTO capture_runs (id, run_id, started_utc, pipeline_version) "
        "VALUES (?, ?, '2026-06-06T00:00:00.000Z', 'test')",
        (run_pk, f"run-{run_pk}"),
    )
    store.connection.commit()


def test_contiguous_same_song_fragments_are_stitched(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 8, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 57, "Modest Mouse", "Float On")
        _seed_run(store, 444)
        # Block 1 tail: ends exactly at the block boundary.
        e1 = _song_event(
            store, start=base, seconds=120, song_id=57, artist="Modest Mouse",
            title="Float On", confidence=140.0, run_id=444,
        )
        # Block 2 head: starts exactly at e1's end (the 30-min block edge).
        e2 = _song_event(
            store, start=base + timedelta(seconds=120), seconds=90, song_id=57,
            artist="Modest Mouse", title="Float On", confidence=131.0, run_id=444,
        )

        report = stitch_song_plays(store, dry_run=False)
        assert len(report.groups) == 1
        assert report.events_absorbed == 1
        g = report.groups[0]
        assert g.survivor_event_id == e1
        assert g.absorbed_event_ids == [e2]

        rows = store.connection.execute(
            "SELECT id, timestamp_start, timestamp_end, duration, confidence "
            "FROM broadcast_events WHERE category='SONG' ORDER BY id"
        ).fetchall()
        assert len(rows) == 1, "fragments must collapse to a single event"
        survivor = rows[0]
        assert survivor[0] == e1
        assert survivor[1] == _iso(base)
        assert survivor[2] == _iso(base + timedelta(seconds=210))
        assert survivor[3] == 210.0  # duration recomputed across the boundary
        assert survivor[4] == 140.0  # highest confidence kept


def test_three_way_chain_folds_into_one(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 9, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 5, "Weezer", "Hash Pipe")
        _song_event(store, start=base, seconds=100, song_id=5, artist="Weezer", title="Hash Pipe")
        _song_event(store, start=base + timedelta(seconds=100), seconds=100, song_id=5,
                    artist="Weezer", title="Hash Pipe")
        _song_event(store, start=base + timedelta(seconds=200), seconds=80, song_id=5,
                    artist="Weezer", title="Hash Pipe")

        report = stitch_song_plays(store, dry_run=False)
        assert len(report.groups) == 1
        assert report.events_absorbed == 2
        rows = store.connection.execute(
            "SELECT timestamp_end, duration FROM broadcast_events WHERE category='SONG'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == _iso(base + timedelta(seconds=280))
        assert rows[0][1] == 280.0


def test_different_songs_are_not_stitched(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 10, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 1, "A", "song-a")
        _seed_song(store, 2, "B", "song-b")
        _song_event(store, start=base, seconds=120, song_id=1, artist="A", title="song-a")
        _song_event(store, start=base + timedelta(seconds=120), seconds=120, song_id=2,
                    artist="B", title="song-b")

        report = stitch_song_plays(store, dry_run=False)
        assert report.groups == []
        n = store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE category='SONG'"
        ).fetchone()[0]
        assert n == 2


def test_same_song_with_real_gap_is_kept_separate(tmp_path: Path) -> None:
    """A re-airing separated by a gap (other airtime) must stay two plays."""
    base = datetime(2026, 6, 6, 11, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 9, "Nirvana", "Heart-Shaped Box")
        _song_event(store, start=base, seconds=120, song_id=9, artist="Nirvana",
                    title="Heart-Shaped Box")
        # 5-minute gap → a separate airing, not a block-boundary fragment.
        _song_event(store, start=base + timedelta(seconds=420), seconds=120, song_id=9,
                    artist="Nirvana", title="Heart-Shaped Box")

        report = stitch_song_plays(store, dry_run=False)
        assert report.groups == []
        n = store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE category='SONG'"
        ).fetchone()[0]
        assert n == 2


def test_null_song_id_events_are_ignored(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 12, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        # Two contiguous unknown-music events (song_id NULL) must not merge —
        # we can't prove they are the same song.
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=120)),
                category=BroadcastCategory.SONG,
            )
        )
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base + timedelta(seconds=120)),
                timestamp_end=_iso(base + timedelta(seconds=240)),
                category=BroadcastCategory.SONG,
            )
        )
        report = stitch_song_plays(store, dry_run=False)
        assert report.groups == []
        n = store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE category='SONG'"
        ).fetchone()[0]
        assert n == 2


def test_dry_run_does_not_mutate(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 13, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 7, "blink-182", "Dammit")
        _song_event(store, start=base, seconds=120, song_id=7, artist="blink-182", title="Dammit")
        _song_event(store, start=base + timedelta(seconds=120), seconds=120, song_id=7,
                    artist="blink-182", title="Dammit")

        report = stitch_song_plays(store, dry_run=True)
        assert len(report.groups) == 1
        assert report.dry_run is True
        n = store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE category='SONG'"
        ).fetchone()[0]
        assert n == 2, "dry-run must not delete anything"


def test_cross_capture_run_fragments_flagged(tmp_path: Path) -> None:
    base = datetime(2026, 6, 6, 14, 0, 0, tzinfo=timezone.utc)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _seed_song(store, 11, "Tove Lo", "Habits")
        _seed_run(store, 444)
        _seed_run(store, 445)
        _song_event(store, start=base, seconds=120, song_id=11, artist="Tove Lo",
                    title="Habits", run_id=444)
        _song_event(store, start=base + timedelta(seconds=120), seconds=120, song_id=11,
                    artist="Tove Lo", title="Habits", run_id=445)
        report = stitch_song_plays(store, dry_run=False)
        assert len(report.groups) == 1
        assert report.groups[0].spanned_capture_runs is True
