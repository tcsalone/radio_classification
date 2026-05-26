"""Tests for Shazam discovery listing and tracklist promotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radio_classifier.discovery import (
    list_shazam_discoveries,
    promote_to_tracklist,
)
from radio_classifier.persistence import BroadcastStore
from radio_classifier.reports import format_discoveries
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _seed_shazam_db(tmp_path: Path) -> BroadcastStore:
    store = BroadcastStore(tmp_path / "rc.db")
    now = datetime.now(tz=timezone.utc)

    djo = store.upsert_song(
        artist="Djo", title="End of Beginning", source="shazam"
    )
    float_on = store.upsert_song(
        artist="Modest Mouse", title="Float On", source="shazam"
    )
    guns = store.upsert_song(
        artist="Green Day", title="21 Guns", source="shazam"
    )
    audfprint = store.upsert_song(
        artist="Nirvana", title="Smells Like Teen Spirit", source="audfprint"
    )

    def song_play(song_id: int, artist: str, title: str, *, minutes_ago: int) -> None:
        start = now - timedelta(minutes=minutes_ago)
        end = start + timedelta(seconds=180)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.SONG,
                artist=artist,
                track_title=title,
                song_id=song_id,
            )
        )

    song_play(djo, "Djo", "End of Beginning", minutes_ago=30)
    song_play(djo, "Djo", "End of Beginning", minutes_ago=25)
    song_play(djo, "Djo", "End of Beginning", minutes_ago=20)
    song_play(float_on, "Modest Mouse", "Float On", minutes_ago=15)
    song_play(guns, "Green Day", "21 Guns", minutes_ago=10)
    song_play(audfprint, "Nirvana", "Smells Like Teen Spirit", minutes_ago=5)
    return store


def _write_tracklist(path: Path, lines: list[str]) -> None:
    path.write_text(
        "# test tracklist\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_list_shazam_discoveries_ranks_by_plays(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            tracklist_path=tracklist,
        )
        assert len(rows) == 2
        assert rows[0].artist == "Djo"
        assert rows[0].play_count == 3
        assert rows[0].in_tracklist is False
        assert rows[1].artist == "Modest Mouse"
        assert rows[1].play_count == 1
    finally:
        store.close()


def test_list_shazam_discoveries_hides_indexed_by_default(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            include_indexed=True,
            tracklist_path=tracklist,
        )
        assert any(r.artist == "Green Day" and r.in_tracklist for r in rows)
        hidden = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            include_indexed=False,
            tracklist_path=tracklist,
        )
        assert all(r.artist != "Green Day" for r in hidden)
    finally:
        store.close()


def test_list_shazam_discoveries_min_plays_filter(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            min_plays=2,
            tracklist_path=None,
        )
        assert len(rows) == 1
        assert rows[0].artist == "Djo"
    finally:
        store.close()


def test_format_discoveries_renders_table(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(store, since_utc=since, top_n=5, tracklist_path=None)
        text = format_discoveries(rows)
        assert "Djo" in text
        assert "missing" in text
        assert "plays" in text
    finally:
        store.close()


def test_promote_appends_new_tracks(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    djo_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Djo'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert result.appended_count == 1
        body = tracklist.read_text(encoding="utf-8")
        assert "Djo | End of Beginning" in body
        assert "promoted from Shazam discoveries on 2026-05-25" in body
    finally:
        store.close()


def test_promote_skips_already_in_tracklist(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    guns_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Green Day'"
    ).fetchone()[0]
    before = tracklist.read_text(encoding="utf-8")
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[guns_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert result.appended_count == 0
        assert result.skipped_count == 1
        assert result.promoted[0].reason == "already in tracklist"
        assert tracklist.read_text(encoding="utf-8") == before
    finally:
        store.close()


def test_promote_skips_normalized_duplicate(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["green day | 21 guns"])
    guns_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Green Day'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[guns_id],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "already in tracklist" in result.promoted[0].reason
    finally:
        store.close()


def test_promote_refuses_non_shazam_source(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    nirvana_id = store.connection.execute(
        "SELECT id FROM songs WHERE source = 'audfprint'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[nirvana_id],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "source='audfprint'" in result.promoted[0].reason
    finally:
        store.close()


def test_promote_is_idempotent(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    djo_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Djo'"
    ).fetchone()[0]
    try:
        first = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        second = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert first.appended_count == 1
        assert second.appended_count == 0
        assert second.promoted[0].reason == "already in tracklist"
        assert tracklist.read_text(encoding="utf-8").count("Djo | End of Beginning") == 1
    finally:
        store.close()


def test_promote_unknown_song_id(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[9999],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "not found" in result.promoted[0].reason
    finally:
        store.close()
