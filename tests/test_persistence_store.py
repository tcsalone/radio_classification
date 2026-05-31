"""BroadcastStore (schema v2) tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from radio_classifier.persistence import BroadcastStore
from radio_classifier.persistence.coordinator import persist_finalize, persist_input
from radio_classifier.segments import (
    BroadcastCategory,
    SegmentInput,
    SegmentKey,
    SegmentReducer,
    SegmentTransition,
)


def test_store_creates_schema_v2(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        assert store.schema_version() == "2"
        # CHECK constraint should now reject the old 3-class enum value.
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO broadcast_events "
                "(timestamp_start, category) VALUES (?, ?)",
                ("2020-01-01T00:00:00.000Z", "MUSIC"),
            )


def test_apply_transition_round_trips(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        brand_id = store.upsert_brand("Geico")
        song_id = store.upsert_song(artist="Taylor Swift", title="Anti-Hero")
        t = SegmentTransition(
            timestamp_start="2020-01-01T00:00:00.000Z",
            timestamp_end="2020-01-01T00:00:20.000Z",
            category=BroadcastCategory.COMMERCIAL,
            artist=None,
            track_title=None,
            brand_name="Geico",
            song_id=None,
            commercial_id=None,
            brand_id=brand_id,
            transcript_excerpt="Save 15%",
            confidence=0.9,
        )
        event_id = store.apply_transition(t)
        assert event_id > 0
        row = store.connection.execute(
            "SELECT category, duration, brand_id, brand_name FROM broadcast_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert row == ("COMMERCIAL", 20.0, brand_id, "Geico")
        # Sanity: songs row exists too.
        assert store.upsert_song(artist="Taylor Swift", title="Anti-Hero") == song_id


def test_upsert_brand_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        a = store.upsert_brand("Toyota")
        b = store.upsert_brand("Toyota")
        assert a == b


def test_coordinator_persists_on_key_change(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        reducer = SegmentReducer()
        k_dj = SegmentKey(BroadcastCategory.DJ)
        k_com = SegmentKey(BroadcastCategory.COMMERCIAL, brand_key="geico", commercial_id=None)
        assert persist_input(
            reducer,
            store,
            SegmentInput("2020-01-01T00:00:00.000Z", k_dj),
        ) == []
        ids = persist_input(
            reducer,
            store,
            SegmentInput("2020-01-01T00:00:30.000Z", k_com, brand_name="Geico"),
        )
        assert len(ids) == 1
        # finalize closes the open COMMERCIAL segment
        final_ids = persist_finalize(
            reducer,
            store,
            last_window_start_utc="2020-01-01T00:00:30.000Z",
            window_seconds=10.0,
        )
        assert len(final_ids) == 1
        rows = store.connection.execute(
            "SELECT category, brand_name, brand_id FROM broadcast_events ORDER BY id"
        ).fetchall()
        assert [r[0] for r in rows] == ["DJ", "COMMERCIAL"]
        # brand_id was resolved at persistence time
        assert rows[1][1] == "Geico"
        assert rows[1][2] is not None


def test_upsert_song_collapses_shazam_then_audfprint_into_one_row(tmp_path: Path) -> None:
    """A Shazam discovery followed by an audfprint match for the same track
    must NOT create a second row. The existing row should absorb the
    audfprint_track_id and have its source upgraded.

    Regression for the 2026-05-30 dedupe sweep: prior to this fix, the upsert
    keyed on (artist, title, source), so a Shazam row + a later audfprint
    match (with the same artist/title but different source) coexisted and
    every report counted the song twice.
    """
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        first = store.upsert_song(artist="Garbage", title="Only Happy When It Rains", source="shazam")
        second = store.upsert_song(
            artist="Garbage",
            title="Only Happy When It Rains",
            audfprint_track_id="data/reference/songs/Garbage - Only Happy When It Rains.mp3",
            source="audfprint",
        )
        assert first == second, "shazam+audfprint for the same song must reuse one row"

        row = store.connection.execute(
            "SELECT audfprint_track_id, source FROM songs WHERE id = ?", (first,)
        ).fetchone()
        assert row[0] == "data/reference/songs/Garbage - Only Happy When It Rains.mp3"
        assert row[1] == "audfprint", "source should be upgraded once we have a reference file"

        # And there's only one Garbage row, full stop.
        n = store.connection.execute(
            "SELECT COUNT(*) FROM songs WHERE artist = 'Garbage'"
        ).fetchone()[0]
        assert n == 1


def test_upsert_song_is_case_insensitive(tmp_path: Path) -> None:
    """Trivial casing/whitespace differences between Shazam and audfprint
    titles must not produce two rows."""
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        a = store.upsert_song(artist="Evanescence", title="Bring Me to Life", source="shazam")
        b = store.upsert_song(
            artist="evanescence",
            title="Bring Me To Life",  # casing differs
            audfprint_track_id="evan.mp3",
            source="audfprint",
        )
        c = store.upsert_song(
            artist="  Evanescence  ",  # extra whitespace
            title="  bring me to life  ",
            source="shazam",
        )
        assert a == b == c

        n = store.connection.execute(
            "SELECT COUNT(*) FROM songs WHERE LOWER(artist) = 'evanescence'"
        ).fetchone()[0]
        assert n == 1


def test_upsert_song_prefers_canonical_artist_display_casing(tmp_path: Path) -> None:
    """Known all-caps Shazam artist artifacts should be cleaned on write.

    ``LINKIN PARK`` was observed in the 12-hour reports because Shazam often
    returns the artist in all caps. The row should still dedupe
    case-insensitively, but the stored display name should be the canonical
    presentation.
    """
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        first = store.upsert_song(artist="LINKIN PARK", title="Somewhere I Belong", source="shazam")
        second = store.upsert_song(
            artist="Linkin Park",
            title="Somewhere I Belong",
            audfprint_track_id="linkin.mp3",
            source="audfprint",
        )
        assert first == second

        row = store.connection.execute(
            "SELECT artist, title, source, audfprint_track_id FROM songs WHERE id = ?",
            (first,),
        ).fetchone()
        assert row[0] == "Linkin Park"
        assert row[1] == "Somewhere I Belong"
        assert row[2] == "audfprint"
        assert row[3] == "linkin.mp3"


def test_upsert_song_keeps_legitimate_artist_acronyms(tmp_path: Path) -> None:
    """Do not title-case every all-caps artist, because names like AFI are real."""
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        song_id = store.upsert_song(artist="AFI", title="Miss Murder", source="shazam")
        row = store.connection.execute(
            "SELECT artist, title FROM songs WHERE id = ?",
            (song_id,),
        ).fetchone()
        assert row[0] == "AFI"
        assert row[1] == "Miss Murder"


def test_upsert_song_does_not_clobber_existing_audfprint_track_id(tmp_path: Path) -> None:
    """If an existing row already has an audfprint_track_id, a subsequent
    Shazam-only upsert must NOT wipe it back to NULL."""
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        first = store.upsert_song(
            artist="Nirvana", title="Lithium", audfprint_track_id="ref.mp3", source="audfprint"
        )
        second = store.upsert_song(artist="Nirvana", title="Lithium", source="shazam")
        assert first == second
        row = store.connection.execute(
            "SELECT audfprint_track_id, source FROM songs WHERE id = ?", (first,)
        ).fetchone()
        assert row[0] == "ref.mp3"
        assert row[1] == "audfprint"


def test_brand_mention_insert(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        brand_id = store.upsert_brand("Toyota")
        t = SegmentTransition(
            timestamp_start="2020-01-01T00:00:00.000Z",
            timestamp_end="2020-01-01T00:00:30.000Z",
            category=BroadcastCategory.DJ,
        )
        event_id = store.apply_transition(t)
        store.insert_brand_mention(
            segment_id=event_id,
            brand_id=brand_id,
            mention_type="dj_shoutout",
            heard_utc="2020-01-01T00:00:15.000Z",
        )
        rows = store.connection.execute(
            "SELECT mention_type FROM brand_mentions WHERE segment_id = ?", (event_id,)
        ).fetchall()
        assert rows == [("dj_shoutout",)]
