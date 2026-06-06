"""BroadcastStore schema and persistence tests."""

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


def _create_minimal_v2_db(db: Path) -> None:
    """Create just enough of schema v2 to exercise the v3 migration path."""

    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO schema_meta (key, value) VALUES ('version', '2');

            CREATE TABLE broadcast_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_start TEXT NOT NULL,
                timestamp_end TEXT,
                duration REAL,
                category TEXT NOT NULL CHECK (
                    category IN ('SONG', 'DJ', 'COMMERCIAL', 'STATION', 'PSA_NEWS')
                ),
                song_id INTEGER,
                commercial_id INTEGER,
                brand_id INTEGER,
                artist TEXT,
                track_title TEXT,
                brand_name TEXT,
                transcript_excerpt TEXT,
                confidence REAL,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            );

            INSERT INTO broadcast_events (
                timestamp_start, timestamp_end, duration, category, artist, track_title
            ) VALUES
                ('2026-05-31T07:00:00.000Z', '2026-05-31T07:03:00.000Z', 180.0,
                 'SONG', 'In Color', 'Headlights'),
                ('2026-05-31T07:03:00.000Z', '2026-05-31T07:03:30.000Z', 30.0,
                 'COMMERCIAL', NULL, NULL);
            """
        )


def test_store_creates_schema_v4(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        assert store.schema_version() == "4"
        columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(broadcast_events)").fetchall()
        }
        assert "capture_run_id" in columns
        song_columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(songs)").fetchall()
        }
        assert "release_date" in song_columns
        assert store.connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'capture_runs'"
        ).fetchone()[0] == 1
        # CHECK constraint should now reject the old 3-class enum value.
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO broadcast_events "
                "(timestamp_start, category) VALUES (?, ?)",
                ("2020-01-01T00:00:00.000Z", "MUSIC"),
            )


def test_store_migrates_v2_to_v4_and_backfills_legacy_run(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    _create_minimal_v2_db(db)

    with BroadcastStore(db) as store:
        assert store.schema_version() == "4"
        columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(broadcast_events)").fetchall()
        }
        assert "capture_run_id" in columns
        song_columns = {
            row[1]
            for row in store.connection.execute("PRAGMA table_info(songs)").fetchall()
        }
        assert "release_date" in song_columns

        run = store.connection.execute(
            """
            SELECT id, run_id, started_utc, ended_utc, pipeline_version
            FROM capture_runs
            """
        ).fetchone()
        assert run[1:] == (
            "legacy_pre_v3",
            "2026-05-31T07:00:00.000Z",
            "2026-05-31T07:03:30.000Z",
            "unknown",
        )
        assert store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE capture_run_id = ?",
            (run[0],),
        ).fetchone()[0] == 2

    # Reopening should be idempotent: no duplicate capture_run rows, no errors.
    with BroadcastStore(db) as store:
        assert store.schema_version() == "4"
        assert store.connection.execute("SELECT COUNT(*) FROM capture_runs").fetchone()[0] == 1
        assert store.connection.execute("SELECT COUNT(*) FROM broadcast_events").fetchone()[0] == 2


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


def test_capture_run_lifecycle_and_event_link(tmp_path: Path) -> None:
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        run_id = store.open_capture_run(
            run_id="continuous_test",
            started_utc="2026-06-01T00:00:00.000Z",
            pipeline_version="0.3.0+test",
            host="test-host",
            notes="smoke",
        )
        assert store.open_capture_run(
            run_id="continuous_test",
            started_utc="2026-06-01T00:00:00.000Z",
            pipeline_version="0.3.0+test",
        ) == run_id

        event_id = store.apply_transition(
            SegmentTransition(
                timestamp_start="2026-06-01T00:00:00.000Z",
                timestamp_end="2026-06-01T00:03:00.000Z",
                category=BroadcastCategory.SONG,
                artist="Radiohead",
                track_title="Karma Police",
            ),
            capture_run_id=run_id,
        )
        assert store.connection.execute(
            "SELECT capture_run_id FROM broadcast_events WHERE id = ?",
            (event_id,),
        ).fetchone()[0] == run_id

        store.close_capture_run(
            run_id="continuous_test",
            ended_utc="2026-06-01T00:03:00.000Z",
            notes="done",
        )
        row = store.connection.execute(
            "SELECT ended_utc, notes FROM capture_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        assert row == ("2026-06-01T00:03:00.000Z", "done")


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


def test_upsert_song_folds_feature_suffix_punctuation_drift(tmp_path: Path) -> None:
    """Shazam's '(feat. X)' parens and the audfprint filename's '_feat. X'
    underscore are the SAME recording and must resolve to one row, while a
    plain title with no feature credit stays distinct.

    Mirrors the Yellowcard 'Bedroom Posters' rows observed in the live DB:
    id 82 (shazam, parens) and id 224 (audfprint, underscore) were two rows for
    one recording, fragmenting the play log.
    """
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        shazam_feat = store.upsert_song(
            artist="Yellowcard",
            title="Bedroom Posters (feat. Good Charlotte)",
            source="shazam",
        )
        audfprint_feat = store.upsert_song(
            artist="Yellowcard",
            title="Bedroom Posters _feat. Good Charlotte",
            audfprint_track_id="data/reference/songs/Yellowcard - Bedroom Posters _feat. Good Charlotte.mp3",
            source="audfprint",
        )
        assert shazam_feat == audfprint_feat, (
            "parens vs underscore feature credit must be one song"
        )

        # The non-feature base recording must remain a separate row.
        base = store.upsert_song(
            artist="Yellowcard",
            title="Bedroom Posters",
            audfprint_track_id="data/reference/songs/Yellowcard - Bedroom Posters.mp3",
            source="audfprint",
        )
        assert base != shazam_feat, "feat version and base version stay distinct"

        n = store.connection.execute(
            "SELECT COUNT(*) FROM songs WHERE artist = 'Yellowcard'"
        ).fetchone()[0]
        assert n == 2


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


def test_upsert_song_is_unicode_casefold_insensitive(tmp_path: Path) -> None:
    """SQLite LOWER() is ASCII-only, so non-ASCII Shazam text needs the same
    Unicode-aware normalization as Python before we try to insert again.
    """
    db = tmp_path / "rc.db"
    with BroadcastStore(db) as store:
        first = store.upsert_song(artist="MÀREL", title="the wave", source="shazam")
        second = store.upsert_song(artist="màrel", title="the wave", source="shazam")

        assert first == second
        n = store.connection.execute(
            "SELECT COUNT(*) FROM songs"
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
