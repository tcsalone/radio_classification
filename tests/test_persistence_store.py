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
