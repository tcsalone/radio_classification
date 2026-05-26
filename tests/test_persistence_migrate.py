"""Migration: live105sux v1 broadcast_events -> radio-classifier v2."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from radio_classifier.persistence import BroadcastStore
from radio_classifier.persistence.migrate import migrate_from_live105sux


_V1_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS broadcast_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_start TEXT NOT NULL,
    timestamp_end TEXT,
    duration REAL,
    category TEXT NOT NULL CHECK (category IN ('MUSIC', 'DJ', 'AD')),
    artist TEXT, track_title TEXT, brand_name TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def _make_v1(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(_V1_SCHEMA)
        conn.executemany(
            "INSERT INTO broadcast_events "
            "(timestamp_start, timestamp_end, duration, category, artist, track_title, brand_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("2020-01-01T00:00:00.000Z", "2020-01-01T00:00:30.000Z", 30.0, "MUSIC", "Taylor", "Anti-Hero", None),
                ("2020-01-01T00:00:30.000Z", "2020-01-01T00:01:00.000Z", 30.0, "DJ", None, None, None),
                ("2020-01-01T00:01:00.000Z", "2020-01-01T00:01:30.000Z", 30.0, "AD", None, None, "Geico"),
                ("2020-01-01T00:01:30.000Z", "2020-01-01T00:02:00.000Z", 30.0, "AD", None, None, "Toyota"),
                ("2020-01-01T00:02:00.000Z", "2020-01-01T00:02:30.000Z", 30.0, "AD", None, None, "Geico"),  # duplicate brand
            ],
        )
        conn.commit()


def test_migration_maps_categories_and_links_brands(tmp_path: Path) -> None:
    src = tmp_path / "v1.db"
    dst = tmp_path / "v2.db"
    _make_v1(src)
    report = migrate_from_live105sux(src_db=src, dst_db=dst)
    assert report.rows_read == 5
    assert report.rows_inserted == 5
    assert report.rows_skipped == 0
    assert report.brands_created == 2  # Geico and Toyota; duplicate Geico not double-counted

    with BroadcastStore(dst) as store:
        # Five rows mapped to five v2 categories: SONG, DJ, COMMERCIAL x3.
        cats = [
            r[0]
            for r in store.connection.execute(
                "SELECT category FROM broadcast_events ORDER BY id"
            ).fetchall()
        ]
        assert cats == ["SONG", "DJ", "COMMERCIAL", "COMMERCIAL", "COMMERCIAL"]
        # Geico commercials all share one brand_id.
        brand_ids = [
            r[0]
            for r in store.connection.execute(
                "SELECT brand_id FROM broadcast_events WHERE brand_name = 'Geico'"
            ).fetchall()
        ]
        assert len(set(brand_ids)) == 1
        # No commercial_id was inferred retroactively.
        ad_commercial_ids = [
            r[0]
            for r in store.connection.execute(
                "SELECT commercial_id FROM broadcast_events WHERE category = 'COMMERCIAL'"
            ).fetchall()
        ]
        assert ad_commercial_ids == [None, None, None]
