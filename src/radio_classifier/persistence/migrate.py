"""Migrate a ``live105sux`` v1 SQLite DB into a fresh ``radio-classifier`` v2 DB.

v1 schema: ``broadcast_events(category IN ('MUSIC', 'DJ', 'AD'), artist, track_title, brand_name, ...)``.

Mapping rules:

* ``MUSIC`` → ``SONG``. ``song_id`` stays ``NULL`` (no audfprint identity for
  historical rows). ``artist`` / ``track_title`` carried over.
* ``DJ`` → ``DJ``. ``brand_name`` carried over for posterity; no ``brand_id``.
* ``AD`` → ``COMMERCIAL``. ``brand_name`` resolved into ``brands`` and linked
  via ``brand_id``. ``commercial_id`` remains ``NULL`` (no signature
  retroactively available).

The migrator does **not** modify the source database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.reducer import duration_seconds


_V1_TO_V2_CATEGORY: dict[str, str] = {
    "MUSIC": "SONG",
    "DJ": "DJ",
    "AD": "COMMERCIAL",
}


@dataclass
class MigrationReport:
    rows_read: int = 0
    rows_inserted: int = 0
    rows_skipped: int = 0
    brands_created: int = 0


def migrate_from_live105sux(
    *,
    src_db: Path,
    dst_db: Path,
    schema_path: Path | None = None,
) -> MigrationReport:
    """Copy rows from a v1 ``broadcast_events`` table into a v2 store.

    Both DBs are SQLite files. ``dst_db`` is created/initialised from the v2
    schema if it does not exist. The function is idempotent insofar as rows
    are appended, not deduped — call it against an empty destination.
    """
    src_db = Path(src_db).resolve()
    dst_db = Path(dst_db).resolve()
    if not src_db.exists():
        raise FileNotFoundError(f"source db not found: {src_db}")

    report = MigrationReport()
    with sqlite3.connect(src_db) as src_conn:
        src_conn.row_factory = sqlite3.Row
        src_rows = list(
            src_conn.execute(
                """
                SELECT timestamp_start, timestamp_end, duration,
                       category, artist, track_title, brand_name
                FROM broadcast_events
                ORDER BY id ASC
                """
            ).fetchall()
        )
        report.rows_read = len(src_rows)

    with BroadcastStore(dst_db, schema_path=schema_path) as dst:
        # Track brands already upserted in this run for the counter only.
        seen_brand_names: set[str] = set()
        for r in src_rows:
            v1_cat = (r["category"] or "").upper()
            v2_cat = _V1_TO_V2_CATEGORY.get(v1_cat)
            if v2_cat is None:
                report.rows_skipped += 1
                continue

            brand_name = r["brand_name"]
            brand_id: int | None = None
            if v2_cat == "COMMERCIAL" and brand_name:
                brand_id = dst.upsert_brand(brand_name.strip())
                if brand_name not in seen_brand_names:
                    seen_brand_names.add(brand_name)
                    report.brands_created += 1

            duration = r["duration"]
            if duration is None and r["timestamp_end"]:
                try:
                    duration = duration_seconds(
                        r["timestamp_start"], r["timestamp_end"]
                    )
                except Exception:
                    duration = None

            dst.connection.execute(
                """
                INSERT INTO broadcast_events (
                    timestamp_start, timestamp_end, duration,
                    category, song_id, commercial_id, brand_id,
                    artist, track_title, brand_name,
                    transcript_excerpt, confidence
                ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    r["timestamp_start"],
                    r["timestamp_end"],
                    duration,
                    v2_cat,
                    brand_id,
                    r["artist"],
                    r["track_title"],
                    brand_name,
                ),
            )
            report.rows_inserted += 1
        dst.connection.commit()

    return report
