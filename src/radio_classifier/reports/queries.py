"""Pure-SQL query helpers for the reporting CLI.

All queries take an open :class:`BroadcastStore` (so callers can use a custom
DB path) and return plain Python types — no printing here. Formatting lives
in :mod:`...reports.cli`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from radio_classifier.persistence.broadcast_store import BroadcastStore


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhd])\s*$", re.IGNORECASE)


def parse_since(value: str) -> str:
    """Convert ``--since`` text into an ISO-8601 UTC timestamp.

    Accepts ``Ns``, ``Nm``, ``Nh``, ``Nd``, or an ISO-8601 timestamp.
    """
    m = _DURATION_RE.match(value or "")
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        unit_seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        dt = datetime.now(tz=timezone.utc) - timedelta(seconds=n * unit_seconds)
        return _iso(dt)
    # Try parsing as ISO-8601.
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return _iso(dt.astimezone(timezone.utc))
    except ValueError:
        raise ValueError(f"unrecognised --since value: {value!r}")


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------- types
@dataclass
class CommercialRow:
    commercial_id: int | None
    brand: str | None
    duration_bucket_seconds: int | None
    play_count: int
    last_heard_utc: str | None
    total_duration_seconds: float


@dataclass
class BrandRow:
    brand: str
    paid_play_count: int
    dj_shoutout_count: int
    tag_count: int


@dataclass
class SongRow:
    song_id: int | None
    artist: str | None
    title: str | None
    play_count: int
    total_duration_seconds: float


@dataclass
class TimelineRow:
    start_utc: str
    end_utc: str | None
    duration_seconds: float | None
    category: str
    brand: str | None
    artist: str | None
    title: str | None
    transcript_excerpt: str | None


@dataclass
class SummaryRow:
    category: str
    segment_count: int
    total_duration_seconds: float


# -------------------------------------------------------------------- queries
def commercials_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    top_n: int = 10,
    brand: str | None = None,
) -> list[CommercialRow]:
    sql = """
        SELECT
            e.commercial_id AS commercial_id,
            b.canonical_name AS brand,
            c.duration_bucket_seconds AS duration_bucket_seconds,
            COUNT(*) AS play_count,
            MAX(e.timestamp_start) AS last_heard_utc,
            COALESCE(SUM(e.duration), 0.0) AS total_duration
        FROM broadcast_events e
        LEFT JOIN commercials c ON e.commercial_id = c.id
        LEFT JOIN brands b ON COALESCE(c.brand_id, e.brand_id) = b.id
        WHERE e.category = 'COMMERCIAL' AND e.timestamp_start >= ?
    """
    args: list = [since_utc]
    if brand:
        sql += " AND b.canonical_name = ?"
        args.append(brand)
    sql += """
        GROUP BY e.commercial_id, b.canonical_name, c.duration_bucket_seconds
        ORDER BY play_count DESC, total_duration DESC
        LIMIT ?
    """
    args.append(top_n)
    rows = store.connection.execute(sql, args).fetchall()
    return [
        CommercialRow(
            commercial_id=r[0],
            brand=r[1],
            duration_bucket_seconds=r[2],
            play_count=int(r[3]),
            last_heard_utc=r[4],
            total_duration_seconds=float(r[5] or 0.0),
        )
        for r in rows
    ]


def brands_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    top_n: int = 10,
) -> list[BrandRow]:
    sql = """
        SELECT
            b.canonical_name AS brand,
            SUM(CASE WHEN bm.mention_type = 'paid_ad' THEN 1 ELSE 0 END) AS paid_ad,
            SUM(CASE WHEN bm.mention_type = 'dj_shoutout' THEN 1 ELSE 0 END) AS dj_shoutout,
            SUM(CASE WHEN bm.mention_type = 'tag' THEN 1 ELSE 0 END) AS tag
        FROM brand_mentions bm
        JOIN brands b ON bm.brand_id = b.id
        WHERE bm.heard_utc >= ?
        GROUP BY b.canonical_name
        ORDER BY (paid_ad + dj_shoutout + tag) DESC, brand ASC
        LIMIT ?
    """
    rows = store.connection.execute(sql, (since_utc, top_n)).fetchall()
    return [
        BrandRow(
            brand=str(r[0]),
            paid_play_count=int(r[1] or 0),
            dj_shoutout_count=int(r[2] or 0),
            tag_count=int(r[3] or 0),
        )
        for r in rows
    ]


def songs_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    top_n: int = 10,
) -> list[SongRow]:
    sql = """
        SELECT
            e.song_id AS song_id,
            COALESCE(s.artist, e.artist) AS artist,
            COALESCE(s.title, e.track_title) AS title,
            COUNT(*) AS play_count,
            COALESCE(SUM(e.duration), 0.0) AS total_duration
        FROM broadcast_events e
        LEFT JOIN songs s ON e.song_id = s.id
        WHERE e.category = 'SONG' AND e.timestamp_start >= ?
        GROUP BY e.song_id, COALESCE(s.artist, e.artist), COALESCE(s.title, e.track_title)
        ORDER BY play_count DESC, total_duration DESC
        LIMIT ?
    """
    rows = store.connection.execute(sql, (since_utc, top_n)).fetchall()
    return [
        SongRow(
            song_id=r[0],
            artist=r[1],
            title=r[2],
            play_count=int(r[3]),
            total_duration_seconds=float(r[4] or 0.0),
        )
        for r in rows
    ]


def timeline(
    store: BroadcastStore,
    *,
    since_utc: str,
    limit: int = 500,
) -> list[TimelineRow]:
    sql = """
        SELECT
            e.timestamp_start,
            e.timestamp_end,
            e.duration,
            e.category,
            COALESCE(b.canonical_name, e.brand_name),
            COALESCE(s.artist, e.artist),
            COALESCE(s.title, e.track_title),
            e.transcript_excerpt
        FROM broadcast_events e
        LEFT JOIN brands b ON e.brand_id = b.id
        LEFT JOIN songs s ON e.song_id = s.id
        WHERE e.timestamp_start >= ?
        ORDER BY e.timestamp_start ASC
        LIMIT ?
    """
    rows = store.connection.execute(sql, (since_utc, limit)).fetchall()
    return [
        TimelineRow(
            start_utc=str(r[0]),
            end_utc=r[1],
            duration_seconds=(float(r[2]) if r[2] is not None else None),
            category=str(r[3]),
            brand=r[4],
            artist=r[5],
            title=r[6],
            transcript_excerpt=r[7],
        )
        for r in rows
    ]


def summary(
    store: BroadcastStore,
    *,
    since_utc: str,
) -> list[SummaryRow]:
    sql = """
        SELECT category,
               COUNT(*) AS segment_count,
               COALESCE(SUM(duration), 0.0) AS total_duration
        FROM broadcast_events
        WHERE timestamp_start >= ?
        GROUP BY category
        ORDER BY total_duration DESC
    """
    rows = store.connection.execute(sql, (since_utc,)).fetchall()
    return [
        SummaryRow(
            category=str(r[0]),
            segment_count=int(r[1]),
            total_duration_seconds=float(r[2] or 0.0),
        )
        for r in rows
    ]
