"""Human-readable table formatting for CLI reports (no CSV/JSON in v1)."""

from __future__ import annotations

from radio_classifier.discovery.songs import DiscoveryRow
from radio_classifier.reports.queries import (
    BrandRow,
    CommercialRow,
    SongRow,
    SummaryRow,
    TimelineRow,
)


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(rem):02d}s"
    hours, rem_min = divmod(minutes, 60)
    return f"{int(hours)}h{int(rem_min):02d}m"


def _render_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    if not rows:
        return "(no rows)"
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))
    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) for r in rows)
    return f"{head}\n{sep}\n{body}"


def format_commercials(rows: list[CommercialRow]) -> str:
    headers = ["id", "brand", "bucket(s)", "plays", "total", "last_heard_utc"]
    body = [
        [
            str(r.commercial_id if r.commercial_id is not None else "?"),
            r.brand or "?",
            str(r.duration_bucket_seconds or "?"),
            str(r.play_count),
            _format_seconds(r.total_duration_seconds),
            r.last_heard_utc or "?",
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_brands(rows: list[BrandRow]) -> str:
    headers = ["brand", "paid_ad", "dj_shoutout", "tag", "total"]
    body = [
        [
            r.brand,
            str(r.paid_play_count),
            str(r.dj_shoutout_count),
            str(r.tag_count),
            str(r.paid_play_count + r.dj_shoutout_count + r.tag_count),
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_songs(rows: list[SongRow]) -> str:
    headers = ["song_id", "artist", "title", "plays", "total"]
    body = [
        [
            str(r.song_id) if r.song_id is not None else "?",
            r.artist or "?",
            r.title or "?",
            str(r.play_count),
            _format_seconds(r.total_duration_seconds),
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_timeline(rows: list[TimelineRow]) -> str:
    headers = ["start_utc", "duration", "category", "brand", "artist/title", "excerpt"]
    body = [
        [
            r.start_utc,
            _format_seconds(r.duration_seconds),
            r.category,
            r.brand or "-",
            _artist_title(r.artist, r.title),
            _truncate(r.transcript_excerpt, 60),
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_discoveries(rows: list[DiscoveryRow]) -> str:
    headers = ["id", "artist", "title", "plays", "last_heard", "tracklist"]
    body = [
        [
            str(r.song_id),
            r.artist or "?",
            r.title or "?",
            str(r.play_count),
            r.last_heard_utc or "-",
            "present" if r.in_tracklist else "missing",
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_summary(rows: list[SummaryRow]) -> str:
    headers = ["category", "segments", "total_airtime"]
    body = [
        [r.category, str(r.segment_count), _format_seconds(r.total_duration_seconds)]
        for r in rows
    ]
    return _render_table(headers, body)


def _artist_title(artist: str | None, title: str | None) -> str:
    if artist and title:
        return f"{artist} - {title}"
    return artist or title or "-"


def _truncate(value: str | None, n: int) -> str:
    if not value:
        return "-"
    if len(value) <= n:
        return value
    return value[: n - 1] + "…"
