"""Human-readable table formatting for CLI reports (no CSV/JSON in v1)."""

from __future__ import annotations

from radio_classifier.discovery.songs import DiscoveryRow
from radio_classifier.reports.queries import (
    ArtistRow,
    BrandRow,
    CommercialRow,
    RunRow,
    SongsAddedRow,
    SongRow,
    SongTimelineRow,
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
    headers = ["song_id", "artist", "title", "spins", "promos", "segments", "total"]
    body = [
        [
            str(r.song_id) if r.song_id is not None else "?",
            r.artist or "?",
            _decorate_promo_title(r.title or "?", r.is_promo_only),
            _spin_cell(r.full_spin_count, r.promo_spin_count),
            _promo_cell(r.promo_spin_count, r.promo_duration_seconds),
            str(r.segment_count),
            _format_seconds(r.total_duration_seconds),
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_songs_timeline(rows: list[SongTimelineRow]) -> str:
    headers = ["start_utc", "duration", "song_id", "artist", "title", "source", "confidence"]
    body = [
        [
            r.start_utc,
            _format_seconds(r.duration_seconds),
            str(r.song_id) if r.song_id is not None else "?",
            r.artist or "?",
            r.title or "?",
            r.detection_source or "unknown",
            f"{r.confidence:.3g}" if r.confidence is not None else "-",
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_artists(rows: list[ArtistRow]) -> str:
    headers = ["artist", "spins", "promos", "titles", "segments", "total"]
    body = [
        [
            _decorate_promo_title(r.artist, r.is_promo_only),
            _spin_cell(r.full_spin_count, r.promo_spin_count),
            _promo_cell(r.promo_spin_count, r.promo_duration_seconds),
            str(r.distinct_titles),
            str(r.segment_count),
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
    headers = ["id", "artist", "title", "plays", "last_heard", "tracklist", "review"]
    body = [
        [
            str(r.song_id),
            r.artist or "?",
            r.title or "?",
            str(r.play_count),
            r.last_heard_utc or "-",
            "present" if r.in_tracklist else "missing",
            "manual" if r.needs_review else "-",
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


def format_songs_added(rows: list[SongsAddedRow]) -> str:
    headers = ["first_seen_utc", "song_id", "artist", "title", "source", "segments"]
    body = [
        [
            r.first_seen_utc,
            str(r.song_id),
            r.artist or "?",
            r.title or "?",
            r.source,
            str(r.segment_count),
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def format_runs(rows: list[RunRow]) -> str:
    headers = ["id", "run_id", "started_utc", "ended_utc", "events", "duration", "pipeline"]
    body = [
        [
            str(r.capture_run_id),
            r.run_id,
            r.started_utc,
            r.ended_utc or "-",
            str(r.event_count),
            _format_seconds(r.duration_seconds),
            r.pipeline_version,
        ]
        for r in rows
    ]
    return _render_table(headers, body)


def _spin_cell(full_spins: int, promo_spins: int) -> str:
    """Render the ``spins`` column with a hint when promo clips are included.

    Examples:
      * ``"3"`` — three normal-length plays, no promos detected.
      * ``"0 (+10)"`` — every detected spin was a short promo clip.
      * ``"2 (+1)"`` — two full plays plus one promo-shaped spin.
    """
    if promo_spins <= 0:
        return str(full_spins)
    return f"{full_spins} (+{promo_spins})"


def _promo_cell(promo_count: int, promo_duration: float) -> str:
    if promo_count <= 0:
        return "-"
    return f"{promo_count} / {_format_seconds(promo_duration)}"


def _decorate_promo_title(label: str, is_promo_only: bool) -> str:
    """Annotate the song/artist label when every spin was promo-shaped."""
    if not is_promo_only:
        return label
    return f"{label} [promo]"


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
