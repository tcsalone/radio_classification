"""CLI-only reporting: brands, commercials, songs, timeline, summary."""

from radio_classifier.reports.dashboard import render_dashboard_html, write_dashboard
from radio_classifier.reports.format import (
    format_artists,
    format_brands,
    format_commercials,
    format_discoveries,
    format_songs,
    format_songs_timeline,
    format_summary,
    format_timeline,
)
from radio_classifier.reports.queries import (
    PROMO_MAX_SPIN_SECONDS,
    artists_top,
    brands_top,
    commercials_top,
    parse_since,
    songs_timeline,
    songs_top,
    summary,
    timeline,
)

__all__ = [
    "PROMO_MAX_SPIN_SECONDS",
    "artists_top",
    "brands_top",
    "commercials_top",
    "format_artists",
    "format_brands",
    "format_commercials",
    "format_discoveries",
    "format_songs",
    "format_songs_timeline",
    "format_summary",
    "format_timeline",
    "parse_since",
    "render_dashboard_html",
    "songs_timeline",
    "songs_top",
    "summary",
    "timeline",
    "write_dashboard",
]
