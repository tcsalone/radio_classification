"""Static HTML "Top Artist Play Log" report.

A drill-down companion to the dashboard's Top Artists panel: one section per
top artist, listing every play (artist, title, timestamp, length) over the
window. Reuses the dashboard's dark theme and table helpers so the two pages
look like one product.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.reports.dashboard import _CSS, _table
from radio_classifier.reports.format import _format_seconds
from radio_classifier.reports.queries import ArtistPlaylog, top_artist_playlogs

_EXTRA_CSS = """
.artist-stack { display: grid; gap: 14px; }
.panel h2 .rank-pill {
  display: inline-block; margin-right: 10px; padding: 2px 10px;
  border-radius: 999px; background: #0e7490; color: #cffafe;
  font-size: .85rem; font-weight: 700; vertical-align: middle;
}
.subhead { color: var(--muted); margin: -8px 0 14px; }
td.when { color: var(--muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
""".strip()


def write_artist_plays(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    out_path: Path,
    top_n: int = 3,
) -> Path:
    """Write the artist play-log HTML page and return its path."""
    html = render_artist_plays_html(store, since_utc=since_utc, until_utc=until_utc, top_n=top_n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_artist_plays_html(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 3,
) -> str:
    """Render the static per-artist play-log page for one DB/time window."""
    logs = top_artist_playlogs(store, since_utc=since_utc, until_utc=until_utc, top_n=top_n)

    if logs:
        body = '<section class="artist-stack">' + "".join(_artist_section(log) for log in logs) + "</section>"
    else:
        body = (
            '<section class="panel"><p class="empty">No artists with song plays '
            "in this window.</p></section>"
        )

    window_line = (
        f"<p>Window <code>{escape(since_utc)}</code>"
        + (f" → <code>{escape(until_utc)}</code>" if until_utc else " → now")
        + f". Top {len(logs)} artist(s) by spins, every play listed.</p>"
    )

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Top Artist Play Log</title>",
            f"<style>{_CSS}\n{_EXTRA_CSS}</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            '<p class="eyebrow">radio-classifier</p>',
            "<h1>Top Artist Play Log</h1>",
            window_line,
            "</header>",
            body,
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _artist_section(log: ArtistPlaylog) -> str:
    head = (
        f'<h2><span class="rank-pill">#{log.rank + 1}</span>{escape(log.artist)}</h2>'
    )
    subhead = (
        f'<p class="subhead">{log.total_plays} play(s) · '
        f"{log.distinct_titles} distinct title(s)</p>"
    )
    rows = [
        [
            f'<span class="when">{escape(_format_ts(play.start_utc))}</span>',
            _title_cell(play.title, play.is_promo),
            _format_seconds(play.duration_seconds),
        ]
        for play in log.plays
    ]
    table = _table(["When (UTC)", "Title", "Length"], rows, raw_columns={0, 1})
    return f'<section class="panel">{head}{subhead}{table}</section>'


def _title_cell(title: str | None, is_promo: bool) -> str:
    label = escape(title or "Unknown title")
    if is_promo:
        label += (
            ' <span class="promo-tag" title="Play shorter than the promo '
            'threshold (likely a teaser clip)">[promo]</span>'
        )
    return label


def _format_ts(iso: str) -> str:
    """Render an ISO-8601 UTC timestamp as ``YYYY-MM-DD HH:MM:SS``."""
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return iso
