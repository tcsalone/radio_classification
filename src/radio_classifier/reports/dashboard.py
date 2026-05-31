"""Static HTML dashboard generation for SQLite report data."""

from __future__ import annotations

from html import escape
from pathlib import Path

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.reports.format import _format_seconds
from radio_classifier.reports.queries import (
    artists_top,
    brands_top,
    commercials_top,
    songs_top,
    summary,
)


def write_dashboard(
    store: BroadcastStore,
    *,
    since_utc: str,
    out_path: Path,
    top_n: int = 10,
) -> Path:
    """Write a dependency-free HTML dashboard and return its path."""
    html = render_dashboard_html(store, since_utc=since_utc, top_n=top_n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_dashboard_html(
    store: BroadcastStore,
    *,
    since_utc: str,
    top_n: int = 10,
) -> str:
    """Render a static metrics page for one DB/time window."""
    summary_rows = summary(store, since_utc=since_utc)
    song_rows = songs_top(store, since_utc=since_utc, top_n=top_n)
    artist_rows = artists_top(store, since_utc=since_utc, top_n=top_n)
    brand_rows = brands_top(store, since_utc=since_utc, top_n=top_n)
    commercial_rows = commercials_top(store, since_utc=since_utc, top_n=top_n)
    hourly_rows = _hourly_category_rows(store, since_utc=since_utc)

    total_airtime = sum(r.total_duration_seconds for r in summary_rows)
    total_segments = sum(r.segment_count for r in summary_rows)
    song_airtime = next((r.total_duration_seconds for r in summary_rows if r.category == "SONG"), 0.0)
    commercial_airtime = next(
        (r.total_duration_seconds for r in summary_rows if r.category == "COMMERCIAL"),
        0.0,
    )
    song_share = (song_airtime / total_airtime * 100.0) if total_airtime else 0.0
    commercial_share = (commercial_airtime / total_airtime * 100.0) if total_airtime else 0.0

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Radio Classifier Dashboard</title>",
            f"<style>{_CSS}</style>",
            "</head>",
            "<body>",
            "<main>",
            "<header>",
            "<p class=\"eyebrow\">radio-classifier</p>",
            "<h1>Broadcast Metrics Dashboard</h1>",
            f"<p>Window starts at <code>{escape(since_utc)}</code>. Generated from the selected SQLite DB.</p>",
            "</header>",
            '<section class="cards">',
            _metric_card("Classified Airtime", _format_seconds(total_airtime), "Sum of event durations"),
            _metric_card("Segments", str(total_segments), "Raw broadcast_events rows"),
            _metric_card("Music Share", f"{song_share:.1f}%", "SONG airtime / classified airtime"),
            _metric_card("Commercial Share", f"{commercial_share:.1f}%", "COMMERCIAL airtime / classified airtime"),
            "</section>",
            '<section class="grid">',
            _panel("Category Airtime", _category_bars(summary_rows)),
            _panel("Top Artists", _artists_table(artist_rows)),
            _panel("Top Songs", _songs_table(song_rows)),
            _panel("Top Brands", _brands_table(brand_rows)),
            _panel("Top Commercials", _commercials_table(commercial_rows)),
            _panel("Hourly Category Mix", _hourly_table(hourly_rows)),
            "</section>",
            "</main>",
            "</body>",
            "</html>",
        ]
    )


def _metric_card(label: str, value: str, note: str) -> str:
    return (
        '<article class="card">'
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value)}</strong>"
        f"<small>{escape(note)}</small>"
        "</article>"
    )


def _panel(title: str, body: str) -> str:
    return f'<section class="panel"><h2>{escape(title)}</h2>{body}</section>'


def _category_bars(rows) -> str:  # type: ignore[no-untyped-def]
    if not rows:
        return '<p class="empty">No data.</p>'
    max_seconds = max((r.total_duration_seconds for r in rows), default=1.0) or 1.0
    parts = ['<div class="bars">']
    for r in rows:
        pct = max(2.0, (r.total_duration_seconds / max_seconds) * 100.0)
        parts.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{escape(r.category)}</span>'
            '<span class="bar-track">'
            f'<span class="bar-fill cat-{escape(r.category.lower())}" style="width:{pct:.1f}%"></span>'
            "</span>"
            f'<span class="bar-value">{escape(_format_seconds(r.total_duration_seconds))}</span>'
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _artists_table(rows) -> str:  # type: ignore[no-untyped-def]
    return _table(
        ["Artist", "Spins", "Promos", "Titles", "Segments", "Airtime"],
        [
            [
                _promo_marked(r.artist, r.is_promo_only),
                _spin_html(r.full_spin_count, r.promo_spin_count),
                _promo_summary(r.promo_spin_count, r.promo_duration_seconds),
                str(r.distinct_titles),
                str(r.segment_count),
                _format_seconds(r.total_duration_seconds),
            ]
            for r in rows
        ],
        raw_columns={0, 1, 2},
    )


def _songs_table(rows) -> str:  # type: ignore[no-untyped-def]
    return _table(
        ["Artist", "Title", "Spins", "Promos", "Segments", "Airtime"],
        [
            [
                r.artist or "?",
                _promo_marked(r.title or "?", r.is_promo_only),
                _spin_html(r.full_spin_count, r.promo_spin_count),
                _promo_summary(r.promo_spin_count, r.promo_duration_seconds),
                str(r.segment_count),
                _format_seconds(r.total_duration_seconds),
            ]
            for r in rows
        ],
        raw_columns={1, 2, 3},
    )


def _spin_html(full_spins: int, promo_spins: int) -> str:
    """Format the spin count for HTML; matches the CLI ``X (+N)`` convention."""
    if promo_spins <= 0:
        return escape(str(full_spins))
    return (
        f'{escape(str(full_spins))} '
        f'<span class="promo-pill">+{escape(str(promo_spins))} promo</span>'
    )


def _promo_summary(promo_count: int, promo_duration: float) -> str:
    if promo_count <= 0:
        return '<span class="muted">&mdash;</span>'
    return (
        f'<span class="promo-pill">{escape(str(promo_count))} / '
        f'{escape(_format_seconds(promo_duration))}</span>'
    )


def _promo_marked(label: str, is_promo_only: bool) -> str:
    escaped = escape(label)
    if not is_promo_only:
        return escaped
    return f'{escaped} <span class="promo-tag" title="Every spin shorter than the promo threshold">[promo]</span>'


def _brands_table(rows) -> str:  # type: ignore[no-untyped-def]
    return _table(
        ["Brand", "Paid", "DJ", "Tags", "Total"],
        [
            [
                r.brand,
                str(r.paid_play_count),
                str(r.dj_shoutout_count),
                str(r.tag_count),
                str(r.paid_play_count + r.dj_shoutout_count + r.tag_count),
            ]
            for r in rows
        ],
    )


def _commercials_table(rows) -> str:  # type: ignore[no-untyped-def]
    return _table(
        ["ID", "Brand", "Plays", "Airtime", "Last Heard"],
        [
            [
                str(r.commercial_id if r.commercial_id is not None else "?"),
                r.brand or "?",
                str(r.play_count),
                _format_seconds(r.total_duration_seconds),
                r.last_heard_utc or "?",
            ]
            for r in rows
        ],
    )


def _hourly_table(rows: list[tuple[str, str, int, float]]) -> str:
    return _table(
        ["Hour", "Category", "Segments", "Airtime"],
        [[hour, category, str(count), _format_seconds(seconds)] for hour, category, count, seconds in rows],
    )


def _table(
    headers: list[str],
    rows: list[list[str]],
    *,
    raw_columns: set[int] | None = None,
) -> str:
    """Render an HTML table.

    Cells listed in ``raw_columns`` are treated as already-escaped HTML so the
    spin/promo helpers can inject ``<span>`` decorations. All other cells are
    escaped to keep the table safe with arbitrary string content.
    """
    if not rows:
        return '<p class="empty">No rows.</p>'
    raw = raw_columns or set()
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body_parts: list[str] = []
    for row in rows:
        cells: list[str] = []
        for idx, cell in enumerate(row):
            if idx in raw:
                cells.append(f"<td>{cell}</td>")
            else:
                cells.append(f"<td>{escape(str(cell))}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    body = "".join(body_parts)
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _hourly_category_rows(store: BroadcastStore, *, since_utc: str) -> list[tuple[str, str, int, float]]:
    rows = store.connection.execute(
        """
        SELECT
            substr(timestamp_start, 1, 13) || ':00Z' AS hour_utc,
            category,
            COUNT(*) AS segments,
            COALESCE(SUM(duration), 0.0) AS total_duration
        FROM broadcast_events
        WHERE timestamp_start >= ?
        GROUP BY hour_utc, category
        ORDER BY hour_utc ASC, total_duration DESC
        """,
        (since_utc,),
    ).fetchall()
    return [(str(r[0]), str(r[1]), int(r[2] or 0), float(r[3] or 0.0)) for r in rows]


_CSS = """
:root {
  color-scheme: dark;
  --bg: #0b1020;
  --panel: #121a2e;
  --muted: #94a3b8;
  --text: #e5edf7;
  --border: #26344f;
  --accent: #7dd3fc;
  --song: #22c55e;
  --commercial: #f97316;
  --dj: #a78bfa;
  --station: #38bdf8;
  --psa_news: #facc15;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top left, #172554, var(--bg) 42rem);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1180px, calc(100vw - 32px)); margin: 0 auto; padding: 32px 0 56px; }
header { margin-bottom: 24px; }
h1 { margin: 0 0 8px; font-size: clamp(2rem, 5vw, 4rem); line-height: 1; }
h2 { margin: 0 0 16px; font-size: 1.05rem; }
p { color: var(--muted); margin: 0; }
code { color: var(--accent); }
.eyebrow { color: var(--accent); text-transform: uppercase; letter-spacing: .16em; font-weight: 700; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 14px; }
.card, .panel { background: color-mix(in srgb, var(--panel) 92%, transparent); border: 1px solid var(--border); border-radius: 18px; box-shadow: 0 20px 60px #0006; }
.card { padding: 16px; }
.card span, .card small { color: var(--muted); display: block; }
.card strong { display: block; font-size: 1.7rem; margin: 8px 0 2px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.panel { padding: 18px; min-width: 0; }
.panel:nth-child(1), .panel:nth-child(6) { grid-column: 1 / -1; }
.bars { display: grid; gap: 10px; }
.bar-row { display: grid; grid-template-columns: 140px 1fr 80px; gap: 10px; align-items: center; }
.bar-label, .bar-value { color: var(--muted); white-space: nowrap; }
.bar-track { height: 12px; background: #020617; border-radius: 999px; overflow: hidden; border: 1px solid var(--border); }
.bar-fill { display: block; height: 100%; border-radius: 999px; background: var(--accent); }
.cat-song { background: var(--song); }
.cat-commercial { background: var(--commercial); }
.cat-dj { background: var(--dj); }
.cat-station { background: var(--station); }
.cat-psa_news { background: var(--psa_news); }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }
td { color: var(--text); }
.empty { padding: 12px; border: 1px dashed var(--border); border-radius: 12px; }
.promo-pill { display: inline-block; padding: 1px 8px; border-radius: 999px; background: #4c1d95; color: #ddd6fe; font-size: .78rem; letter-spacing: .03em; }
.promo-tag { color: #f0abfc; font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; }
.muted { color: var(--muted); }
@media (max-width: 800px) {
  .cards, .grid { grid-template-columns: 1fr; }
  .panel:nth-child(1), .panel:nth-child(6) { grid-column: auto; }
  .bar-row { grid-template-columns: 1fr; }
}
""".strip()
