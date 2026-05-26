"""Scrape a station's "recently played" page into a deduped tracklist.

This module is opt-in (``pip install radio-classifier[seeding]``) and is never
imported by the runtime path. The runtime ingest pipeline does not touch the
network in default mode (Shazam is the only opt-in exception).

Design:

* No station-specific selectors hardcoded — the operator passes a
  CSS selector (or row pattern) that identifies tracklist rows on the page.
* Output is a list of ``Track(artist, title)`` records, deduped.
* Operator pipes / writes the result however they want — no v1 CSV emitter
  (CLI is print-only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    artist: str
    title: str


def fetch_html(url: str, *, timeout: float = 30.0) -> str:
    """Fetch a URL with a polite user-agent string."""
    import requests  # type: ignore

    headers = {"User-Agent": "radio-classifier-seed/0.1 (+https://example.invalid)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_tracklist(
    html: str,
    *,
    row_selector: str,
    artist_selector: str,
    title_selector: str,
) -> list[Track]:
    """Parse one "recently played" page using operator-provided CSS selectors.

    The operator must inspect the station's HTML and supply selectors. Two
    common patterns:

    * Rows in a table: ``row_selector='tr.played-row'``,
      ``artist_selector='.artist'``, ``title_selector='.title'``.
    * List items: ``row_selector='li.song'``, etc.
    """
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html, "html.parser")
    out: list[Track] = []
    seen: set[tuple[str, str]] = set()
    for row in soup.select(row_selector):
        artist_node = row.select_one(artist_selector)
        title_node = row.select_one(title_selector)
        if artist_node is None or title_node is None:
            continue
        artist = artist_node.get_text(" ", strip=True)
        title = title_node.get_text(" ", strip=True)
        if not artist or not title:
            continue
        key = (artist.lower(), title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Track(artist=artist, title=title))
    return out


def dedupe_tracks(tracks: list[Track]) -> list[Track]:
    """Case-insensitive dedupe preserving insertion order."""
    seen: set[tuple[str, str]] = set()
    out: list[Track] = []
    for t in tracks:
        key = (t.artist.lower(), t.title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out
