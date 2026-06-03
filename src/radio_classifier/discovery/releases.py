"""MusicBrainz release-date enrichment for ``songs`` rows.

Uses the public MusicBrainz JSON API (no extra dependency). Respects the
1-request-per-second rate limit. Release dates are stored as ISO ``YYYY-MM-DD``
on ``songs.release_date`` and power dashboard age metrics.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone

from radio_classifier.discovery.songs import _FEATURE_SUFFIX_RE
from radio_classifier.persistence.broadcast_store import BroadcastStore


_MB_USER_AGENT = "radio-classifier/0.1 (local-radio-metrics; contact=ops@localhost)"
_MB_MIN_INTERVAL_SECONDS = 1.05
_LAST_MB_REQUEST_MONO = 0.0


@dataclass
class ReleaseEnrichResult:
    """Outcome of one :func:`enrich_song_releases` call."""

    examined: int
    updated: int
    skipped_existing: int
    not_found: int
    errors: int


def normalize_title_for_lookup(title: str | None) -> str:
    """Strip featured-artist suffixes before querying MusicBrainz."""
    if not title:
        return ""
    return _FEATURE_SUFFIX_RE.sub("", title).strip()


def lookup_release_date(
    artist: str | None,
    title: str | None,
    *,
    user_agent: str = _MB_USER_AGENT,
) -> str | None:
    """Return ``YYYY-MM-DD`` for a recording, or ``None`` if unresolved."""
    clean_artist = (artist or "").strip()
    clean_title = normalize_title_for_lookup(title)
    if not clean_artist or not clean_title:
        return None
    query = f'artist:"{_mb_escape(clean_artist)}" AND recording:"{_mb_escape(clean_title)}"'
    url = (
        "https://musicbrainz.org/ws/2/recording/?"
        + urllib.parse.urlencode({"query": query, "fmt": "json", "limit": "5"})
    )
    try:
        payload = _mb_get_json(url, user_agent=user_agent)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    recordings = payload.get("recordings") or []
    for rec in recordings:
        parsed = _best_release_date_from_recording(rec)
        if parsed is not None:
            return parsed
    return None


def enrich_song_releases(
    store: BroadcastStore,
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    only_missing: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
) -> ReleaseEnrichResult:
    """Fill ``songs.release_date`` via MusicBrainz for catalog rows.

    When ``since_utc`` is set, only songs that appeared as ``SONG`` events in
    that window are considered. ``only_missing=True`` skips rows that already
    have a release date.
    """
    rows = _songs_to_enrich(store, since_utc=since_utc, until_utc=until_utc, only_missing=only_missing)
    if limit is not None:
        rows = rows[: max(0, limit)]

    examined = 0
    updated = 0
    skipped_existing = 0
    not_found = 0
    errors = 0

    for song_id, artist, title, existing_date in rows:
        examined += 1
        if existing_date:
            skipped_existing += 1
            continue
        try:
            release_date = lookup_release_date(artist, title)
        except Exception:
            errors += 1
            continue
        if release_date is None:
            not_found += 1
            continue
        if not dry_run:
            store.set_song_release_date(song_id, release_date)
        updated += 1

    if not dry_run and updated:
        store.connection.commit()
    return ReleaseEnrichResult(
        examined=examined,
        updated=updated,
        skipped_existing=skipped_existing,
        not_found=not_found,
        errors=errors,
    )


def _songs_to_enrich(
    store: BroadcastStore,
    *,
    since_utc: str | None,
    until_utc: str | None,
    only_missing: bool,
) -> list[tuple[int, str | None, str | None, str | None]]:
    if since_utc:
        window = "e.timestamp_start >= ?"
        args: list[object] = [since_utc]
        if until_utc:
            window += " AND e.timestamp_start < ?"
            args.append(until_utc)
        missing_clause = "AND s.release_date IS NULL" if only_missing else ""
        sql = f"""
            SELECT DISTINCT s.id, s.artist, s.title, s.release_date
            FROM songs s
            JOIN broadcast_events e ON e.song_id = s.id
            WHERE e.category = 'SONG'
              AND {window}
              {missing_clause}
            ORDER BY s.id ASC
        """
        return [
            (int(r[0]), r[1], r[2], r[3])
            for r in store.connection.execute(sql, args).fetchall()
        ]

    missing_clause = "WHERE release_date IS NULL" if only_missing else ""
    sql = f"""
        SELECT id, artist, title, release_date
        FROM songs
        {missing_clause}
        ORDER BY id ASC
    """
    return [
        (int(r[0]), r[1], r[2], r[3])
        for r in store.connection.execute(sql).fetchall()
    ]


def _mb_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _mb_get_json(url: str, *, user_agent: str) -> dict:
    global _LAST_MB_REQUEST_MONO
    elapsed = time.monotonic() - _LAST_MB_REQUEST_MONO
    if elapsed < _MB_MIN_INTERVAL_SECONDS:
        time.sleep(_MB_MIN_INTERVAL_SECONDS - elapsed)
    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()
    _LAST_MB_REQUEST_MONO = time.monotonic()
    return json.loads(data.decode("utf-8"))


def _best_release_date_from_recording(recording: dict) -> str | None:
    candidates: list[str] = []
    direct = recording.get("first-release-date")
    if isinstance(direct, str) and direct.strip():
        candidates.append(direct.strip())
    for release in recording.get("releases") or []:
        if isinstance(release, dict):
            value = release.get("date")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    parsed_dates = [d for d in (_parse_mb_date(v) for v in candidates) if d is not None]
    if not parsed_dates:
        return None
    return min(parsed_dates).isoformat()


_MB_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


def _parse_mb_date(raw: str) -> date | None:
    m = _MB_DATE_RE.match(raw.strip())
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2) or 1)
    day = int(m.group(3) or 1)
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_reference_utc(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def song_age_years(release_date_iso: str, reference: datetime) -> float:
    """Years between release date and the reference instant."""
    released = _parse_mb_date(release_date_iso)
    if released is None:
        raise ValueError(f"invalid release_date: {release_date_iso!r}")
    ref_day = reference.date()
    return (ref_day - released).days / 365.25
