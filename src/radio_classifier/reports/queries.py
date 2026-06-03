"""Pure-SQL query helpers for the reporting CLI.

All queries take an open :class:`BroadcastStore` (so callers can use a custom
DB path) and return plain Python types — no printing here. Formatting lives
in :mod:`...reports.cli`.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from radio_classifier.brands import canonicalize_brand
from radio_classifier.discovery.releases import parse_reference_utc, song_age_years
from radio_classifier.persistence.broadcast_store import BroadcastStore

PROMO_MAX_SPIN_SECONDS: float = 90.0
"""Spin durations shorter than this are treated as station-promo clips.

Stations frequently air 10-60 second slices of an upcoming track as a teaser
("hear Julia Wolf next on Live 105"). Those snippets land in the same
``broadcast_events`` table as real plays and can inflate spin counts 10x.
We don't have ground-truth song length, so the threshold is a heuristic:
real radio plays of typical songs run well above 90 seconds, and promo
clips almost never exceed it.
"""


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


def _parse_iso_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _gap_seconds(end_iso: str, next_start_iso: str) -> float:
    """Wall-clock seconds between ``end_iso`` and ``next_start_iso``.

    Returns 0.0 when the next segment starts before (or at) the previous
    segment's end (overlapping windows can do this), so overlap is treated as
    "no gap" for spin merging — never as a negative gap that would force a
    spurious split.
    """
    delta = (_parse_iso_utc(next_start_iso) - _parse_iso_utc(end_iso)).total_seconds()
    return max(0.0, delta)


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
class CommercialBrandRow:
    """Commercial airplay rolled up by canonical brand.

    ``brand`` is ``None`` for the unbranded bucket — COMMERCIAL events the
    funnel detected but could not attribute to an advertiser (no brand
    extracted, so no ``commercial_id`` was assigned). ``distinct_ads`` counts
    the unique ``commercials`` rows that contributed, which exposes per-brand
    creative variety without fragmenting the table into one row per ad.
    """

    brand: str | None
    distinct_ads: int
    play_count: int
    total_duration_seconds: float
    last_heard_utc: str | None


@dataclass
class SongRow:
    """One song's airtime rollup.

    ``spin_count`` is the number of distinct **plays** of the song, computed
    by collapsing same-song segments that are separated by less than
    :data:`SPIN_MERGE_GAP_SECONDS` seconds. This matches radio-industry
    "spins" terminology: a 4-minute song interrupted by 5 seconds of DJ
    talkover counts as one spin, not two.

    ``segment_count`` is the raw count of ``broadcast_events`` rows for the
    song — the metric the report used to call ``play_count``. It is still
    surfaced because it shows how cleanly the funnel locked onto the song.
    """

    song_id: int | None
    artist: str | None
    title: str | None
    spin_count: int
    promo_spin_count: int
    segment_count: int
    total_duration_seconds: float
    promo_duration_seconds: float

    # Compatibility alias for older call sites that read ``play_count``.
    @property
    def play_count(self) -> int:
        return self.segment_count

    @property
    def full_spin_count(self) -> int:
        """Spins that exceeded :data:`PROMO_MAX_SPIN_SECONDS` (i.e. real plays)."""
        return max(0, self.spin_count - self.promo_spin_count)

    @property
    def is_promo_only(self) -> bool:
        """True when every detected spin looked like a station promo clip."""
        return self.spin_count > 0 and self.spin_count == self.promo_spin_count


@dataclass
class SongTimelineRow:
    """One SONG event in chronological order."""

    start_utc: str
    end_utc: str | None
    duration_seconds: float | None
    song_id: int | None
    artist: str | None
    title: str | None
    confidence: float | None
    detection_source: str


SPIN_MERGE_GAP_SECONDS: float = 60.0
"""Maximum gap between same-song segments that still counts as one spin.

Chosen to be larger than typical Tier 3 / DJ-talkover interruptions but
much smaller than a station's "song repeats per show" cadence, so two real
plays of the same track stay separate.
"""


@dataclass
class ArtistRow:
    """One artist's airtime rollup across all titles.

    ``spin_count`` is the total number of distinct plays summed across every
    title the artist had on air. ``distinct_titles`` is how many unique songs
    contributed to those spins. ``segment_count`` is the raw ``broadcast_events``
    row count (kept for transparency, same way :class:`SongRow` does).
    """

    artist: str
    spin_count: int
    promo_spin_count: int
    distinct_titles: int
    segment_count: int
    total_duration_seconds: float
    promo_duration_seconds: float

    @property
    def full_spin_count(self) -> int:
        return max(0, self.spin_count - self.promo_spin_count)

    @property
    def is_promo_only(self) -> bool:
        return self.spin_count > 0 and self.spin_count == self.promo_spin_count


@dataclass
class SpinStats:
    """Per-song spin tally with a separate promo subtotal.

    ``spin_count`` is the total number of distinct plays after collapsing
    short-gap segments; ``promo_spin_count`` counts the subset whose total
    spin duration is below :data:`PROMO_MAX_SPIN_SECONDS`. The corresponding
    durations tell us how much of the airtime was just promo clips vs full
    plays.
    """

    spin_count: int = 0
    promo_spin_count: int = 0
    total_duration_seconds: float = 0.0
    promo_duration_seconds: float = 0.0

    def add(self, other: "SpinStats") -> None:
        self.spin_count += other.spin_count
        self.promo_spin_count += other.promo_spin_count
        self.total_duration_seconds += other.total_duration_seconds
        self.promo_duration_seconds += other.promo_duration_seconds


def _count_spins(
    segments: list[tuple[str, str | None, float | None]],
    gap_seconds: float,
    *,
    promo_max_spin_seconds: float = PROMO_MAX_SPIN_SECONDS,
) -> SpinStats:
    """Walk one song's segments in start-order and return a :class:`SpinStats`.

    Consecutive segments separated by less than ``gap_seconds`` of wall-clock
    silence are folded into the same spin. Overlapping segments (which can
    happen with the sliding-window classifier) count as zero gap, never as a
    negative gap that would force a spurious split.

    Each completed spin shorter than ``promo_max_spin_seconds`` is tallied
    separately as a promo clip. The full counts and durations always include
    those promo spins so existing report consumers see the same headline
    numbers.
    """
    stats = SpinStats()
    current_duration = 0.0
    prev_end_iso: str | None = None

    def _close_spin(duration: float) -> None:
        if duration <= 0:
            return
        stats.spin_count += 1
        stats.total_duration_seconds += duration
        if duration < promo_max_spin_seconds:
            stats.promo_spin_count += 1
            stats.promo_duration_seconds += duration

    for start_iso, end_iso, dur in segments:
        dur_f = float(dur or 0.0)
        if prev_end_iso is None or _gap_seconds(prev_end_iso, start_iso) > gap_seconds:
            _close_spin(current_duration)
            current_duration = dur_f
        else:
            current_duration += dur_f
        if end_iso is not None:
            prev_end_iso = end_iso
        elif dur_f > 0:
            derived_end = _parse_iso_utc(start_iso) + timedelta(seconds=dur_f)
            prev_end_iso = _iso(derived_end)
        else:
            prev_end_iso = start_iso
    _close_spin(current_duration)
    return stats


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


@dataclass
class SongAgeStats:
    """Age of songs played in a report window (years since release)."""

    songs_with_dates: int
    songs_missing_dates: int
    distinct_song_mean_years: float | None
    distinct_song_median_years: float | None
    airtime_weighted_mean_years: float | None
    reference_utc: str


@dataclass
class SongsAddedRow:
    song_id: int
    first_seen_utc: str
    artist: str | None
    title: str | None
    source: str
    spin_count: int
    segment_count: int


@dataclass
class RunRow:
    capture_run_id: int
    run_id: str
    started_utc: str
    ended_utc: str | None
    pipeline_version: str
    event_count: int
    duration_seconds: float | None


def _window_clause(column: str, since_utc: str, until_utc: str | None = None) -> tuple[str, list[str]]:
    clause = f"{column} >= ?"
    args = [since_utc]
    if until_utc:
        clause += f" AND {column} < ?"
        args.append(until_utc)
    return clause, args


# -------------------------------------------------------------------- queries
def commercials_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 10,
    brand: str | None = None,
) -> list[CommercialRow]:
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
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
        WHERE e.category = 'COMMERCIAL' AND {window_clause}
    """.format(window_clause=window_clause)
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


def commercials_by_brand(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 10,
) -> list[CommercialBrandRow]:
    """Roll commercial airplay up by canonical brand.

    Unlike :func:`commercials_top` (one row per ``commercial_id``), this folds
    every ad for an advertiser into a single row and surfaces the unbranded
    bucket (``brand is None``) explicitly. It is the dashboard-friendly view:
    naming variants and same-ad-different-offset duplicates no longer split a
    brand across several rows.

    Brand names are folded through :func:`canonicalize_brand` at query time so
    alias variants (e.g. ``Ethos Insurance`` → ``Ethos``) collapse even when the
    underlying ``brands`` rows were inserted before the alias existed.
    """
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    sql = """
        SELECT
            b.canonical_name AS brand,
            e.commercial_id AS commercial_id,
            COUNT(*) AS play_count,
            COALESCE(SUM(e.duration), 0.0) AS total_duration,
            MAX(e.timestamp_start) AS last_heard_utc
        FROM broadcast_events e
        LEFT JOIN commercials c ON e.commercial_id = c.id
        LEFT JOIN brands b ON COALESCE(c.brand_id, e.brand_id) = b.id
        WHERE e.category = 'COMMERCIAL' AND {window_clause}
        GROUP BY b.canonical_name, e.commercial_id
    """.format(window_clause=window_clause)
    rows = store.connection.execute(sql, args).fetchall()

    folded: dict[str | None, dict] = {}
    for raw_brand, commercial_id, play_count, total_duration, last_heard in rows:
        canonical = canonicalize_brand(raw_brand) if raw_brand else None
        bucket = folded.setdefault(
            canonical,
            {"distinct_ads": set(), "play_count": 0, "total_duration": 0.0, "last_heard": None},
        )
        if commercial_id is not None:
            bucket["distinct_ads"].add(int(commercial_id))
        bucket["play_count"] += int(play_count)
        bucket["total_duration"] += float(total_duration or 0.0)
        if last_heard and (bucket["last_heard"] is None or last_heard > bucket["last_heard"]):
            bucket["last_heard"] = last_heard

    result = [
        CommercialBrandRow(
            brand=brand,
            distinct_ads=len(data["distinct_ads"]),
            play_count=data["play_count"],
            total_duration_seconds=data["total_duration"],
            last_heard_utc=data["last_heard"],
        )
        for brand, data in folded.items()
    ]
    result.sort(key=lambda r: (-r.play_count, -r.total_duration_seconds))
    return result[:top_n]


def brands_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 10,
) -> list[BrandRow]:
    window_clause, args = _window_clause("bm.heard_utc", since_utc, until_utc)
    sql = """
        SELECT
            b.canonical_name AS brand,
            SUM(CASE WHEN bm.mention_type = 'paid_ad' THEN 1 ELSE 0 END) AS paid_ad,
            SUM(CASE WHEN bm.mention_type = 'dj_shoutout' THEN 1 ELSE 0 END) AS dj_shoutout,
            SUM(CASE WHEN bm.mention_type = 'tag' THEN 1 ELSE 0 END) AS tag
        FROM brand_mentions bm
        JOIN brands b ON bm.brand_id = b.id
        WHERE {window_clause}
        GROUP BY b.canonical_name
        ORDER BY (paid_ad + dj_shoutout + tag) DESC, brand ASC
        LIMIT ?
    """.format(window_clause=window_clause)
    args.append(top_n)
    rows = store.connection.execute(sql, args).fetchall()
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
    until_utc: str | None = None,
    top_n: int = 10,
    spin_merge_gap_seconds: float | None = None,
) -> list[SongRow]:
    """Return the top songs by spin count over ``since_utc..now``.

    A *spin* is one full broadcast play of a song. Because Tier 1 / Tier 3 /
    DJ talkover can briefly interrupt detection mid-song, we group the raw
    ``broadcast_events`` rows for each song by identity, sort them by
    ``timestamp_start``, and collapse consecutive segments that are within
    ``spin_merge_gap_seconds`` of each other into one spin.

    Two truly separate plays of the same track (e.g. morning + afternoon
    rotation) stay as two spins because the gap between them exceeds the
    merge window.
    """
    gap = SPIN_MERGE_GAP_SECONDS if spin_merge_gap_seconds is None else float(spin_merge_gap_seconds)
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    raw = store.connection.execute(
        """
        SELECT
            e.song_id AS song_id,
            COALESCE(s.artist, e.artist) AS artist,
            COALESCE(s.title, e.track_title) AS title,
            e.timestamp_start,
            e.timestamp_end,
            e.duration
        FROM broadcast_events e
        LEFT JOIN songs s ON e.song_id = s.id
        WHERE e.category = 'SONG' AND {window_clause}
        ORDER BY e.song_id IS NULL, e.song_id, e.timestamp_start ASC
        """.format(window_clause=window_clause),
        args,
    ).fetchall()

    bucket: dict[tuple[int | None, str | None, str | None], list[tuple[str, str | None, float | None]]] = {}
    order: list[tuple[int | None, str | None, str | None]] = []
    for r in raw:
        key = (r[0], r[1], r[2])
        if key not in bucket:
            bucket[key] = []
            order.append(key)
        bucket[key].append((r[3], r[4], r[5]))

    songs: list[SongRow] = []
    for key in order:
        segments = bucket[key]
        stats = _count_spins(segments, gap)
        songs.append(
            SongRow(
                song_id=key[0],
                artist=key[1],
                title=key[2],
                spin_count=stats.spin_count,
                promo_spin_count=stats.promo_spin_count,
                segment_count=len(segments),
                total_duration_seconds=stats.total_duration_seconds,
                promo_duration_seconds=stats.promo_duration_seconds,
            )
        )

    # Sort by *real* (non-promo) spins first so promo-only entries fall to the
    # bottom of the list even when their raw ``spin_count`` is large. Ties
    # break by non-promo airtime so two songs with identical real-spin counts
    # are ordered by how much genuine play time they got (not by how many
    # promo clips they collected). Total airtime and segment count are
    # appended as final tiebreakers to preserve stable behaviour for legacy
    # data that has no promo spins.
    def _sort_key(s: SongRow) -> tuple:
        full_airtime = s.total_duration_seconds - s.promo_duration_seconds
        return (
            s.full_spin_count,
            full_airtime,
            s.spin_count,
            s.total_duration_seconds,
            s.segment_count,
        )

    songs.sort(key=_sort_key, reverse=True)
    return songs[: max(0, int(top_n))]


def songs_timeline(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    limit: int = 500,
) -> list[SongTimelineRow]:
    """Return SONG events in broadcast order.

    Unlike :func:`songs_top`, this intentionally does not collapse segments into
    spins. It is a raw chronological listening log for manual review and
    debugging.
    """
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    args.append(limit)
    rows = store.connection.execute(
        """
        SELECT
            e.timestamp_start,
            e.timestamp_end,
            e.duration,
            e.song_id,
            COALESCE(s.artist, e.artist) AS artist,
            COALESCE(s.title, e.track_title) AS title,
            e.confidence,
            COALESCE(s.source, 'unknown') AS detection_source
        FROM broadcast_events e
        LEFT JOIN songs s ON e.song_id = s.id
        WHERE e.category = 'SONG' AND {window_clause}
        ORDER BY e.timestamp_start ASC
        LIMIT ?
        """.format(window_clause=window_clause),
        args,
    ).fetchall()
    return [
        SongTimelineRow(
            start_utc=str(r[0]),
            end_utc=r[1],
            duration_seconds=(float(r[2]) if r[2] is not None else None),
            song_id=r[3],
            artist=r[4],
            title=r[5],
            confidence=(float(r[6]) if r[6] is not None else None),
            detection_source=str(r[7] or "unknown"),
        )
        for r in rows
    ]


def _artist_dedupe_key(artist: str) -> str:
    """Stable case- and whitespace-insensitive key for dedup.

    Collapses internal whitespace and trims, so ``"foo  fighters "`` and
    ``"Foo Fighters"`` hash to the same bucket. Returns ``""`` for blank
    input — callers must skip empty keys.
    """
    return " ".join(artist.split()).casefold()


def artists_top(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 10,
    spin_merge_gap_seconds: float | None = None,
) -> list[ArtistRow]:
    """Per-artist airtime rollup with case-folded dedup.

    Returns one :class:`ArtistRow` per *artist* (not per song), summing spins
    across every title the artist had on air. Two casing/whitespace variants
    of the same artist name ("Foo Fighters" and "FOO FIGHTERS") collapse into
    a single row; the most-frequent original casing wins for display.

    Spin counting reuses the same per-title logic as :func:`songs_top`, so an
    artist with two distinct songs (each played once cleanly) reports
    ``spins=2, distinct_titles=2``, while an artist whose one song hit two
    overlapping detection windows reports ``spins=1, distinct_titles=1,
    segment_count=2``.
    """
    gap = SPIN_MERGE_GAP_SECONDS if spin_merge_gap_seconds is None else float(spin_merge_gap_seconds)
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    raw = store.connection.execute(
        """
        SELECT
            e.song_id AS song_id,
            COALESCE(s.artist, e.artist) AS artist,
            COALESCE(s.title, e.track_title) AS title,
            e.timestamp_start,
            e.timestamp_end,
            e.duration
        FROM broadcast_events e
        LEFT JOIN songs s ON e.song_id = s.id
        WHERE e.category = 'SONG' AND {window_clause}
        ORDER BY e.timestamp_start ASC
        """.format(window_clause=window_clause),
        args,
    ).fetchall()

    # Per artist: track each (song_id, title) sub-bucket separately so we can
    # count spins per title and then sum across titles. Also tally original
    # casing so we can display the most-common variant. ``title_key`` falls
    # back to a folded title when ``song_id`` is NULL (an unidentified track),
    # so two distinct unknown songs by the same artist don't collide.
    from collections import Counter

    per_artist_titles: dict[str, dict[tuple[int | None, str], list[tuple[str, str | None, float | None]]]] = {}
    per_artist_casing: dict[str, Counter[str]] = {}
    per_artist_order: list[str] = []

    for song_id, artist_raw, title_raw, start_iso, end_iso, dur in raw:
        if artist_raw is None:
            continue
        artist_str = str(artist_raw).strip()
        if not artist_str:
            continue
        key = _artist_dedupe_key(artist_str)
        if not key:
            continue
        if key not in per_artist_titles:
            per_artist_titles[key] = {}
            per_artist_casing[key] = Counter()
            per_artist_order.append(key)
        per_artist_casing[key][artist_str] += 1

        title_key: tuple[int | None, str]
        if song_id is not None:
            title_key = (int(song_id), "")
        else:
            title_lower = (str(title_raw).strip().casefold() if title_raw else "")
            title_key = (None, title_lower)
        per_artist_titles[key].setdefault(title_key, []).append((start_iso, end_iso, dur))

    rows: list[ArtistRow] = []
    for key in per_artist_order:
        titles = per_artist_titles[key]
        casing = per_artist_casing[key]
        agg = SpinStats()
        total_segments = 0
        for segments in titles.values():
            agg.add(_count_spins(segments, gap))
            total_segments += len(segments)
        display = casing.most_common(1)[0][0] if casing else key
        rows.append(
            ArtistRow(
                artist=display,
                spin_count=agg.spin_count,
                promo_spin_count=agg.promo_spin_count,
                distinct_titles=len(titles),
                segment_count=total_segments,
                total_duration_seconds=agg.total_duration_seconds,
                promo_duration_seconds=agg.promo_duration_seconds,
            )
        )

    # Mirror ``songs_top``: order by real (non-promo) spins first so artists
    # whose airtime is dominated by station promos don't outrank artists with
    # a single genuine play. Non-promo airtime is the next tiebreaker so two
    # artists with identical real spin counts sort by how much real airtime
    # they accumulated, not by promo clip volume.
    def _artist_sort_key(r: ArtistRow) -> tuple:
        full_airtime = r.total_duration_seconds - r.promo_duration_seconds
        return (
            -r.full_spin_count,
            -full_airtime,
            -r.spin_count,
            -r.total_duration_seconds,
            -r.distinct_titles,
            r.artist.casefold(),
        )

    rows.sort(key=_artist_sort_key)
    return rows[: max(0, int(top_n))]


def timeline(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    limit: int = 500,
) -> list[TimelineRow]:
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    args.append(limit)
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
        WHERE {window_clause}
        ORDER BY e.timestamp_start ASC
        LIMIT ?
    """.format(window_clause=window_clause)
    rows = store.connection.execute(sql, args).fetchall()
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
    until_utc: str | None = None,
) -> list[SummaryRow]:
    window_clause, args = _window_clause("timestamp_start", since_utc, until_utc)
    sql = """
        SELECT category,
               COUNT(*) AS segment_count,
               COALESCE(SUM(duration), 0.0) AS total_duration
        FROM broadcast_events
        WHERE {window_clause}
        GROUP BY category
        ORDER BY total_duration DESC
    """.format(window_clause=window_clause)
    rows = store.connection.execute(sql, args).fetchall()
    return [
        SummaryRow(
            category=str(r[0]),
            segment_count=int(r[1]),
            total_duration_seconds=float(r[2] or 0.0),
        )
        for r in rows
    ]


def song_age_stats(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    reference_utc: str | None = None,
) -> SongAgeStats:
    """Mean/median song age for distinct songs heard in the window.

    * **Distinct-song mean/median** — one age per ``song_id`` with a known
      ``release_date``.
    * **Airtime-weighted mean** — weights each song's age by total ``SONG``
      event duration in the window (closer to "how old was the music we
      actually aired?").
    """
    window_clause, args = _window_clause("e.timestamp_start", since_utc, until_utc)
    sql = f"""
        SELECT
            s.id,
            s.release_date,
            COALESCE(SUM(e.duration), 0.0) AS song_airtime
        FROM broadcast_events e
        JOIN songs s ON s.id = e.song_id
        WHERE e.category = 'SONG'
          AND {window_clause}
        GROUP BY s.id, s.release_date
    """
    rows = store.connection.execute(sql, args).fetchall()
    reference = parse_reference_utc(reference_utc or until_utc)
    reference_iso = reference.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    ages: list[float] = []
    weighted_num = 0.0
    weighted_den = 0.0
    missing = 0

    for r in rows:
        release_date = r[1]
        airtime = float(r[2] or 0.0)
        if not release_date:
            missing += 1
            continue
        try:
            age = song_age_years(str(release_date), reference)
        except ValueError:
            missing += 1
            continue
        ages.append(age)
        if airtime > 0:
            weighted_num += age * airtime
            weighted_den += airtime

    mean_years = statistics.mean(ages) if ages else None
    median_years = statistics.median(ages) if ages else None
    airtime_mean = (weighted_num / weighted_den) if weighted_den > 0 else None

    return SongAgeStats(
        songs_with_dates=len(ages),
        songs_missing_dates=missing,
        distinct_song_mean_years=mean_years,
        distinct_song_median_years=median_years,
        airtime_weighted_mean_years=airtime_mean,
        reference_utc=reference_iso,
    )


def songs_added(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    source: str | None = None,
    top_n: int = 50,
) -> list[SongsAddedRow]:
    song_window, args = _window_clause("s.first_seen_utc", since_utc, until_utc)
    event_window, event_args = _window_clause("e.timestamp_start", since_utc, until_utc)
    args = event_args + args
    where = [song_window]
    if source:
        where.append("s.source = ?")
        args.append(source)
    args.append(top_n)
    sql = """
        SELECT
            s.id,
            s.first_seen_utc,
            s.artist,
            s.title,
            s.source,
            COUNT(e.id) AS segment_count
        FROM songs s
        LEFT JOIN broadcast_events e
          ON e.song_id = s.id
         AND {event_window}
        WHERE {where_clause}
        GROUP BY s.id
        ORDER BY s.first_seen_utc DESC, s.id DESC
        LIMIT ?
    """.format(event_window=event_window, where_clause=" AND ".join(where))
    rows = store.connection.execute(sql, args).fetchall()
    return [
        SongsAddedRow(
            song_id=int(r[0]),
            first_seen_utc=str(r[1]),
            artist=r[2],
            title=r[3],
            source=str(r[4]),
            spin_count=int(r[5] or 0),
            segment_count=int(r[5] or 0),
        )
        for r in rows
    ]


def runs_summary(
    store: BroadcastStore,
    *,
    since_utc: str,
    until_utc: str | None = None,
    top_n: int = 50,
) -> list[RunRow]:
    window_clause, args = _window_clause("cr.started_utc", since_utc, until_utc)
    args.append(top_n)
    rows = store.connection.execute(
        """
        SELECT
            cr.id,
            cr.run_id,
            cr.started_utc,
            cr.ended_utc,
            cr.pipeline_version,
            COUNT(be.id) AS event_count
        FROM capture_runs cr
        LEFT JOIN broadcast_events be ON be.capture_run_id = cr.id
        WHERE {window_clause}
        GROUP BY cr.id
        ORDER BY cr.started_utc DESC, cr.id DESC
        LIMIT ?
        """.format(window_clause=window_clause),
        args,
    ).fetchall()
    out: list[RunRow] = []
    for r in rows:
        duration = None
        if r[3]:
            duration = (_parse_iso_utc(str(r[3])) - _parse_iso_utc(str(r[2]))).total_seconds()
        out.append(
            RunRow(
                capture_run_id=int(r[0]),
                run_id=str(r[1]),
                started_utc=str(r[2]),
                ended_utc=r[3],
                pipeline_version=str(r[4]),
                event_count=int(r[5] or 0),
                duration_seconds=duration,
            )
        )
    return out
