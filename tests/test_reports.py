"""Reports CLI subcommand queries + formatters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radio_classifier.persistence import BroadcastStore
from radio_classifier.reports import (
    artists_top,
    brands_top,
    commercials_by_brand,
    commercials_top,
    format_artists,
    format_brands,
    format_commercials,
    format_songs,
    format_songs_timeline,
    format_summary,
    format_timeline,
    parse_since,
    render_dashboard_html,
    songs_timeline,
    songs_top,
    summary,
    timeline,
)
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _seed_db(tmp_path: Path) -> BroadcastStore:
    store = BroadcastStore(tmp_path / "rc.db")
    now = datetime.now(tz=timezone.utc)

    geico = store.upsert_brand("Geico")
    toyota = store.upsert_brand("Toyota")
    song = store.upsert_song(artist="Taylor Swift", title="Anti-Hero")
    geico_ad = store.insert_commercial(
        brand_id=geico,
        duration_bucket_seconds=15,
        minhash_hex="00" * 8,
        reference_transcript="save fifteen percent",
    )
    toyota_ad = store.insert_commercial(
        brand_id=toyota,
        duration_bucket_seconds=30,
        minhash_hex="11" * 8,
        reference_transcript="lets go places",
    )

    # 2 SONG plays
    for i in range(2):
        start = now - timedelta(minutes=10 + i * 5)
        end = start + timedelta(seconds=180)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.SONG,
                artist="Taylor Swift",
                track_title="Anti-Hero",
                song_id=song,
            )
        )
    # 3 Geico ads + 1 Toyota
    for i in range(3):
        start = now - timedelta(minutes=8 + i)
        end = start + timedelta(seconds=15)
        ev_id = store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.COMMERCIAL,
                brand_name="Geico",
                brand_id=geico,
                commercial_id=geico_ad,
            )
        )
        store.insert_brand_mention(
            segment_id=ev_id, brand_id=geico, mention_type="paid_ad", heard_utc=_iso(start)
        )
    start = now - timedelta(minutes=2)
    end = start + timedelta(seconds=30)
    ev_id = store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(end),
            category=BroadcastCategory.COMMERCIAL,
            brand_name="Toyota",
            brand_id=toyota,
            commercial_id=toyota_ad,
        )
    )
    store.insert_brand_mention(
        segment_id=ev_id, brand_id=toyota, mention_type="paid_ad", heard_utc=_iso(start)
    )
    # 1 DJ with a Toyota shoutout
    dj_start = now - timedelta(minutes=1)
    dj_end = dj_start + timedelta(seconds=20)
    ev_id = store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(dj_start),
            timestamp_end=_iso(dj_end),
            category=BroadcastCategory.DJ,
        )
    )
    store.insert_brand_mention(
        segment_id=ev_id, brand_id=toyota, mention_type="dj_shoutout", heard_utc=_iso(dj_start)
    )
    return store


def test_parse_since_accepts_relative() -> None:
    out = parse_since("1h")
    assert out.endswith("Z")
    out = parse_since("30m")
    assert out.endswith("Z")


def test_parse_since_rejects_bad() -> None:
    with pytest.raises(ValueError):
        parse_since("yesterday")


def test_commercials_top_orders_by_play_count(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = commercials_top(store, since_utc=parse_since("1d"))
        assert rows[0].brand == "Geico"
        assert rows[0].play_count == 3
        assert any(r.brand == "Toyota" and r.play_count == 1 for r in rows)
    finally:
        store.close()


def test_commercials_by_brand_folds_aliases_and_surfaces_unbranded(tmp_path: Path) -> None:
    """Brand rollup collapses alias variants and exposes the no-brand bucket."""
    store = BroadcastStore(tmp_path / "rollup.db")
    try:
        now = datetime.now(tz=timezone.utc)
        # Two physically separate brand rows that canonicalize to one advertiser.
        ethos = store.upsert_brand("Ethos")
        ethos_ins = store.upsert_brand("Ethos Insurance")
        ad_a = store.insert_commercial(
            brand_id=ethos,
            duration_bucket_seconds=20,
            minhash_hex="aa" * 8,
            reference_transcript="get a quote in seconds",
        )
        ad_b = store.insert_commercial(
            brand_id=ethos_ins,
            duration_bucket_seconds=20,
            minhash_hex="bb" * 8,
            reference_transcript="no medical exam life insurance",
        )
        for idx, (brand_name, brand_id, ad) in enumerate(
            [("Ethos", ethos, ad_a), ("Ethos Insurance", ethos_ins, ad_b)]
        ):
            start = now - timedelta(minutes=10 + idx)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=20)),
                    category=BroadcastCategory.COMMERCIAL,
                    brand_name=brand_name,
                    brand_id=brand_id,
                    commercial_id=ad,
                )
            )
        # An unbranded commercial event: detected COMMERCIAL, no brand/identity.
        start = now - timedelta(minutes=5)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(start + timedelta(seconds=10)),
                category=BroadcastCategory.COMMERCIAL,
            )
        )

        rows = commercials_by_brand(store, since_utc=parse_since("1d"))
        by_brand = {r.brand: r for r in rows}

        # Both Ethos variants fold into a single canonical row with 2 distinct ads.
        assert "Ethos" in by_brand
        assert "Ethos Insurance" not in by_brand
        assert by_brand["Ethos"].distinct_ads == 2
        assert by_brand["Ethos"].play_count == 2

        # The unbranded bucket is surfaced explicitly with no distinct-ad count.
        assert None in by_brand
        assert by_brand[None].play_count == 1
        assert by_brand[None].distinct_ads == 0
    finally:
        store.close()


def test_brands_top_aggregates_all_mention_types(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = brands_top(store, since_utc=parse_since("1d"))
        by_name = {r.brand: r for r in rows}
        assert by_name["Geico"].paid_play_count == 3
        assert by_name["Toyota"].paid_play_count == 1
        assert by_name["Toyota"].dj_shoutout_count == 1
    finally:
        store.close()


def test_songs_top_counts_plays(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = songs_top(store, since_utc=parse_since("1d"))
        # Two distinct 3-minute plays starting 5 minutes apart should count
        # as 2 spins and 2 segments. ``play_count`` is the legacy alias.
        match = next(r for r in rows if r.title == "Anti-Hero")
        assert match.spin_count == 2
        assert match.segment_count == 2
        assert match.play_count == match.segment_count
    finally:
        store.close()


def test_songs_top_collapses_segments_within_gap_into_one_spin(tmp_path: Path) -> None:
    """A song split into multiple short segments by a tiny gap should report
    as ONE spin, with the segment count preserved for transparency.

    This is the radio-industry "spins" notion: one airing of the track is
    one spin regardless of how many DB rows the funnel produced for it.
    """
    store = BroadcastStore(tmp_path / "spins.db")
    try:
        song = store.upsert_song(artist="Linkin Park", title="Numb")
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        # Three contiguous segments of "Numb" with a 5s gap between each —
        # simulating brief Tier 1 dropouts mid-song. Total airtime ~3m20s.
        offsets = [(0, 100), (105, 80), (190, 40)]
        for off, dur in offsets:
            start = base + timedelta(seconds=off)
            end = start + timedelta(seconds=dur)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(end),
                    category=BroadcastCategory.SONG,
                    artist="Linkin Park",
                    track_title="Numb",
                    song_id=song,
                )
            )

        rows = songs_top(store, since_utc=parse_since("1d"))
        match = next(r for r in rows if r.title == "Numb")
        assert match.spin_count == 1, "small gaps should not split a single spin"
        assert match.segment_count == 3
        assert match.total_duration_seconds == pytest.approx(100 + 80 + 40)
    finally:
        store.close()


def test_songs_top_splits_far_apart_segments_into_separate_spins(tmp_path: Path) -> None:
    """Two real plays of the same song separated by minutes should count as
    two spins."""
    store = BroadcastStore(tmp_path / "spins.db")
    try:
        song = store.upsert_song(artist="Nirvana", title="Lithium")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        # First spin at base, second spin 30 minutes later. Each is one
        # 3-minute segment.
        for offset_min in (0, 30):
            start = base + timedelta(minutes=offset_min)
            end = start + timedelta(seconds=180)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(end),
                    category=BroadcastCategory.SONG,
                    artist="Nirvana",
                    track_title="Lithium",
                    song_id=song,
                )
            )

        rows = songs_top(store, since_utc=parse_since("1d"))
        match = next(r for r in rows if r.title == "Lithium")
        assert match.spin_count == 2
        assert match.segment_count == 2


    finally:
        store.close()


def test_songs_top_flags_short_clip_spins_as_promos(tmp_path: Path) -> None:
    """A song that the station only ever plays as 10-60 second teaser clips
    should be tagged as promo-only.

    This is the Julia Wolf pattern observed in the morning 12-hour run: 11
    spins averaging 41 seconds each is unmistakably a station promo, not
    eleven full plays. We expect ``spin_count`` to still match the raw
    play tally so existing dashboards don't lose data, but ``promo_spin_count``
    should equal it and ``full_spin_count`` / ``is_promo_only`` should reflect
    that none of those spins look like real airings.
    """
    store = BroadcastStore(tmp_path / "promos.db")
    try:
        song = store.upsert_song(artist="Julia Wolf", title="In My Room (Acoustic)")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=4)
        # Three well-separated promo windows, each only 30 seconds long.
        # The gap between them (>30 minutes) keeps them as separate spins.
        for offset_min in (0, 45, 120):
            start = base + timedelta(minutes=offset_min)
            end = start + timedelta(seconds=30)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(end),
                    category=BroadcastCategory.SONG,
                    artist="Julia Wolf",
                    track_title="In My Room (Acoustic)",
                    song_id=song,
                )
            )

        rows = songs_top(store, since_utc=parse_since("1d"))
        match = next(r for r in rows if r.title == "In My Room (Acoustic)")
        assert match.spin_count == 3
        assert match.promo_spin_count == 3
        assert match.full_spin_count == 0
        assert match.is_promo_only is True
        assert match.promo_duration_seconds == pytest.approx(90.0)
        assert match.total_duration_seconds == pytest.approx(90.0)
    finally:
        store.close()


def test_songs_top_counts_long_and_short_spins_separately(tmp_path: Path) -> None:
    """A real song with one full play and one short teaser should report
    both, with the promo subtotal isolated for filtering."""
    store = BroadcastStore(tmp_path / "promos.db")
    try:
        song = store.upsert_song(artist="Tame Impala", title="Dracula")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        # One genuine 3m20s play.
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=200)),
                category=BroadcastCategory.SONG,
                artist="Tame Impala",
                track_title="Dracula",
                song_id=song,
            )
        )
        # A 25-second teaser an hour later — should be flagged as promo only.
        teaser_start = base + timedelta(minutes=60)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(teaser_start),
                timestamp_end=_iso(teaser_start + timedelta(seconds=25)),
                category=BroadcastCategory.SONG,
                artist="Tame Impala",
                track_title="Dracula",
                song_id=song,
            )
        )

        rows = songs_top(store, since_utc=parse_since("1d"))
        match = next(r for r in rows if r.title == "Dracula")
        assert match.spin_count == 2
        assert match.promo_spin_count == 1
        assert match.full_spin_count == 1
        assert match.is_promo_only is False
        assert match.promo_duration_seconds == pytest.approx(25.0)
        assert match.total_duration_seconds == pytest.approx(225.0)
    finally:
        store.close()


def test_songs_top_tiebreaks_equal_real_spins_by_non_promo_airtime(tmp_path: Path) -> None:
    """Two songs with identical full_spin_count should sort by non-promo
    airtime, not by total (promo-inflated) airtime.

    This catches a real bug observed against the 12-hour DB: Royel Otis with
    3 clean spins (~8 minutes) should outrank Julia Wolf with 3 marginal
    spins + 8 promo clips (~3 minutes of real airtime padded to ~7m30s by
    teasers).
    """
    store = BroadcastStore(tmp_path / "tiebreak.db")
    try:
        clean = store.upsert_song(artist="Royel Otis", title="Clean Song")
        padded = store.upsert_song(artist="Promo Heavy", title="Padded Song")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=4)

        # Clean Song: 3 full plays of 160s each => 8m00s of real airtime.
        for i, offset_min in enumerate((0, 30, 60)):
            start = base + timedelta(minutes=offset_min)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=160)),
                    category=BroadcastCategory.SONG,
                    artist="Royel Otis",
                    track_title="Clean Song",
                    song_id=clean,
                )
            )

        # Padded Song: 3 full plays of 100s + 8 promo clips of 20s each.
        for offset_min in (0, 35, 70):
            start = base + timedelta(minutes=offset_min)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=100)),
                    category=BroadcastCategory.SONG,
                    artist="Promo Heavy",
                    track_title="Padded Song",
                    song_id=padded,
                )
            )
        for i in range(8):
            promo_start = base + timedelta(minutes=120 + 8 * i)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(promo_start),
                    timestamp_end=_iso(promo_start + timedelta(seconds=20)),
                    category=BroadcastCategory.SONG,
                    artist="Promo Heavy",
                    track_title="Padded Song",
                    song_id=padded,
                )
            )

        rows = songs_top(store, since_utc=parse_since("1d"))
        order = [r.title for r in rows]
        clean_row = next(r for r in rows if r.title == "Clean Song")
        padded_row = next(r for r in rows if r.title == "Padded Song")
        assert clean_row.full_spin_count == padded_row.full_spin_count == 3
        # Real airtime is what should decide the order.
        assert order.index("Clean Song") < order.index("Padded Song")
    finally:
        store.close()


def test_songs_top_ranks_real_spins_above_promo_only_songs(tmp_path: Path) -> None:
    """A song with one real spin must outrank a promo-only song even when
    the promo-only one has more raw spin_count.

    Without the promo-aware sort, ``promo_spin_count=10`` would push station
    teasers to the top of the report.
    """
    store = BroadcastStore(tmp_path / "promos.db")
    try:
        real = store.upsert_song(artist="Real Band", title="Real Song")
        promo = store.upsert_song(artist="Promo Band", title="Promo Song")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=3)

        # Real Band: one 4-minute spin.
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=240)),
                category=BroadcastCategory.SONG,
                artist="Real Band",
                track_title="Real Song",
                song_id=real,
            )
        )
        # Promo Band: ten 30-second clips, well-separated.
        for i in range(10):
            promo_start = base + timedelta(minutes=10 + 12 * i)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(promo_start),
                    timestamp_end=_iso(promo_start + timedelta(seconds=30)),
                    category=BroadcastCategory.SONG,
                    artist="Promo Band",
                    track_title="Promo Song",
                    song_id=promo,
                )
            )

        rows = songs_top(store, since_utc=parse_since("1d"))
        order = [r.title for r in rows]
        assert order.index("Real Song") < order.index("Promo Song")
    finally:
        store.close()


def test_songs_top_orders_by_spins_then_airtime(tmp_path: Path) -> None:
    """A song with more spins ranks above a song with more airtime."""
    store = BroadcastStore(tmp_path / "spins.db")
    try:
        a = store.upsert_song(artist="Artist A", title="Two Spin Track")
        b = store.upsert_song(artist="Artist B", title="One Long Track")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=2)

        # Artist A: 2 spins of 2 minutes each = 4 minutes total
        for offset_min in (0, 20):
            start = base + timedelta(minutes=offset_min)
            end = start + timedelta(seconds=120)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(end),
                    category=BroadcastCategory.SONG,
                    artist="Artist A",
                    track_title="Two Spin Track",
                    song_id=a,
                )
            )

        # Artist B: 1 spin of 10 minutes
        start = base + timedelta(minutes=40)
        end = start + timedelta(seconds=600)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.SONG,
                artist="Artist B",
                track_title="One Long Track",
                song_id=b,
            )
        )

        rows = songs_top(store, since_utc=parse_since("1d"))
        ordered_titles = [r.title for r in rows]
        assert ordered_titles.index("Two Spin Track") < ordered_titles.index("One Long Track")
    finally:
        store.close()


def test_artists_top_dedupes_case_and_sums_spins_across_titles(tmp_path: Path) -> None:
    """Same artist with mixed casing collapses to one row and sums spins
    across every title they had on air."""
    store = BroadcastStore(tmp_path / "artists.db")
    try:
        # Two Foo Fighters songs, both with the "real" casing.
        ff1 = store.upsert_song(artist="Foo Fighters", title="Times Like These")
        ff2 = store.upsert_song(artist="Foo Fighters", title="The Pretender")
        # Greenday with one song, written two different ways in the events to
        # exercise case-fold dedup. We don't go through ``upsert_song`` here
        # because we want the raw event artist string to vary; the LEFT JOIN
        # with ``songs`` is what would otherwise normalize it.
        base = datetime.now(tz=timezone.utc) - timedelta(hours=2)

        def _put(start: datetime, dur: int, *, artist: str, title: str, song_id: int | None) -> None:
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=dur)),
                    category=BroadcastCategory.SONG,
                    artist=artist,
                    track_title=title,
                    song_id=song_id,
                )
            )

        # Foo Fighters: 1 spin of Times Like These (3 mins) + 1 spin of The
        # Pretender (4 mins) — should report spins=2, titles=2.
        _put(base, 180, artist="Foo Fighters", title="Times Like These", song_id=ff1)
        _put(base + timedelta(minutes=10), 240, artist="Foo Fighters", title="The Pretender", song_id=ff2)

        # Unidentified Greenday play with two casing variants in raw event
        # text — should collapse to ONE artist row, with the casing that
        # appears most frequently winning. We deliberately use song_id=None
        # so the row goes through the raw-artist code path.
        _put(base + timedelta(minutes=30), 60, artist="Green Day", title="Holiday", song_id=None)
        _put(base + timedelta(minutes=31), 60, artist="green day", title="Holiday", song_id=None)
        _put(base + timedelta(minutes=32), 60, artist="GREEN DAY", title="Holiday", song_id=None)

        rows = artists_top(store, since_utc=parse_since("1d"))
        by_artist = {r.artist.casefold(): r for r in rows}

        ff_row = by_artist["foo fighters"]
        assert ff_row.spin_count == 2
        assert ff_row.distinct_titles == 2
        assert ff_row.segment_count == 2
        assert ff_row.total_duration_seconds == pytest.approx(180 + 240)

        gd_row = by_artist["green day"]
        # The 3 raw Green Day segments are 60s apart end-to-start (each lasts
        # 60s and the next starts 60s later), so they collapse into ONE spin.
        assert gd_row.spin_count == 1
        assert gd_row.distinct_titles == 1
        assert gd_row.segment_count == 3
        assert gd_row.total_duration_seconds == pytest.approx(60 * 3)
        # The "Green Day" variant is no more common than the other two — any
        # of them is acceptable, but it must not be empty or weirdly-cased.
        assert gd_row.artist.casefold() == "green day"

        # Only Foo Fighters and Green Day should appear — no blank rows.
        assert all(r.artist.strip() for r in rows)
    finally:
        store.close()


def test_artists_top_separates_promo_spins_from_real_spins(tmp_path: Path) -> None:
    """An artist whose only airtime is promo clips reports
    ``full_spin_count=0`` and ``is_promo_only`` is true, even when
    ``spin_count`` is non-zero."""
    store = BroadcastStore(tmp_path / "promo_artist.db")
    try:
        promo = store.upsert_song(artist="Julia Wolf", title="In My Room (Acoustic)")
        real = store.upsert_song(artist="The Killers", title="Somebody Told Me")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=4)

        # Two 30-second promos for Julia Wolf, well-separated.
        for offset_min in (0, 30):
            start = base + timedelta(minutes=offset_min)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=30)),
                    category=BroadcastCategory.SONG,
                    artist="Julia Wolf",
                    track_title="In My Room (Acoustic)",
                    song_id=promo,
                )
            )
        # One full 3-minute play of a Killers song.
        killer_start = base + timedelta(minutes=70)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(killer_start),
                timestamp_end=_iso(killer_start + timedelta(seconds=180)),
                category=BroadcastCategory.SONG,
                artist="The Killers",
                track_title="Somebody Told Me",
                song_id=real,
            )
        )

        rows = artists_top(store, since_utc=parse_since("1d"))
        by_artist = {r.artist.casefold(): r for r in rows}

        julia = by_artist["julia wolf"]
        assert julia.spin_count == 2
        assert julia.promo_spin_count == 2
        assert julia.full_spin_count == 0
        assert julia.is_promo_only is True

        killers = by_artist["the killers"]
        assert killers.spin_count == 1
        assert killers.promo_spin_count == 0
        assert killers.full_spin_count == 1
        assert killers.is_promo_only is False

        order = [r.artist for r in rows]
        # Real spin must outrank promo-only artist.
        assert order.index("The Killers") < order.index("Julia Wolf")
    finally:
        store.close()


def test_artists_top_orders_by_spins_then_airtime(tmp_path: Path) -> None:
    """An artist with more spins must rank above one with more airtime."""
    store = BroadcastStore(tmp_path / "artists.db")
    try:
        short_artist_song = store.upsert_song(artist="Sprinter", title="Two Spinners")
        long_artist_song = store.upsert_song(artist="Marathoner", title="Single Long Track")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=2)

        # Sprinter: 2 spins x 2 min each = 4 minutes airtime
        for offset_min in (0, 30):
            start = base + timedelta(minutes=offset_min)
            end = start + timedelta(seconds=120)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(end),
                    category=BroadcastCategory.SONG,
                    artist="Sprinter",
                    track_title="Two Spinners",
                    song_id=short_artist_song,
                )
            )

        # Marathoner: 1 spin x 12 min = more airtime, but only one spin
        start = base + timedelta(minutes=60)
        end = start + timedelta(seconds=720)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.SONG,
                artist="Marathoner",
                track_title="Single Long Track",
                song_id=long_artist_song,
            )
        )

        rows = artists_top(store, since_utc=parse_since("1d"))
        artists_in_order = [r.artist for r in rows]
        assert artists_in_order.index("Sprinter") < artists_in_order.index("Marathoner")
    finally:
        store.close()


def test_artists_top_ignores_null_or_blank_artist(tmp_path: Path) -> None:
    """SONG events with NULL or whitespace artist must not produce a row."""
    store = BroadcastStore(tmp_path / "artists.db")
    try:
        named = store.upsert_song(artist="Real Artist", title="Real Track")
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=120)),
                category=BroadcastCategory.SONG,
                artist="Real Artist",
                track_title="Real Track",
                song_id=named,
            )
        )
        # An unknown-song event with no artist at all.
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base + timedelta(minutes=5)),
                timestamp_end=_iso(base + timedelta(minutes=6)),
                category=BroadcastCategory.SONG,
                artist=None,
                track_title=None,
                song_id=None,
            )
        )
        # A whitespace-only artist (should also be skipped).
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base + timedelta(minutes=10)),
                timestamp_end=_iso(base + timedelta(minutes=11)),
                category=BroadcastCategory.SONG,
                artist="   ",
                track_title=None,
                song_id=None,
            )
        )

        rows = artists_top(store, since_utc=parse_since("1d"))
        assert len(rows) == 1
        assert rows[0].artist == "Real Artist"
    finally:
        store.close()


def test_format_artists_renders_table(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "artists.db")
    try:
        song = store.upsert_song(artist="The Cure", title="Friday I'm In Love")
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=200)),
                category=BroadcastCategory.SONG,
                artist="The Cure",
                track_title="Friday I'm In Love",
                song_id=song,
            )
        )
        out = format_artists(artists_top(store, since_utc=parse_since("1d")))
        assert "artist" in out
        assert "spins" in out
        assert "promos" in out
        assert "titles" in out
        assert "segments" in out
        assert "The Cure" in out
    finally:
        store.close()


def test_format_songs_and_artists_decorate_promo_only_entries(tmp_path: Path) -> None:
    """A promo-only song should render with a ``[promo]`` tag in both the
    songs and artists tables, and the ``spins`` column should call out the
    promo subtotal so the headline number isn't misleading."""
    store = BroadcastStore(tmp_path / "promo_fmt.db")
    try:
        song = store.upsert_song(artist="Julia Wolf", title="In My Room (Acoustic)")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        for offset_min in (0, 45, 90):
            start = base + timedelta(minutes=offset_min)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=30)),
                    category=BroadcastCategory.SONG,
                    artist="Julia Wolf",
                    track_title="In My Room (Acoustic)",
                    song_id=song,
                )
            )

        song_out = format_songs(songs_top(store, since_utc=parse_since("1d")))
        artist_out = format_artists(artists_top(store, since_utc=parse_since("1d")))

        # The headline "spins" cell shows zero real spins plus a "(+3)" hint;
        # the title and artist labels carry a [promo] marker.
        assert "0 (+3)" in song_out
        assert "[promo]" in song_out
        assert "In My Room (Acoustic)" in song_out

        assert "0 (+3)" in artist_out
        assert "[promo]" in artist_out
        assert "Julia Wolf" in artist_out
    finally:
        store.close()


def test_timeline_returns_chronological(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = timeline(store, since_utc=parse_since("1d"))
        starts = [r.start_utc for r in rows]
        assert starts == sorted(starts)
    finally:
        store.close()


def test_songs_timeline_returns_song_events_only_in_chronological_order(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = songs_timeline(store, since_utc=parse_since("1d"), limit=10)
        assert len(rows) == 2
        assert [r.start_utc for r in rows] == sorted(r.start_utc for r in rows)
        assert {r.title for r in rows} == {"Anti-Hero"}
        assert all(r.detection_source == "audfprint" for r in rows)
        assert all(r.duration_seconds == pytest.approx(180.0) for r in rows)
    finally:
        store.close()


def test_songs_timeline_includes_unknown_song_rows(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "song-timeline.db")
    try:
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=10)),
                category=BroadcastCategory.SONG,
                artist=None,
                track_title=None,
                song_id=None,
                confidence=0.95,
            )
        )

        rows = songs_timeline(store, since_utc=parse_since("1d"))
        assert len(rows) == 1
        assert rows[0].song_id is None
        assert rows[0].artist is None
        assert rows[0].title is None
        assert rows[0].detection_source == "unknown"
        assert rows[0].confidence == pytest.approx(0.95)
    finally:
        store.close()


def test_summary_groups_by_category(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        rows = summary(store, since_utc=parse_since("1d"))
        cats = {r.category: r for r in rows}
        assert cats["SONG"].segment_count == 2
        assert cats["COMMERCIAL"].segment_count == 4
        assert cats["DJ"].segment_count == 1
    finally:
        store.close()


def test_formatters_produce_tables(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        out_c = format_commercials(commercials_top(store, since_utc=parse_since("1d")))
        out_b = format_brands(brands_top(store, since_utc=parse_since("1d")))
        out_s = format_songs(songs_top(store, since_utc=parse_since("1d")))
        out_st = format_songs_timeline(songs_timeline(store, since_utc=parse_since("1d")))
        out_t = format_timeline(timeline(store, since_utc=parse_since("1d")))
        out_sum = format_summary(summary(store, since_utc=parse_since("1d")))
        for out in (out_c, out_b, out_s, out_st, out_t, out_sum):
            assert "\n" in out
            assert "(no rows)" not in out
        # The songs table must surface the spin metric prominently.
        assert "spins" in out_s
        assert "segments" in out_s
        assert "start_utc" in out_st
        assert "source" in out_st
    finally:
        store.close()


def test_dashboard_renders_static_html_with_core_sections(tmp_path: Path) -> None:
    store = _seed_db(tmp_path)
    try:
        html = render_dashboard_html(store, since_utc=parse_since("1d"), top_n=5)
        assert "<!doctype html>" in html
        assert "Broadcast Metrics Dashboard" in html
        assert "Category Airtime" in html
        assert "Top Artists" in html
        assert "Top Songs" in html
        assert "Top Brands" in html
        assert "Top Commercials" in html
        assert "Hourly Category Mix" in html
        assert "Taylor Swift" in html
        assert "Geico" in html
        # Promo columns are always present in the rendered table headers, even
        # when the seed DB has no promo-shaped spins (Anti-Hero is a full
        # 3-minute play, so it stays out of the promo bucket).
        assert "Promos" in html
    finally:
        store.close()


def test_dashboard_highlights_promo_only_entries(tmp_path: Path) -> None:
    """Promo-only songs/artists render with the visual promo annotation."""
    store = BroadcastStore(tmp_path / "promo_dash.db")
    try:
        song = store.upsert_song(artist="Julia Wolf", title="In My Room (Acoustic)")
        base = datetime.now(tz=timezone.utc) - timedelta(hours=3)
        for offset_min in (0, 30, 60):
            start = base + timedelta(minutes=offset_min)
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=_iso(start),
                    timestamp_end=_iso(start + timedelta(seconds=30)),
                    category=BroadcastCategory.SONG,
                    artist="Julia Wolf",
                    track_title="In My Room (Acoustic)",
                    song_id=song,
                )
            )

        html = render_dashboard_html(store, since_utc=parse_since("1d"), top_n=5)
        # The [promo] decoration and the "+3 promo" pill should both land in
        # the song/artist tables (HTML-escaped or not).
        assert "promo-tag" in html
        assert "promo-pill" in html
        assert "+3 promo" in html
        assert "[promo]" in html
    finally:
        store.close()
