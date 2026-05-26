"""Reports CLI subcommand queries + formatters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radio_classifier.persistence import BroadcastStore
from radio_classifier.reports import (
    brands_top,
    commercials_top,
    format_brands,
    format_commercials,
    format_songs,
    format_summary,
    format_timeline,
    parse_since,
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
        assert any(r.title == "Anti-Hero" and r.play_count == 2 for r in rows)
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
        out_t = format_timeline(timeline(store, since_utc=parse_since("1d")))
        out_sum = format_summary(summary(store, since_utc=parse_since("1d")))
        for out in (out_c, out_b, out_s, out_t, out_sum):
            assert "\n" in out
            assert "(no rows)" not in out
    finally:
        store.close()
