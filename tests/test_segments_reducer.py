"""Reducer + normalize tests for the 5-class taxonomy."""

from __future__ import annotations

import pytest

from radio_classifier.segments.normalize import (
    normalize_token,
    segment_input_for_song,
    segment_input_for_speech,
    segment_input_for_unknown_song,
)
from radio_classifier.segments.reducer import SegmentReducer, duration_seconds
from radio_classifier.segments.types import BroadcastCategory, SegmentInput, SegmentKey


def test_normalize_token_strips_quotes_and_casefolds() -> None:
    assert normalize_token("  Foo  ") == "foo"
    assert normalize_token("O'Brien") == "o'brien"
    assert normalize_token("'quoted'") == "quoted"
    assert normalize_token(None) is None
    assert normalize_token("   ") is None


def test_duration_seconds_half_open() -> None:
    assert duration_seconds(
        "2020-01-01T00:00:00.000Z",
        "2020-01-01T00:00:10.000Z",
    ) == pytest.approx(10.0)


def test_same_key_feeds_merge_finalize_closes() -> None:
    r = SegmentReducer()
    k = SegmentKey(BroadcastCategory.SONG, "a", "b", None, song_id=42)
    assert r.feed(SegmentInput("2020-01-01T00:00:00.000Z", k, "Artist", "T1", None)) == []
    assert r.feed(SegmentInput("2020-01-01T00:00:10.000Z", k, "Artist", "T2", None)) == []
    closed = r.finalize("2020-01-01T00:00:10.000Z", 10.0)
    assert len(closed) == 1
    c = closed[0]
    assert c.category is BroadcastCategory.SONG
    assert c.song_id == 42
    assert c.track_title == "T2"  # last-non-None wins


def test_key_change_emits_closed_segment() -> None:
    r = SegmentReducer()
    k1 = SegmentKey(BroadcastCategory.DJ, None, None, None)
    k2 = SegmentKey(BroadcastCategory.COMMERCIAL, None, None, "geico", commercial_id=7)
    assert r.feed(SegmentInput("2020-01-01T00:00:00.000Z", k1, brand_name=None)) == []
    out = r.feed(SegmentInput("2020-01-01T00:00:30.000Z", k2, brand_name="Geico"))
    assert len(out) == 1
    assert out[0].category is BroadcastCategory.DJ
    assert out[0].timestamp_end == "2020-01-01T00:00:30.000Z"
    fin = r.finalize("2020-01-01T00:00:30.000Z", 10.0)
    assert len(fin) == 1
    assert fin[0].category is BroadcastCategory.COMMERCIAL
    assert fin[0].brand_name == "Geico"
    assert fin[0].commercial_id == 7


def test_different_song_ids_dont_merge() -> None:
    r = SegmentReducer()
    k_song_a = SegmentKey(BroadcastCategory.SONG, "artist", "title", None, song_id=1)
    k_song_b = SegmentKey(BroadcastCategory.SONG, "artist", "title", None, song_id=2)
    assert r.feed(SegmentInput("2020-01-01T00:00:00.000Z", k_song_a)) == []
    out = r.feed(SegmentInput("2020-01-01T00:00:10.000Z", k_song_b))
    assert len(out) == 1, "different song_id should close the previous segment"


def test_finalize_empty_when_never_fed() -> None:
    r = SegmentReducer()
    assert r.finalize("2020-01-01T00:00:00.000Z", 10.0) == []


def test_segment_input_for_song_normalizes_keys() -> None:
    si = segment_input_for_song(
        window_start_utc="2020-01-01T00:00:00.000Z",
        artist=" Taylor Swift ",
        title="ANTI-HERO",
        song_id=11,
    )
    assert si.key.category is BroadcastCategory.SONG
    assert si.key.artist_key == "taylor swift"
    assert si.key.title_key == "anti-hero"
    assert si.key.song_id == 11
    assert si.artist == " Taylor Swift "  # display field preserves original


def test_segment_input_for_unknown_song_has_null_keys() -> None:
    si = segment_input_for_unknown_song(window_start_utc="2020-01-01T00:00:00.000Z")
    assert si.key.category is BroadcastCategory.SONG
    assert si.key.artist_key is None
    assert si.key.title_key is None
    assert si.key.song_id is None


def test_segment_input_for_speech_rejects_song() -> None:
    with pytest.raises(ValueError):
        segment_input_for_speech(
            window_start_utc="2020-01-01T00:00:00.000Z",
            category=BroadcastCategory.SONG,
            brand=None,
            commercial_id=None,
            transcript_excerpt=None,
        )


def test_segment_input_for_speech_carries_commercial_id() -> None:
    si = segment_input_for_speech(
        window_start_utc="2020-01-01T00:00:00.000Z",
        category=BroadcastCategory.COMMERCIAL,
        brand="Toyota",
        commercial_id=99,
        transcript_excerpt="Buy a Toyota today",
        confidence=0.91,
    )
    assert si.key.category is BroadcastCategory.COMMERCIAL
    assert si.key.brand_key == "toyota"
    assert si.key.commercial_id == 99
    assert si.brand_name == "Toyota"
    assert si.transcript_excerpt == "Buy a Toyota today"
    assert si.confidence == 0.91
