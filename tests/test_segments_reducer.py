"""Reducer + normalize tests for the 5-class taxonomy."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

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


def _ts(seconds: int) -> str:
    dt = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _song(seconds: int, *, song_id: int = 42, artist: str = "Incubus", title: str = "Wish You Were Here") -> SegmentInput:
    return SegmentInput(
        _ts(seconds),
        SegmentKey(
            BroadcastCategory.SONG,
            artist.lower(),
            title.lower(),
            None,
            song_id=song_id,
        ),
        artist=artist,
        track_title=title,
        confidence=0.95,
    )


def _unknown_song(seconds: int, *, confidence: float = 0.7) -> SegmentInput:
    return SegmentInput(
        _ts(seconds),
        SegmentKey(BroadcastCategory.SONG),
        confidence=confidence,
    )


def _dj(seconds: int) -> SegmentInput:
    return SegmentInput(
        _ts(seconds),
        SegmentKey(BroadcastCategory.DJ),
        transcript_excerpt="DJ talk",
        confidence=0.8,
    )


def test_unknown_song_bridge_merges_into_bracketing_song() -> None:
    r = SegmentReducer()
    out = []
    out.extend(r.feed(_song(0)))
    out.extend(r.feed(_song(10)))
    out.extend(r.feed(_unknown_song(20)))
    out.extend(r.feed(_song(30)))
    out.extend(r.feed(_song(40)))
    out.extend(r.finalize(_ts(40), 10.0))

    assert len(out) == 1
    assert out[0].category is BroadcastCategory.SONG
    assert out[0].song_id == 42
    assert out[0].timestamp_start == _ts(0)
    assert out[0].timestamp_end == _ts(50)
    assert out[0].track_title == "Wish You Were Here"


def test_unknown_song_bridge_does_not_swallow_long_gap() -> None:
    r = SegmentReducer(max_unknown_song_bridge_seconds=20.0)
    out = []
    out.extend(r.feed(_song(0)))
    out.extend(r.feed(_unknown_song(10)))
    out.extend(r.feed(_unknown_song(20)))
    out.extend(r.feed(_unknown_song(30)))
    out.extend(r.feed(_unknown_song(40)))
    out.extend(r.feed(_song(50)))
    out.extend(r.finalize(_ts(50), 10.0))

    assert [(t.category, t.song_id, t.timestamp_start, t.timestamp_end) for t in out] == [
        (BroadcastCategory.SONG, 42, _ts(0), _ts(10)),
        (BroadcastCategory.SONG, None, _ts(10), _ts(50)),
        (BroadcastCategory.SONG, 42, _ts(50), _ts(60)),
    ]


def test_unknown_followed_by_different_song_still_emits_unknown() -> None:
    r = SegmentReducer()
    out = []
    out.extend(r.feed(_song(0, song_id=1, artist="A", title="One")))
    out.extend(r.feed(_unknown_song(10)))
    out.extend(r.feed(_song(20, song_id=2, artist="B", title="Two")))
    out.extend(r.finalize(_ts(20), 10.0))

    assert [(t.category, t.song_id, t.timestamp_start, t.timestamp_end) for t in out] == [
        (BroadcastCategory.SONG, 1, _ts(0), _ts(10)),
        (BroadcastCategory.SONG, None, _ts(10), _ts(20)),
        (BroadcastCategory.SONG, 2, _ts(20), _ts(30)),
    ]


def test_unknown_followed_by_dj_still_emits_unknown() -> None:
    r = SegmentReducer()
    out = []
    out.extend(r.feed(_song(0)))
    out.extend(r.feed(_unknown_song(10)))
    out.extend(r.feed(_dj(20)))
    out.extend(r.finalize(_ts(20), 10.0))

    assert [(t.category, t.song_id, t.timestamp_start, t.timestamp_end) for t in out] == [
        (BroadcastCategory.SONG, 42, _ts(0), _ts(10)),
        (BroadcastCategory.SONG, None, _ts(10), _ts(20)),
        (BroadcastCategory.DJ, None, _ts(20), _ts(30)),
    ]


def test_unknown_then_finalize_emits_unknown() -> None:
    r = SegmentReducer()
    out = []
    out.extend(r.feed(_song(0)))
    out.extend(r.feed(_unknown_song(10)))
    out.extend(r.finalize(_ts(10), 10.0))

    assert [(t.category, t.song_id, t.timestamp_start, t.timestamp_end) for t in out] == [
        (BroadcastCategory.SONG, 42, _ts(0), _ts(10)),
        (BroadcastCategory.SONG, None, _ts(10), _ts(20)),
    ]


def test_max_unknown_song_bridge_seconds_zero_disables_feature() -> None:
    r = SegmentReducer(max_unknown_song_bridge_seconds=0.0)
    out = []
    out.extend(r.feed(_song(0)))
    out.extend(r.feed(_unknown_song(10)))
    out.extend(r.feed(_song(20)))
    out.extend(r.finalize(_ts(20), 10.0))

    assert [(t.category, t.song_id, t.timestamp_start, t.timestamp_end) for t in out] == [
        (BroadcastCategory.SONG, 42, _ts(0), _ts(10)),
        (BroadcastCategory.SONG, None, _ts(10), _ts(20)),
        (BroadcastCategory.SONG, 42, _ts(20), _ts(30)),
    ]


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
