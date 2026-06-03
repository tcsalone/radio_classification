"""Song age stats and MusicBrainz date parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from radio_classifier.discovery.releases import (
    _best_release_date_from_recording,
    _parse_mb_date,
    normalize_title_for_lookup,
    song_age_years,
)
from radio_classifier.persistence import BroadcastStore
from radio_classifier.reports.queries import song_age_stats
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def test_normalize_title_for_lookup_strips_feature_suffix() -> None:
    assert normalize_title_for_lookup("Go Away (feat. Best Coast)") == "Go Away"


def test_parse_mb_date_accepts_year_only() -> None:
    assert _parse_mb_date("1995") == _parse_mb_date("1995-01-01")


def test_best_release_date_picks_earliest() -> None:
    recording = {
        "first-release-date": "1996",
        "releases": [{"date": "1995-10-02"}],
    }
    assert _best_release_date_from_recording(recording) == "1995-10-02"


def test_song_age_years() -> None:
    ref = datetime(2026, 6, 2, tzinfo=timezone.utc)
    age = song_age_years("1995-10-02", ref)
    assert 30.0 < age < 31.0


def test_song_age_stats_mean_and_median(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "age.db")
    try:
        old = store.upsert_song(artist="Oasis", title="Wonderwall", source="audfprint")
        new = store.upsert_song(artist="Dexter", title="Freakin Out", source="shazam")
        store.set_song_release_date(old, "1995-10-02")
        store.set_song_release_date(new, "2024-01-01")
        store.connection.commit()

        for song_id, start, end in (
            (old, "2026-06-02T10:00:00.000Z", "2026-06-02T10:05:00.000Z"),
            (new, "2026-06-02T10:10:00.000Z", "2026-06-02T10:15:00.000Z"),
        ):
            store.apply_transition(
                SegmentTransition(
                    timestamp_start=start,
                    timestamp_end=end,
                    category=BroadcastCategory.SONG,
                    song_id=song_id,
                    artist="x",
                    track_title="y",
                )
            )

        stats = song_age_stats(
            store,
            since_utc="2026-06-02T00:00:00.000Z",
            until_utc="2026-06-03T00:00:00.000Z",
            reference_utc="2026-06-02T12:00:00.000Z",
        )
        assert stats.songs_with_dates == 2
        assert stats.songs_missing_dates == 0
        assert stats.distinct_song_mean_years is not None
        assert stats.distinct_song_median_years is not None
        assert stats.airtime_weighted_mean_years is not None
        assert stats.distinct_song_mean_years > stats.airtime_weighted_mean_years * 0.5
    finally:
        store.close()
