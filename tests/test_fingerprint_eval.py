"""Tests for the fingerprint recall harness."""

from __future__ import annotations

from pathlib import Path

from radio_classifier.fingerprint.types import FingerprintResult, FingerprintStatus
from radio_classifier.seeding.eval import evaluate


class _FakeIndex:
    def match_file(self, wav_path: Path) -> FingerprintResult:
        return FingerprintResult(
            status=FingerprintStatus.match,
            window_start_utc="",
            track_id="/tmp/reference/Bad Omens - Dying To Love.mp3",
            artist="Bad Omens",
            title="Dying To Love",
            match_score=30.0,
        )


def test_evaluate_compares_parsed_artist_title_before_raw_track_path(tmp_path: Path) -> None:
    """audfprint emits path-like track ids; parser extracts artist/title from
    the basename. Eval should compare that normalized label, not the raw path.
    """
    clip = tmp_path / "clip.wav"
    truth = {clip: "Bad Omens - Dying To Love"}

    report = evaluate(_FakeIndex(), truth)

    assert report.correct == 1
    assert report.rows[0].correct is True

