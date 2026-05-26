"""Tests for the seed-download skip-if-exists guard."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from radio_classifier.seeding.download import (
    DownloadConfig,
    download_track,
    existing_audio_for_stem,
    safe_filename_stem,
)
from radio_classifier.seeding.scrape import Track


# -------------------------------------------------------- helpers
def _cfg(tmp_path: Path) -> DownloadConfig:
    return DownloadConfig(output_dir=tmp_path)


# -------------------------------------------------------- safe_filename_stem
def test_safe_filename_stem_keeps_canonical_artist_dash_title() -> None:
    stem = safe_filename_stem(Track(artist="Red Hot Chili Peppers", title="Otherside"))
    assert stem == "Red Hot Chili Peppers - Otherside"


def test_safe_filename_stem_strips_disallowed_chars() -> None:
    stem = safe_filename_stem(Track(artist="AC/DC", title="T.N.T."))
    assert "/" not in stem
    assert stem.startswith("AC")


# -------------------------------------------------------- existing_audio_for_stem
def test_existing_audio_finds_mp3(tmp_path: Path) -> None:
    target = tmp_path / "Coldplay - Clocks.mp3"
    target.write_bytes(b"\x00")
    found = existing_audio_for_stem(tmp_path, "Coldplay - Clocks")
    assert found == target


def test_existing_audio_finds_fallback_webm(tmp_path: Path) -> None:
    target = tmp_path / "The Cranberries - Zombie.webm"
    target.write_bytes(b"\x00")
    found = existing_audio_for_stem(tmp_path, "The Cranberries - Zombie")
    assert found == target


def test_existing_audio_ignores_non_audio_extensions(tmp_path: Path) -> None:
    (tmp_path / "Coldplay - Clocks.txt").write_text("notes")
    assert existing_audio_for_stem(tmp_path, "Coldplay - Clocks") is None


def test_existing_audio_returns_none_when_dir_missing(tmp_path: Path) -> None:
    assert existing_audio_for_stem(tmp_path / "nope", "Anything") is None


# -------------------------------------------------------- download_track skip path
def test_download_track_skips_when_mp3_already_on_disk(tmp_path: Path) -> None:
    track = Track(artist="Coldplay", title="Clocks")
    existing = tmp_path / "Coldplay - Clocks.mp3"
    existing.write_bytes(b"\x00")

    with mock.patch("radio_classifier.seeding.download.subprocess.run") as run:
        result = download_track(track, _cfg(tmp_path))

    assert result.ok is True
    assert result.skipped is True
    assert result.output_path == existing
    assert "already on disk" in result.message
    run.assert_not_called()


def test_download_track_skips_when_only_webm_on_disk(tmp_path: Path) -> None:
    """The downloader prefers mp3, but if a prior run produced a webm fallback
    we should treat that as 'we already have audio' and not re-fetch."""
    track = Track(artist="The Cranberries", title="Zombie")
    existing = tmp_path / "The Cranberries - Zombie.webm"
    existing.write_bytes(b"\x00")

    with mock.patch("radio_classifier.seeding.download.subprocess.run") as run:
        result = download_track(track, _cfg(tmp_path))

    assert result.skipped is True
    assert result.output_path == existing
    run.assert_not_called()


def test_download_track_invokes_yt_dlp_when_no_existing_file(tmp_path: Path) -> None:
    track = Track(artist="Nirvana", title="In Bloom")
    expected = tmp_path / "Nirvana - In Bloom.mp3"

    def fake_run(cmd, capture_output: bool = False, text: bool = False):
        expected.write_bytes(b"\x00")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        side_effect=fake_run,
    ) as run:
        result = download_track(track, _cfg(tmp_path))

    assert result.ok is True
    assert result.skipped is False
    assert result.output_path == expected
    run.assert_called_once()


def test_download_track_reports_yt_dlp_failure(tmp_path: Path) -> None:
    track = Track(artist="Edgehill", title="Doubletake")

    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        return_value=mock.Mock(returncode=1, stdout="", stderr="ERROR: no match"),
    ):
        result = download_track(track, _cfg(tmp_path))

    assert result.ok is False
    assert result.skipped is False
    assert "ERROR" in result.message
