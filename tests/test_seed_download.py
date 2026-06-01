"""Tests for the seed-download skip-if-exists guard."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest import mock

from radio_classifier.seeding.download import (
    DownloadConfig,
    _build_yt_dlp_cmd,
    download_track,
    download_tracks,
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
    """Default ``mp3`` format runs yt-dlp's transcode and the downloader must
    locate the produced file by stem (any audio extension on disk)."""
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


def test_download_track_picks_up_any_audio_extension(tmp_path: Path) -> None:
    """If yt-dlp falls back to a non-mp3 codec (e.g. m4a/webm), the downloader
    must still discover the produced file via ``existing_audio_for_stem``."""
    track = Track(artist="Nirvana", title="In Bloom")
    expected = tmp_path / "Nirvana - In Bloom.m4a"

    def fake_run(cmd, capture_output: bool = False, text: bool = False):
        expected.write_bytes(b"\x00")
        return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        side_effect=fake_run,
    ):
        result = download_track(track, _cfg(tmp_path))

    assert result.ok is True
    assert result.output_path == expected


def test_download_track_reports_missing_file_after_yt_dlp_success(tmp_path: Path) -> None:
    """yt-dlp can exit 0 yet leave no audio file behind (e.g. only metadata).
    We must surface that as a failure instead of silently returning ``ok=True``.
    """
    track = Track(artist="Ghost", title="Vanished")

    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
    ):
        result = download_track(track, _cfg(tmp_path))

    assert result.ok is False
    assert "no matching audio file" in result.message


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


# -------------------------------------------------------- _build_yt_dlp_cmd
def test_build_yt_dlp_cmd_default_transcodes_to_mp3(tmp_path: Path) -> None:
    """Default ``audio_format='mp3'`` must go through ffmpeg's ``-x
    --audio-format mp3 --audio-quality 192`` pipeline so all references on
    disk are encoded uniformly. Mixing mp3/m4a/opus refs in one audfprint
    index produced bloated, spurious-match fingerprints on the
    ``.webm/opus`` files in May 2026.
    """
    cfg = DownloadConfig(output_dir=tmp_path)
    cmd = _build_yt_dlp_cmd("Foo - Bar", "ytsearch1:Foo Bar audio", cfg)
    assert "-x" in cmd
    fmt_index = cmd.index("--audio-format")
    assert cmd[fmt_index + 1] == "mp3"
    q_index = cmd.index("--audio-quality")
    assert cmd[q_index + 1] == "192"


def test_build_yt_dlp_cmd_best_skips_ffmpeg_transcode(tmp_path: Path) -> None:
    """Explicit ``audio_format='best'`` switches to the no-re-encode fast
    path for operators who have validated fingerprint recall on m4a/opus.
    """
    cfg = DownloadConfig(output_dir=tmp_path, audio_format="best")
    cmd = _build_yt_dlp_cmd("Foo - Bar", "ytsearch1:Foo Bar audio", cfg)
    assert "-f" in cmd
    f_index = cmd.index("-f")
    assert cmd[f_index + 1] == "bestaudio/best"
    assert "--no-keep-video" in cmd
    assert "-x" not in cmd
    assert "--audio-format" not in cmd


# -------------------------------------------------------- download_tracks concurrency
def test_download_tracks_runs_in_parallel_and_preserves_order(tmp_path: Path) -> None:
    """``download_tracks`` must (a) actually overlap subprocess calls when
    ``concurrency > 1`` and (b) return results in input order so the caller's
    summary loop matches up with the supplied tracks.
    """
    tracks = [Track(artist=f"Artist{i}", title=f"Title{i}") for i in range(4)]
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def fake_run(cmd, capture_output: bool = False, text: bool = False):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        out_index = cmd.index("-o") + 1
        stem_template = cmd[out_index]
        stem = Path(stem_template).name.replace(".%(ext)s", "")
        (tmp_path / f"{stem}.mp3").write_bytes(b"\x00")
        with lock:
            in_flight -= 1
        return mock.Mock(returncode=0, stdout="", stderr="")

    cfg = DownloadConfig(output_dir=tmp_path, concurrency=4)
    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        side_effect=fake_run,
    ):
        results = download_tracks(tracks, cfg)

    assert [r.track.artist for r in results] == [t.artist for t in tracks]
    assert all(r.ok for r in results)
    assert max_in_flight > 1, "expected concurrent yt-dlp invocations when concurrency=4"


def test_download_tracks_serial_when_concurrency_is_one(tmp_path: Path) -> None:
    tracks = [Track(artist="A", title="One"), Track(artist="B", title="Two")]
    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def fake_run(cmd, capture_output: bool = False, text: bool = False):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        out_index = cmd.index("-o") + 1
        stem = Path(cmd[out_index]).name.replace(".%(ext)s", "")
        (tmp_path / f"{stem}.mp3").write_bytes(b"\x00")
        with lock:
            in_flight -= 1
        return mock.Mock(returncode=0, stdout="", stderr="")

    cfg = DownloadConfig(output_dir=tmp_path, concurrency=1)
    with mock.patch(
        "radio_classifier.seeding.download.subprocess.run",
        side_effect=fake_run,
    ):
        results = download_tracks(tracks, cfg)

    assert [r.ok for r in results] == [True, True]
    assert max_in_flight == 1
