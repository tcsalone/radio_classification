"""CLI smoke tests — argparse + db init + report happy path, no network/GPU."""

from __future__ import annotations

import os
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "radio_classifier", *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def test_help_lists_subcommands() -> None:
    proc = _run("--help")
    assert proc.returncode == 0
    out = proc.stdout
    for sub in (
        "prereq-check",
        "db",
        "fingerprint",
        "capture",
        "runs",
        "classify",
        "ingest",
        "report",
        "seed",
        "songs",
        "commercials",
    ):
        assert sub in out, f"missing subcommand {sub} in --help"


def test_fingerprint_index_help_documents_rebuild_default() -> None:
    proc = _run("fingerprint", "index", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "REBUILT" in proc.stdout
    assert "--extend" in proc.stdout


def test_classify_help_documents_audfprint_min_count() -> None:
    proc = _run("classify", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--audfprint-min-count" in proc.stdout
    assert "--capture-run-id" in proc.stdout
    assert "default: 30" in proc.stdout


def test_fingerprint_explain_help_lists_options() -> None:
    proc = _run("fingerprint", "explain", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--input" in proc.stdout
    assert "--expected" in proc.stdout
    assert "--max-matches" in proc.stdout
    assert "--min-count" in proc.stdout


def test_fingerprint_subcommand_help_lists_explain() -> None:
    proc = _run("fingerprint", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "explain" in proc.stdout
    assert "candidate score" in proc.stdout


def test_capture_chunks_help_lists_chunk_options() -> None:
    proc = _run("capture", "chunks", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--chunk-seconds" in proc.stdout
    assert "--out-dir" in proc.stdout


def test_fingerprint_index_rebuilds_existing_index_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without --extend, an existing index file is removed before audfprint runs.

    Regression for the 2026-05-29 indexing failure: the previous behaviour
    auto-extended any existing index, which caused ``audfprint add`` to choke
    on the full corpus. The new default is to rebuild, so the second invocation
    must delete and recreate the index file.

    Invoked in-process (not via subprocess) so monkeypatch can stub audfprint.
    """
    from radio_classifier.cli import main as cli_main

    songs_dir = tmp_path / "songs"
    songs_dir.mkdir()
    (songs_dir / "ref.txt").write_text("fake reference", encoding="utf-8")
    index_path = tmp_path / "songs.pklz"

    calls: list[tuple[Path, bool]] = []

    def fake_build_or_extend(self, audio_files, *, extend=False):  # type: ignore[no-untyped-def]
        calls.append((self.index_path, extend))
        self.index_path.write_text("mock-index", encoding="utf-8")

    monkeypatch.setattr(
        "radio_classifier.fingerprint.audfprint_engine.AudfprintIndex.build_or_extend",
        fake_build_or_extend,
    )

    rc = cli_main(["fingerprint", "index", "--dir", str(songs_dir), "--out", str(index_path)])
    assert rc == 0
    assert index_path.exists()
    assert calls[-1][1] is False  # default: rebuild
    err = capsys.readouterr().err
    assert "rebuilt" in err.lower()

    # Rerun without --extend: the existing index file must be removed first
    # so audfprint sees a clean slate.
    rc = cli_main(["fingerprint", "index", "--dir", str(songs_dir), "--out", str(index_path)])
    assert rc == 0
    assert calls[-1][1] is False
    assert "rebuilt" in capsys.readouterr().err.lower()

    # With --extend, the existing index must be left alone and extend=True passed.
    rc = cli_main(
        ["fingerprint", "index", "--dir", str(songs_dir), "--out", str(index_path), "--extend"]
    )
    assert rc == 0
    assert calls[-1][1] is True
    err = capsys.readouterr().err
    assert "extended" in err.lower()


def test_fingerprint_index_extend_without_existing_index_falls_back_to_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radio_classifier.cli import main as cli_main

    songs_dir = tmp_path / "songs"
    songs_dir.mkdir()
    (songs_dir / "ref.txt").write_text("fake reference", encoding="utf-8")
    index_path = tmp_path / "songs.pklz"

    calls: list[bool] = []

    def fake_build_or_extend(self, audio_files, *, extend=False):  # type: ignore[no-untyped-def]
        calls.append(extend)
        self.index_path.write_text("mock-index", encoding="utf-8")

    monkeypatch.setattr(
        "radio_classifier.fingerprint.audfprint_engine.AudfprintIndex.build_or_extend",
        fake_build_or_extend,
    )

    rc = cli_main(
        ["fingerprint", "index", "--dir", str(songs_dir), "--out", str(index_path), "--extend"]
    )
    assert rc == 0
    assert calls == [False], "no existing index: --extend should silently degrade to rebuild"
    err = capsys.readouterr().err
    assert "creating fresh index" in err.lower()


def test_report_rejects_empty_db_path_with_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--db-path ""` (common foot-gun from an unset ``$DB`` shell var) must
    produce a one-line stderr message and a non-zero exit code, NOT a Python
    traceback to sqlite3.OperationalError.

    Invoked in-process so we can inspect the exact stderr.
    """
    from radio_classifier.cli import main as cli_main

    rc = cli_main(["report", "songs", "--db-path", "", "--since", "1d"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--db-path is empty" in err
    assert "Traceback" not in err


def test_db_init_rejects_empty_db_path_with_clean_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radio_classifier.cli import main as cli_main

    rc = cli_main(["db", "init", "--db-path", ""])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--db-path is empty" in err
    assert "Traceback" not in err


def test_songs_discovered_rejects_empty_db_path_with_clean_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from radio_classifier.cli import main as cli_main

    rc = cli_main(["songs", "discovered", "--db-path", "", "--since", "1d"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--db-path is empty" in err
    assert "Traceback" not in err


def test_report_help_lists_artists_subcommand() -> None:
    """``report --help`` must advertise the new ``artists`` subcommand."""
    proc = _run("report", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "artists" in proc.stdout
    assert "songs-timeline" in proc.stdout
    assert "dashboard" in proc.stdout


def test_commercials_help_lists_dedupe_subcommand() -> None:
    proc = _run("commercials", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "dedupe" in proc.stdout


def test_report_artists_runs_against_v2_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end happy path: init a v2 DB, write a SONG event, run
    ``report artists`` in-process, and verify the table headers + a row
    for the artist make it into stdout."""
    from datetime import datetime, timedelta, timezone

    from radio_classifier.cli import main as cli_main
    from radio_classifier.persistence import BroadcastStore
    from radio_classifier.segments.types import BroadcastCategory, SegmentTransition

    db_path = tmp_path / "rc.db"
    rc = cli_main(["db", "init", "--db-path", str(db_path)])
    assert rc == 0

    base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)

    def _iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    with BroadcastStore(db_path) as store:
        song = store.upsert_song(artist="Linkin Park", title="Numb")
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=180)),
                category=BroadcastCategory.SONG,
                artist="Linkin Park",
                track_title="Numb",
                song_id=song,
            )
        )

    proc = _run("report", "artists", "--db-path", str(db_path), "--since", "1d")
    assert proc.returncode == 0, proc.stderr
    assert "artist" in proc.stdout
    assert "spins" in proc.stdout
    assert "titles" in proc.stdout
    assert "Linkin Park" in proc.stdout


def test_report_songs_timeline_runs_against_v2_db(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from radio_classifier.cli import main as cli_main
    from radio_classifier.persistence import BroadcastStore
    from radio_classifier.segments.types import BroadcastCategory, SegmentTransition

    db_path = tmp_path / "rc.db"
    rc = cli_main(["db", "init", "--db-path", str(db_path)])
    assert rc == 0

    base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)

    def _iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    with BroadcastStore(db_path) as store:
        song = store.upsert_song(artist="Nirvana", title="Lithium", source="shazam")
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=180)),
                category=BroadcastCategory.SONG,
                artist="Nirvana",
                track_title="Lithium",
                song_id=song,
                confidence=0.91,
            )
        )

    proc = _run("report", "songs-timeline", "--db-path", str(db_path), "--since", "1d")
    assert proc.returncode == 0, proc.stderr
    assert "start_utc" in proc.stdout
    assert "source" in proc.stdout
    assert "Nirvana" in proc.stdout
    assert "Lithium" in proc.stdout
    assert "shazam" in proc.stdout


def test_classify_uses_capture_start_utc_for_delayed_chunks(tmp_path: Path) -> None:
    from radio_classifier.ingest.wav import write_mono_s16le_wav

    wav_path = tmp_path / "chunk.wav"
    # Two 1-second windows at 4 Hz. Disable all tiers so no models load.
    write_mono_s16le_wav(wav_path, np.zeros(8, dtype=np.int16), 4)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "radio_classifier",
            "classify",
            "-i",
            str(wav_path),
            "--sample-rate-override",
            "4",
            "--window-seconds",
            "1",
            "--overlap-fraction",
            "0",
            "--capture-start-utc",
            "2020-01-01T00:00:00.000Z",
            "--no-tier1",
            "--no-tier2",
            "--no-tier3",
            "--json-lines",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"window_start_utc":"2020-01-01T00:00:00.000Z"' in proc.stdout
    assert '"window_start_utc":"2020-01-01T00:00:01.000Z"' in proc.stdout


def test_capture_chunks_writes_contiguous_wavs_and_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from radio_classifier.cli import main as cli_main

    class FakeRtlFmStream:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def start(self) -> None:
            return None

        def iter_stdout_bytes(self, max_wall_seconds=None):  # type: ignore[no-untyped-def]
            # sample_rate=4, chunk_seconds=1 => 4 samples/chunk => 8 bytes.
            # Emit exactly two complete chunks.
            yield (b"\x01\x00" * 8)

    monkeypatch.setattr("radio_classifier.cli.RtlFmStream", FakeRtlFmStream)

    out_dir = tmp_path / "chunks"
    rc = cli_main(
        [
            "capture",
            "chunks",
            "--out-dir",
            str(out_dir),
            "--run-id",
            "test_run",
            "--sample-rate",
            "4",
            "--chunk-seconds",
            "1",
            "--duration-limit",
            "2",
        ]
    )
    assert rc == 0

    wavs = sorted(out_dir.glob("*.wav"))
    sidecars = sorted(out_dir.glob("*.json"))
    assert [p.name for p in wavs] == ["test_run_block0001.wav", "test_run_block0002.wav"]
    assert [p.name for p in sidecars] == ["test_run_block0001.json", "test_run_block0002.json"]

    first = sidecars[0].read_text(encoding="utf-8")
    second = sidecars[1].read_text(encoding="utf-8")
    assert '"complete": true' in first
    assert '"complete": true' in second
    assert '"duration_seconds": 1.0' in first

    with wave.open(str(wavs[0]), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 4
        assert wf.getnframes() == 4


def test_report_dashboard_writes_html(tmp_path: Path) -> None:
    from datetime import datetime, timedelta, timezone

    from radio_classifier.cli import main as cli_main
    from radio_classifier.persistence import BroadcastStore
    from radio_classifier.segments.types import BroadcastCategory, SegmentTransition

    db_path = tmp_path / "rc.db"
    out_path = tmp_path / "dashboard.html"
    rc = cli_main(["db", "init", "--db-path", str(db_path)])
    assert rc == 0

    base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)

    def _iso(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    with BroadcastStore(db_path) as store:
        song = store.upsert_song(artist="Green Day", title="Holiday")
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=180)),
                category=BroadcastCategory.SONG,
                artist="Green Day",
                track_title="Holiday",
                song_id=song,
            )
        )

    proc = _run(
        "report",
        "dashboard",
        "--db-path",
        str(db_path),
        "--since",
        "1d",
        "--out",
        str(out_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert "Broadcast Metrics Dashboard" in body
    assert "Green Day" in body


def test_classify_help_lists_progress_flags() -> None:
    proc = _run("classify", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--progress" in proc.stdout
    assert "--no-progress" in proc.stdout
    assert "--no-batch-tier1" in proc.stdout
    assert "--whisper-beam-size" in proc.stdout
    assert "--whisper-vad-filter" in proc.stdout


def test_db_init_creates_v4_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "rc.db"
    proc = _run("db", "init", "--db-path", str(db_path))
    assert proc.returncode == 0, proc.stderr
    assert db_path.exists()
    # Schema v4 is in place.
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()
        assert row == ("4",)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {
            "broadcast_events",
            "brands",
            "songs",
            "commercials",
            "brand_mentions",
            "capture_runs",
        } <= tables
        song_columns = {
            r[1]
            for r in conn.execute("PRAGMA table_info(songs)")
        }
        assert "release_date" in song_columns


def test_default_db_path_is_persistent_store() -> None:
    from radio_classifier.cli import _default_db_path

    path = _default_db_path()
    assert path.name == "broadcast.db"
    assert path.parent.name == "store"


def test_runs_start_end_list(tmp_path: Path) -> None:
    db_path = tmp_path / "broadcast.db"
    init = _run("db", "init", "--db-path", str(db_path))
    assert init.returncode == 0, init.stderr

    start = _run(
        "runs",
        "start",
        "--db-path",
        str(db_path),
        "--run-id",
        "continuous_test",
        "--started-utc",
        "2026-06-01T00:00:00.000Z",
        "--pipeline-version",
        "0.3.0+test",
        "--host",
        "pytest",
    )
    assert start.returncode == 0, start.stderr
    capture_run_id = int(start.stdout.strip())
    assert capture_run_id > 0

    end = _run(
        "runs",
        "end",
        "--db-path",
        str(db_path),
        "--run-id",
        "continuous_test",
        "--ended-utc",
        "2026-06-01T00:30:00.000Z",
        "--notes",
        "done",
    )
    assert end.returncode == 0, end.stderr

    listed = _run("runs", "list", "--db-path", str(db_path))
    assert listed.returncode == 0, listed.stderr
    assert "continuous_test" in listed.stdout
    assert "0.3.0+test" in listed.stdout


def test_report_songs_added_filters_explicit_window(tmp_path: Path) -> None:
    from radio_classifier.persistence import BroadcastStore

    db_path = tmp_path / "broadcast.db"
    with BroadcastStore(db_path, use_wal=False) as store:
        old_id = store.upsert_song(artist="Old Artist", title="Old Song", source="manual")
        new_id = store.upsert_song(artist="New Artist", title="New Song", source="shazam")
        store.connection.execute(
            "UPDATE songs SET first_seen_utc = ? WHERE id = ?",
            ("2026-05-01T00:00:00.000Z", old_id),
        )
        store.connection.execute(
            "UPDATE songs SET first_seen_utc = ? WHERE id = ?",
            ("2026-06-01T12:00:00.000Z", new_id),
        )
        store.connection.commit()

    proc = _run(
        "report",
        "songs-added",
        "--db-path",
        str(db_path),
        "--from",
        "2026-06-01T00:00:00.000Z",
        "--to",
        "2026-06-02T00:00:00.000Z",
    )
    assert proc.returncode == 0, proc.stderr
    assert "New Artist" in proc.stdout
    assert "Old Artist" not in proc.stdout


def test_report_runs_lists_capture_runs(tmp_path: Path) -> None:
    db_path = tmp_path / "broadcast.db"
    init = _run("db", "init", "--db-path", str(db_path))
    assert init.returncode == 0, init.stderr
    start = _run(
        "runs",
        "start",
        "--db-path",
        str(db_path),
        "--run-id",
        "continuous_report_test",
        "--started-utc",
        "2026-06-01T12:00:00.000Z",
        "--pipeline-version",
        "0.3.0+test",
    )
    assert start.returncode == 0, start.stderr

    proc = _run(
        "report",
        "runs",
        "--db-path",
        str(db_path),
        "--from",
        "2026-06-01T00:00:00.000Z",
        "--to",
        "2026-06-02T00:00:00.000Z",
    )
    assert proc.returncode == 0, proc.stderr
    assert "continuous_report_test" in proc.stdout
    assert "0.3.0+test" in proc.stdout


def test_report_rejects_since_with_from(tmp_path: Path) -> None:
    db_path = tmp_path / "broadcast.db"
    init = _run("db", "init", "--db-path", str(db_path))
    assert init.returncode == 0, init.stderr
    proc = _run(
        "report",
        "summary",
        "--db-path",
        str(db_path),
        "--since",
        "1d",
        "--from",
        "2026-06-01T00:00:00.000Z",
    )
    assert proc.returncode == 2
    assert "either --since or --from/--to" in proc.stderr


def test_report_summary_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "rc.db"
    init = _run("db", "init", "--db-path", str(db_path))
    assert init.returncode == 0, init.stderr
    rep = _run("report", "summary", "--db-path", str(db_path), "--since", "1h")
    assert rep.returncode == 0, rep.stderr
    assert "(no rows)" in rep.stdout


def test_songs_discovered_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "rc.db"
    init = _run("db", "init", "--db-path", str(db_path))
    assert init.returncode == 0, init.stderr
    proc = _run("songs", "discovered", "--db-path", str(db_path), "--since", "1h")
    assert proc.returncode == 0, proc.stderr
    assert "(no rows)" in proc.stdout
