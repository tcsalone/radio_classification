"""CLI smoke tests — argparse + db init + report happy path, no network/GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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
        "classify",
        "ingest",
        "report",
        "seed",
        "songs",
    ):
        assert sub in out, f"missing subcommand {sub} in --help"


def test_db_init_creates_v2_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "rc.db"
    proc = _run("db", "init", "--db-path", str(db_path))
    assert proc.returncode == 0, proc.stderr
    assert db_path.exists()
    # Schema v2 is in place.
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM schema_meta WHERE key='version'"
        ).fetchone()
        assert row == ("2",)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"broadcast_events", "brands", "songs", "commercials", "brand_mentions"} <= tables


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
