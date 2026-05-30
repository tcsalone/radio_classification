"""Ground-truth fixture checks for manually validated captures."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "eval" / "morning_20260527_truth.json"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_morning_ground_truth_fixture_schema() -> None:
    data = _load_fixture()

    assert data["name"] == "morning_20260527_drive"
    assert data["capture_path"].endswith("20260527T131242Z.wav")
    assert isinstance(data["checks"], list)
    assert len(data["checks"]) >= 10

    ids = set()
    for check in data["checks"]:
        ids.add(check["id"])
        assert isinstance(check["offset_seconds"], int)
        assert check["offset_seconds"] >= 0
        assert check["truth"]["category"] in {
            "SONG",
            "DJ",
            "COMMERCIAL",
            "STATION",
            "PSA_NEWS",
        }
        if check["truth"]["category"] == "SONG":
            assert check["truth"].get("artist")
            assert check["truth"].get("title")
        if check["truth"]["category"] == "COMMERCIAL":
            assert check["truth"].get("brand")
        assert isinstance(check.get("assert_current", False), bool)

    assert len(ids) == len(data["checks"])
    assert any(not c.get("assert_current", False) for c in data["checks"])


def _event_at(conn: sqlite3.Connection, *, offset_seconds: int) -> sqlite3.Row | None:
    base_s = conn.execute("SELECT MIN(timestamp_start) FROM broadcast_events").fetchone()[0]
    if base_s is None:
        return None
    base = datetime.fromisoformat(str(base_s).replace("Z", "+00:00"))
    ts = (
        base + timedelta(seconds=offset_seconds)
    ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return conn.execute(
        """
        SELECT category, artist, track_title, brand_name, transcript_excerpt
        FROM broadcast_events
        WHERE timestamp_start <= ?
          AND COALESCE(timestamp_end, timestamp_start) > ?
        ORDER BY timestamp_start DESC
        LIMIT 1
        """,
        (ts, ts),
    ).fetchone()


def test_morning_eval_db_matches_asserted_ground_truth() -> None:
    data = _load_fixture()
    db_path = Path(data["local_eval_db"])
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    if not db_path.exists():
        pytest.skip(f"local eval DB not present: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for check in data["checks"]:
            if not check.get("assert_current", False):
                continue
            row = _event_at(conn, offset_seconds=check["offset_seconds"])
            assert row is not None, check["id"]
            truth = check["truth"]
            assert row["category"] == truth["category"], check["id"]
            if truth["category"] == "SONG":
                assert row["artist"] == truth["artist"], check["id"]
                assert truth["title"] in row["track_title"], check["id"]
            if truth["category"] == "COMMERCIAL":
                assert row["brand_name"] == truth["brand"], check["id"]
    finally:
        conn.close()
