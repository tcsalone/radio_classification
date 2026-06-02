"""Tests for completed-WAV retention pruning."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prune_old_wavs.sh"


def _write_chunk(path: Path, *, complete: bool, age_days: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"wav")
    path.with_suffix(".json").write_text(
        json.dumps({"complete": complete, "wav_path": str(path)}),
        encoding="utf-8",
    )
    mtime = time.time() - age_days * 24 * 60 * 60
    os.utime(path, (mtime, mtime))


def test_prune_old_wavs_deletes_only_old_complete_chunks(tmp_path: Path) -> None:
    old_complete = tmp_path / "run1" / "old_complete.wav"
    old_partial = tmp_path / "run1" / "old_partial.wav"
    recent_complete = tmp_path / "run1" / "recent_complete.wav"
    orphan = tmp_path / "run1" / "orphan.wav"

    _write_chunk(old_complete, complete=True, age_days=10)
    _write_chunk(old_partial, complete=False, age_days=10)
    _write_chunk(recent_complete, complete=True, age_days=1)
    orphan.write_bytes(b"wav")
    old_mtime = time.time() - 10 * 24 * 60 * 60
    os.utime(orphan, (old_mtime, old_mtime))

    proc = subprocess.run(
        ["bash", str(SCRIPT), str(tmp_path)],
        env={**os.environ, "RETENTION_DAYS": "7"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "deleted=1" in proc.stdout
    assert not old_complete.exists()
    assert old_complete.with_suffix(".json").exists(), "sidecar should be retained"
    assert old_partial.exists()
    assert recent_complete.exists()
    assert orphan.exists()
