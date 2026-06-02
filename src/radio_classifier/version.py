"""Version helpers for persisted pipeline provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

from radio_classifier import __version__


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_short_sha(repo_root: Path | None = None) -> str | None:
    """Return the current git commit short SHA if this checkout has one."""

    root = repo_root or _repo_root()
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def pipeline_version(repo_root: Path | None = None) -> str:
    """Return a stable pipeline-version string for capture run provenance."""

    return f"{__version__}+{git_short_sha(repo_root) or 'nogit'}"
