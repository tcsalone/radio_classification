"""Platform helpers for cross-platform operator scripts and prereq checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_macos() -> bool:
    return sys.platform == "darwin"


def default_ollama_host() -> str:
    """Return the Ollama base URL from env or the Python default (:11434)."""
    return os.environ.get("RADIO_CLASSIFIER_OLLAMA_HOST", "http://127.0.0.1:11434")


def default_whisper_device() -> str:
    """CPU on macOS; cuda elsewhere (Linux backward compat)."""
    if is_macos():
        return "cpu"
    return "cuda"


def file_size_bytes(path: str | Path) -> int:
    """Portable file size in bytes (GNU stat vs BSD stat)."""
    p = Path(path)
    if not p.is_file():
        return 0
    if is_macos():
        return p.stat().st_size
    return p.stat().st_size
