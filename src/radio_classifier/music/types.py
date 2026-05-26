"""Types for the optional Shazam fallback (``--enable-shazam``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ShazamStatus(str, Enum):
    """Outcome of a Shazam identification attempt."""

    match = "match"
    no_match = "no_match"
    low_confidence = "low_confidence"
    error = "error"
    skipped = "skipped"


@dataclass
class ShazamResult:
    """Structured result for one window."""

    status: ShazamStatus
    window_start_utc: str
    artist: str | None = None
    title: str | None = None
    confidence: float | None = None
    message: str | None = None
    raw: dict[str, Any] | None = field(default=None)
