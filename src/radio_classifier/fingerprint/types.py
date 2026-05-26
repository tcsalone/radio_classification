"""Types for the Tier-1 audio fingerprint engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FingerprintStatus(str, Enum):
    """Outcome of an audfprint match attempt."""

    match = "match"
    no_match = "no_match"
    error = "error"
    skipped = "skipped"   # e.g. index missing, engine disabled


@dataclass
class FingerprintResult:
    """One Tier-1 attempt against the seeded song index."""

    status: FingerprintStatus
    window_start_utc: str
    track_id: str | None = None
    artist: str | None = None
    title: str | None = None
    match_score: float | None = None
    message: str | None = None
