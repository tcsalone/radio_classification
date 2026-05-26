"""Types for the Tier-2 acoustic classifier."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AcousticLabel(str, Enum):
    """3-way roll-up of YAMNet / PANNs output for routing.

    The fine classes (521 AudioSet labels) are aggregated into these three
    buckets at the classifier layer. Callers only see this enum.
    """

    MUSIC = "MUSIC"
    SPEECH = "SPEECH"
    OTHER = "OTHER"


@dataclass
class AcousticResult:
    """Result of running the acoustic classifier over one analysis window."""

    label: AcousticLabel
    window_start_utc: str
    music_prob: float
    speech_prob: float
    other_prob: float
    top_classes: list[tuple[str, float]]  # (class_name, prob) sorted desc, ≤ 5

    @property
    def max_prob(self) -> float:
        return max(self.music_prob, self.speech_prob, self.other_prob)
