"""Types for speech transcription and 5-class LLM classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from radio_classifier.segments.types import BroadcastCategory


class TranscribeStatus(str, Enum):
    """Outcome of faster-whisper transcription."""

    ok = "ok"
    error = "error"


@dataclass
class SpeechTranscriptResult:
    """Structured transcript for one analysis window."""

    status: TranscribeStatus
    window_start_utc: str
    text: str
    message: str | None = None


class SpeechPipelineStatus(str, Enum):
    """Combined transcribe + classify outcome."""

    ok = "ok"
    error = "error"
    skipped = "skipped"


@dataclass
class BrandMention:
    """One brand mention extracted by the LLM."""

    name: str
    mention_type: str  # 'paid_ad' | 'dj_shoutout' | 'tag'


@dataclass
class CommercialSignature:
    """LLM-emitted signal used by the commercial identity resolver."""

    key_phrases: list[str]
    duration_bucket_seconds: int


@dataclass
class SpeechPipelineResult:
    """Transcription + LLM 5-class classification for one window."""

    window_start_utc: str
    status: SpeechPipelineStatus
    transcript: str
    category: BroadcastCategory | None = None
    brand: str | None = None
    brand_mentions: list[BrandMention] = field(default_factory=list)
    commercial_signature: CommercialSignature | None = None
    confidence: float | None = None
    rationale: str | None = None
    message: str | None = None
