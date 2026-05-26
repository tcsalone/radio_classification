"""Broadcast segment state (reducer) for schema v2."""

from radio_classifier.segments.normalize import (
    normalize_token,
    segment_input_for_song,
    segment_input_for_speech,
    segment_input_for_unknown_song,
)
from radio_classifier.segments.reducer import SegmentReducer, duration_seconds
from radio_classifier.segments.types import (
    BroadcastCategory,
    SegmentInput,
    SegmentKey,
    SegmentTransition,
)

__all__ = [
    "BroadcastCategory",
    "SegmentInput",
    "SegmentKey",
    "SegmentReducer",
    "SegmentTransition",
    "duration_seconds",
    "normalize_token",
    "segment_input_for_song",
    "segment_input_for_speech",
    "segment_input_for_unknown_song",
]
