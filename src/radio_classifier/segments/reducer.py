"""Pure segment reducer — emits :class:`SegmentTransition` rows (no I/O).

Lifted from ``live105sux`` and extended to carry the 5-class enum plus
``song_id`` / ``commercial_id`` / ``brand_id`` / transcript / confidence
display fields.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from radio_classifier.segments.types import (
    SegmentInput,
    SegmentKey,
    SegmentTransition,
)


def _parse_utc(iso_z: str) -> datetime:
    return datetime.fromisoformat(iso_z.replace("Z", "+00:00"))


def _format_utc_ms(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def duration_seconds(start_iso: str, end_iso: str) -> float:
    """Length of half-open interval ``[start, end)`` in seconds."""
    a = _parse_utc(start_iso)
    b = _parse_utc(end_iso)
    return (b - a).total_seconds()


class SegmentReducer:
    """Accumulates one logical segment; emits a closed transition on key change.

    **Last window wins** for ``artist``, ``track_title``, ``brand_name``,
    ``transcript_excerpt``, and ``confidence`` while the :class:`SegmentKey`
    is unchanged.

    The reducer is pure: callers (see ``persistence.coordinator``) feed inputs
    and persist the returned closed transitions.
    """

    def __init__(self) -> None:
        self._current_key: SegmentKey | None = None
        self._current_start_utc: str | None = None
        self._artist: str | None = None
        self._title: str | None = None
        self._brand: str | None = None
        self._transcript: str | None = None
        self._confidence: float | None = None

    def feed(self, inp: SegmentInput | None) -> list[SegmentTransition]:
        if inp is None:
            return []

        if self._current_key is None:
            self._current_key = inp.key
            self._current_start_utc = inp.window_start_utc
            self._merge_display(inp)
            return []

        if inp.key == self._current_key:
            self._merge_display(inp)
            return []

        assert self._current_start_utc is not None
        closed = SegmentTransition(
            timestamp_start=self._current_start_utc,
            timestamp_end=inp.window_start_utc,
            category=self._current_key.category,
            artist=self._artist,
            track_title=self._title,
            brand_name=self._brand,
            song_id=self._current_key.song_id,
            commercial_id=self._current_key.commercial_id,
            transcript_excerpt=self._transcript,
            confidence=self._confidence,
        )
        self._current_key = inp.key
        self._current_start_utc = inp.window_start_utc
        self._artist = inp.artist
        self._title = inp.track_title
        self._brand = inp.brand_name
        self._transcript = inp.transcript_excerpt
        self._confidence = inp.confidence
        return [closed]

    def _merge_display(self, inp: SegmentInput) -> None:
        if inp.artist is not None:
            self._artist = inp.artist
        if inp.track_title is not None:
            self._title = inp.track_title
        if inp.brand_name is not None:
            self._brand = inp.brand_name
        if inp.transcript_excerpt is not None:
            self._transcript = inp.transcript_excerpt
        if inp.confidence is not None:
            self._confidence = inp.confidence

    def finalize(self, last_window_start_utc: str, window_seconds: float) -> list[SegmentTransition]:
        """Close the final open segment using ``last_window_start + window_seconds`` as end."""
        if self._current_key is None or self._current_start_utc is None:
            return []
        start_dt = _parse_utc(last_window_start_utc)
        end_dt = start_dt + timedelta(seconds=window_seconds)
        end_iso = _format_utc_ms(end_dt)
        closed = SegmentTransition(
            timestamp_start=self._current_start_utc,
            timestamp_end=end_iso,
            category=self._current_key.category,
            artist=self._artist,
            track_title=self._title,
            brand_name=self._brand,
            song_id=self._current_key.song_id,
            commercial_id=self._current_key.commercial_id,
            transcript_excerpt=self._transcript,
            confidence=self._confidence,
        )
        self._current_key = None
        self._current_start_utc = None
        self._artist = None
        self._title = None
        self._brand = None
        self._transcript = None
        self._confidence = None
        return [closed]
