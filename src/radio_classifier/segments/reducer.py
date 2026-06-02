"""Pure segment reducer — emits :class:`SegmentTransition` rows (no I/O).

Lifted from ``live105sux`` and extended to carry the 5-class enum plus
``song_id`` / ``commercial_id`` / ``brand_id`` / transcript / confidence
display fields.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from radio_classifier.segments.types import (
    BroadcastCategory,
    SegmentInput,
    SegmentKey,
    SegmentTransition,
)
from radio_classifier.text import text_similarity


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

    def __init__(
        self,
        *,
        max_unknown_song_bridge_seconds: float = 60.0,
        adjacent_commercial_similarity_threshold: float = 0.35,
        max_adjacent_commercial_combined_seconds: float = 90.0,
    ) -> None:
        self.max_unknown_song_bridge_seconds = max(0.0, max_unknown_song_bridge_seconds)
        self.adjacent_commercial_similarity_threshold = max(
            0.0,
            adjacent_commercial_similarity_threshold,
        )
        self.max_adjacent_commercial_combined_seconds = max(
            0.0,
            max_adjacent_commercial_combined_seconds,
        )
        self._current_key: SegmentKey | None = None
        self._current_start_utc: str | None = None
        self._artist: str | None = None
        self._title: str | None = None
        self._brand: str | None = None
        self._transcript: str | None = None
        self._confidence: float | None = None
        self._bridge_start_utc: str | None = None
        self._bridge_artist: str | None = None
        self._bridge_title: str | None = None
        self._bridge_brand: str | None = None
        self._bridge_transcript: str | None = None
        self._bridge_confidence: float | None = None

    def feed(self, inp: SegmentInput | None) -> list[SegmentTransition]:
        if inp is None:
            return []

        if self._current_key is None:
            self._current_key = inp.key
            self._current_start_utc = inp.window_start_utc
            self._merge_display(inp)
            return []

        if self._bridge_start_utc is not None:
            return self._feed_with_pending_unknown_song_bridge(inp)

        if self._can_start_unknown_song_bridge(inp):
            self._start_unknown_song_bridge(inp)
            return []

        if inp.key == self._current_key:
            self._merge_display(inp)
            return []

        if self._can_merge_adjacent_commercial(inp):
            self._merge_commercial_identity(inp)
            self._merge_display(inp)
            return []

        assert self._current_start_utc is not None
        closed = self._current_transition(inp.window_start_utc)
        self._start_current(inp)
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
        if self._bridge_start_utc is not None:
            closed = [
                self._current_transition(self._bridge_start_utc),
                self._unknown_bridge_transition(end_iso),
            ]
        else:
            closed = [self._current_transition(end_iso)]
        self._reset_current()
        self._reset_bridge()
        return closed

    def _feed_with_pending_unknown_song_bridge(self, inp: SegmentInput) -> list[SegmentTransition]:
        assert self._current_key is not None
        assert self._current_start_utc is not None
        assert self._bridge_start_utc is not None

        if _is_unknown_song(inp.key):
            self._merge_bridge_display(inp)
            return []

        if (
            inp.key == self._current_key
            and self._bridge_duration_before(inp.window_start_utc)
            <= self.max_unknown_song_bridge_seconds
        ):
            self._reset_bridge()
            self._merge_display(inp)
            return []

        closed = [
            self._current_transition(self._bridge_start_utc),
            self._unknown_bridge_transition(inp.window_start_utc),
        ]
        self._reset_bridge()
        self._start_current(inp)
        return closed

    def _current_transition(self, end_utc: str) -> SegmentTransition:
        assert self._current_key is not None
        assert self._current_start_utc is not None
        return SegmentTransition(
            timestamp_start=self._current_start_utc,
            timestamp_end=end_utc,
            category=self._current_key.category,
            artist=self._artist,
            track_title=self._title,
            brand_name=self._brand,
            song_id=self._current_key.song_id,
            commercial_id=self._current_key.commercial_id,
            transcript_excerpt=self._transcript,
            confidence=self._confidence,
        )

    def _unknown_bridge_transition(self, end_utc: str) -> SegmentTransition:
        assert self._bridge_start_utc is not None
        return SegmentTransition(
            timestamp_start=self._bridge_start_utc,
            timestamp_end=end_utc,
            category=BroadcastCategory.SONG,
            artist=self._bridge_artist,
            track_title=self._bridge_title,
            brand_name=self._bridge_brand,
            transcript_excerpt=self._bridge_transcript,
            confidence=self._bridge_confidence,
        )

    def _start_current(self, inp: SegmentInput) -> None:
        self._current_key = inp.key
        self._current_start_utc = inp.window_start_utc
        self._artist = inp.artist
        self._title = inp.track_title
        self._brand = inp.brand_name
        self._transcript = inp.transcript_excerpt
        self._confidence = inp.confidence

    def _reset_current(self) -> None:
        self._current_key = None
        self._current_start_utc = None
        self._artist = None
        self._title = None
        self._brand = None
        self._transcript = None
        self._confidence = None

    def _can_start_unknown_song_bridge(self, inp: SegmentInput) -> bool:
        return (
            self.max_unknown_song_bridge_seconds > 0
            and _is_identified_song(self._current_key)
            and _is_unknown_song(inp.key)
        )

    def _start_unknown_song_bridge(self, inp: SegmentInput) -> None:
        self._bridge_start_utc = inp.window_start_utc
        self._bridge_artist = inp.artist
        self._bridge_title = inp.track_title
        self._bridge_brand = inp.brand_name
        self._bridge_transcript = inp.transcript_excerpt
        self._bridge_confidence = inp.confidence

    def _merge_bridge_display(self, inp: SegmentInput) -> None:
        if inp.artist is not None:
            self._bridge_artist = inp.artist
        if inp.track_title is not None:
            self._bridge_title = inp.track_title
        if inp.brand_name is not None:
            self._bridge_brand = inp.brand_name
        if inp.transcript_excerpt is not None:
            self._bridge_transcript = inp.transcript_excerpt
        if inp.confidence is not None:
            self._bridge_confidence = inp.confidence

    def _reset_bridge(self) -> None:
        self._bridge_start_utc = None
        self._bridge_artist = None
        self._bridge_title = None
        self._bridge_brand = None
        self._bridge_transcript = None
        self._bridge_confidence = None

    def _bridge_duration_before(self, end_utc: str) -> float:
        assert self._bridge_start_utc is not None
        return duration_seconds(self._bridge_start_utc, end_utc)

    def _can_merge_adjacent_commercial(self, inp: SegmentInput) -> bool:
        if (
            self._current_key is None
            or self._current_start_utc is None
            or self._current_key.category is not BroadcastCategory.COMMERCIAL
            or inp.key.category is not BroadcastCategory.COMMERCIAL
        ):
            return False
        if self.max_adjacent_commercial_combined_seconds <= 0:
            return False
        current_duration = duration_seconds(self._current_start_utc, inp.window_start_utc)
        if current_duration > self.max_adjacent_commercial_combined_seconds:
            return False
        if _commercial_brand_identity(self._current_key, self._brand) != _commercial_brand_identity(
            inp.key,
            inp.brand_name,
        ):
            return False
        return (
            text_similarity(self._transcript or "", inp.transcript_excerpt or "")
            >= self.adjacent_commercial_similarity_threshold
        )

    def _merge_commercial_identity(self, inp: SegmentInput) -> None:
        assert self._current_key is not None
        if self._current_key.commercial_id is None and inp.key.commercial_id is not None:
            self._current_key = inp.key


def _commercial_brand_identity(key: SegmentKey, brand_name: str | None) -> str:
    from radio_classifier.brands import canonicalize_brand

    canonical = canonicalize_brand(brand_name)
    if canonical:
        return canonical.casefold()
    return key.brand_key or ""


def _is_identified_song(key: SegmentKey | None) -> bool:
    return (
        key is not None
        and key.category is BroadcastCategory.SONG
        and (key.song_id is not None or key.artist_key is not None or key.title_key is not None)
    )


def _is_unknown_song(key: SegmentKey) -> bool:
    return (
        key.category is BroadcastCategory.SONG
        and key.song_id is None
        and key.artist_key is None
        and key.title_key is None
    )
