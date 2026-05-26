"""Broadcast segment types for state machine + SQLite (schema v2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BroadcastCategory(str, Enum):
    """Must match ``broadcast_events.category`` CHECK constraint (schema v2)."""

    SONG = "SONG"
    DJ = "DJ"
    COMMERCIAL = "COMMERCIAL"
    STATION = "STATION"
    PSA_NEWS = "PSA_NEWS"


@dataclass(frozen=True)
class SegmentKey:
    """Normalized identity for segment equality.

    The reducer treats two consecutive windows as the **same** segment when
    their ``SegmentKey`` is equal. Display fields on :class:`SegmentInput`
    (artist, title, brand, transcript) are merged across the segment with
    "last non-None wins" semantics.

    Identity choices per class:

    * ``SONG`` — ``(SONG, artist_key, title_key, song_id)`` for identified
      tracks; ``(SONG, None, None, None)`` for unknown / unidentified music.
      ``song_id`` participates in equality so back-to-back identified tracks
      don't merge.
    * ``DJ`` / ``STATION`` / ``PSA_NEWS`` — ``(class, None, None, brand_key)``
      (brand_key is usually ``None``; included for sponsored station IDs).
    * ``COMMERCIAL`` — ``(COMMERCIAL, None, None, brand_key)`` with
      ``commercial_id`` participating via the extra field so two different
      ads back-to-back for the same brand don't merge.
    """

    category: BroadcastCategory
    artist_key: str | None = None
    title_key: str | None = None
    brand_key: str | None = None
    song_id: int | None = None
    commercial_id: int | None = None


@dataclass
class SegmentInput:
    """One window's contribution to the segment reducer.

    Display fields (``artist``, ``track_title``, ``brand_name``,
    ``transcript_excerpt``, ``confidence``) are written to SQLite. When the
    segment key is unchanged across windows, **the last non-None value wins**
    for each display field.
    """

    window_start_utc: str
    key: SegmentKey
    artist: str | None = None
    track_title: str | None = None
    brand_name: str | None = None
    transcript_excerpt: str | None = None
    confidence: float | None = None


@dataclass
class SegmentTransition:
    """A **closed** contiguous segment ready for persistence."""

    timestamp_start: str
    timestamp_end: str
    category: BroadcastCategory
    artist: str | None = None
    track_title: str | None = None
    brand_name: str | None = None
    song_id: int | None = None
    commercial_id: int | None = None
    brand_id: int | None = None
    transcript_excerpt: str | None = None
    confidence: float | None = None
