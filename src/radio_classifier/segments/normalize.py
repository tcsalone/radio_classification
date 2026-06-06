"""Helpers to turn tier results into :class:`SegmentInput` rows.

This module is intentionally I/O-free. Adapters here may import lightweight
types from :mod:`radio_classifier.fingerprint`, :mod:`...speech`, and
:mod:`...music`; they MUST NOT import the orchestrator or persistence layer.
"""

from __future__ import annotations

import re

from radio_classifier.segments.types import (
    BroadcastCategory,
    SegmentInput,
    SegmentKey,
)


def normalize_token(s: str | None) -> str | None:
    """Normalize a single token for segment key equality.

    ``None`` or blank → ``None``. Otherwise: strip, casefold, collapse internal
    whitespace, strip trailing/leading ``.`` ``'`` and Unicode apostrophe ``'``
    (U+2019).
    """
    if s is None:
        return None
    t = " ".join(s.split())
    if not t:
        return None
    t = t.casefold()
    t = re.sub(r"^[`'’.]+|[`'’.]+$", "", t)
    return t or None


def segment_input_for_song(
    *,
    window_start_utc: str,
    artist: str | None,
    title: str | None,
    song_id: int | None,
    confidence: float | None = None,
) -> SegmentInput:
    """Build a ``SONG`` :class:`SegmentInput` from a Tier-1 / Shazam match.

    When ``song_id`` is set it is the **sole** identity for segment equality:
    ``artist_key``/``title_key`` are left ``None``. ``song_id`` is the canonical
    id returned by :meth:`BroadcastStore.upsert_song`, which already folds the
    apostrophe/underscore/feature-suffix drift between Shazam (``Picking
    Dragons' Pockets``) and the audfprint reference filename (``Picking
    Dragons_ Pockets``). Keying the reducer on the raw normalized title instead
    fragmented one continuous play into many tiny events whenever consecutive
    windows alternated between the two detectors. Riding on ``song_id`` keeps a
    single play contiguous regardless of which tier identified each window.

    ``song_id=None`` is permitted (unknown-song segment from Tier 2). In that
    case the key still normalizes by ``(artist_key, title_key)`` so back-to-back
    unknown songs merge into one segment (we can't distinguish them).
    """
    has_song_id = song_id is not None
    return SegmentInput(
        window_start_utc=window_start_utc,
        key=SegmentKey(
            category=BroadcastCategory.SONG,
            artist_key=None if has_song_id else normalize_token(artist),
            title_key=None if has_song_id else normalize_token(title),
            brand_key=None,
            song_id=song_id,
            commercial_id=None,
        ),
        artist=artist,
        track_title=title,
        brand_name=None,
        transcript_excerpt=None,
        confidence=confidence,
    )


def segment_input_for_unknown_song(
    *,
    window_start_utc: str,
    confidence: float | None = None,
) -> SegmentInput:
    """``SONG`` segment with no identification — Tier 2 fall-through."""
    return SegmentInput(
        window_start_utc=window_start_utc,
        key=SegmentKey(
            category=BroadcastCategory.SONG,
            artist_key=None,
            title_key=None,
            brand_key=None,
            song_id=None,
            commercial_id=None,
        ),
        confidence=confidence,
    )


def segment_input_for_speech(
    *,
    window_start_utc: str,
    category: BroadcastCategory,
    brand: str | None,
    commercial_id: int | None,
    transcript_excerpt: str | None,
    confidence: float | None = None,
) -> SegmentInput:
    """Build a SPEECH-derived segment input (``DJ`` / ``COMMERCIAL`` / ``STATION`` / ``PSA_NEWS``).

    ``commercial_id`` is the resolver's output (or ``None`` for un-resolvable
    commercials). It participates in the segment key so a stretch of identical
    ad plays back-to-back gets ONE segment, while two different ads for the
    same brand stay separate.
    """
    if category is BroadcastCategory.SONG:
        raise ValueError("use segment_input_for_song for SONG segments")
    return SegmentInput(
        window_start_utc=window_start_utc,
        key=SegmentKey(
            category=category,
            artist_key=None,
            title_key=None,
            brand_key=normalize_token(brand),
            song_id=None,
            commercial_id=commercial_id,
        ),
        brand_name=brand,
        transcript_excerpt=transcript_excerpt,
        confidence=confidence,
    )
