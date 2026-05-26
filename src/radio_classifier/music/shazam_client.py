"""Thin shazamio wrapper used only when ``--enable-shazam`` is passed.

This is the only network-touching code path in radio-classifier and it is
opt-in. Logic adapted from ``live105sux/src/live105sux/music/`` but slimmed
down: the change-gate and dedupe logic from live105sux can be added back as a
follow-up if Shazam usage proves heavy in practice.
"""

from __future__ import annotations

import asyncio
from typing import Any

from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.music.types import ShazamResult, ShazamStatus
from radio_classifier.speech.wav_temp import temp_wav_for_window


def _parse_recognition(raw: dict[str, Any]) -> tuple[str | None, str | None, float | None]:
    """Extract ``(artist, title, confidence)`` from a shazamio recognize payload."""
    track = raw.get("track") if isinstance(raw, dict) else None
    if not isinstance(track, dict):
        return None, None, None
    title = track.get("title")
    artist = track.get("subtitle")
    if not isinstance(title, str):
        title = None
    if not isinstance(artist, str):
        artist = None
    # shazamio doesn't expose a numeric confidence by default; leave None.
    return artist, title, None


def identify_window_sync(
    window: AudioWindow,
    *,
    min_confidence: float | None = None,
) -> ShazamResult:
    """Synchronous wrapper around the async shazamio API.

    Lazily imports shazamio so the module is importable without the extra.
    """
    try:
        from shazamio import Shazam  # type: ignore
    except ImportError as exc:
        return ShazamResult(
            status=ShazamStatus.skipped,
            window_start_utc=window.window_start_utc,
            message=f"shazamio not installed: {exc}",
        )

    async def _go() -> ShazamResult:
        shazam = Shazam()
        try:
            with temp_wav_for_window(window) as wav_path:
                raw = await shazam.recognize(str(wav_path))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            return ShazamResult(
                status=ShazamStatus.error,
                window_start_utc=window.window_start_utc,
                message=str(exc),
            )
        artist, title, conf = _parse_recognition(raw if isinstance(raw, dict) else {})
        if not title and not artist:
            return ShazamResult(
                status=ShazamStatus.no_match,
                window_start_utc=window.window_start_utc,
                raw=raw if isinstance(raw, dict) else None,
            )
        if min_confidence is not None and conf is not None and conf < min_confidence:
            return ShazamResult(
                status=ShazamStatus.low_confidence,
                window_start_utc=window.window_start_utc,
                artist=artist,
                title=title,
                confidence=conf,
                raw=raw,
            )
        return ShazamResult(
            status=ShazamStatus.match,
            window_start_utc=window.window_start_utc,
            artist=artist,
            title=title,
            confidence=conf,
            raw=raw,
        )

    try:
        return asyncio.run(_go())
    except RuntimeError:
        # Already inside an event loop (rare in CLI path). Use a fresh loop.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_go())
        finally:
            loop.close()
