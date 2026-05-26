"""Human and JSON formatting for speech pipeline results."""

from __future__ import annotations

import json

from radio_classifier.speech.types import SpeechPipelineResult


def format_speech_human(result: SpeechPipelineResult, *, transcript_max: int = 80) -> str:
    """Single stderr line with ``speech:`` prefix."""
    parts = ["speech:", f"start_utc={result.window_start_utc}", f"status={result.status.value}"]
    if result.category is not None:
        parts.append(f"category={result.category.value}")
    if result.brand is not None:
        parts.append(f"brand={result.brand!r}")
    if result.confidence is not None:
        parts.append(f"confidence={result.confidence:.4f}")
    if result.brand_mentions:
        mentions = ",".join(f"{m.name}:{m.mention_type}" for m in result.brand_mentions)
        parts.append(f"brand_mentions=[{mentions}]")
    if result.message:
        parts.append(f"message={result.message!r}")
    t = result.transcript or ""
    if len(t) > transcript_max:
        parts.append(f"transcript={t[:transcript_max]!r}...")
    else:
        parts.append(f"transcript={t!r}")
    return " ".join(parts)


def format_speech_json(result: SpeechPipelineResult) -> str:
    """Single JSON object line for stdout."""
    payload: dict = {
        "window_start_utc": result.window_start_utc,
        "status": result.status.value,
        "transcript": result.transcript,
        "category": result.category.value if result.category else None,
        "brand": result.brand,
        "brand_mentions": [
            {"name": m.name, "type": m.mention_type} for m in result.brand_mentions
        ],
        "commercial_signature": (
            {
                "key_phrases": list(result.commercial_signature.key_phrases),
                "duration_bucket_seconds": result.commercial_signature.duration_bucket_seconds,
            }
            if result.commercial_signature is not None
            else None
        ),
        "confidence": result.confidence,
        "rationale": result.rationale,
        "message": result.message,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
