"""Human and JSON formatting for Tier-2 acoustic results."""

from __future__ import annotations

import json

from radio_classifier.acoustic.types import AcousticResult


def format_acoustic_human(result: AcousticResult) -> str:
    top = ",".join(f"{n}:{p:.3f}" for n, p in result.top_classes[:3])
    return (
        f"acoustic: start_utc={result.window_start_utc} label={result.label.value} "
        f"music={result.music_prob:.3f} speech={result.speech_prob:.3f} "
        f"other={result.other_prob:.3f} top=[{top}]"
    )


def format_acoustic_json(result: AcousticResult) -> str:
    payload = {
        "window_start_utc": result.window_start_utc,
        "tier2_label": result.label.value,
        "music_prob": result.music_prob,
        "speech_prob": result.speech_prob,
        "other_prob": result.other_prob,
        "top_classes": [{"name": n, "prob": p} for n, p in result.top_classes],
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
