"""Brand name normalization for LLM-derived commercial metadata."""

from __future__ import annotations

import re

from radio_classifier.segments.normalize import normalize_token

_SPACE_RE = re.compile(r"\s+")


def _clean_brand(name: str) -> str:
    return _SPACE_RE.sub(" ", name.strip())


_BRAND_ALIASES: dict[str, str] = {
    # Whisper/LLM variants seen in morning-drive captures.
    normalize_token("Rolaid"): "Rolaids",
    normalize_token("Rolaids"): "Rolaids",
    # Graton Resort and Casino — Whisper hears the brand 4+ different ways.
    normalize_token("Creighton Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Greaten Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Grayton Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Grayton Resort & Casino"): "Graton Resort and Casino",
    normalize_token("Graton Resort & Casino"): "Graton Resort and Casino",
    normalize_token("GreatOn.com"): "Graton Resort and Casino",
    normalize_token("Greaton.com"): "Graton Resort and Casino",
    normalize_token("Graton.com"): "Graton Resort and Casino",
    normalize_token("Zorton Casino"): "Graton Resort and Casino",
    normalize_token("Graton Resort and Casino"): "Graton Resort and Casino",
    # Izervay — Whisper consistently mangles this FDA-approved eye treatment.
    # Validated 2026-05-28: same ad appeared in DB as Eyservé, iZERVE, and
    # EvasenQ across consecutive windows.
    normalize_token("Izervay"): "Izervay",
    normalize_token("Izerve"): "Izervay",
    normalize_token("iZERVE"): "Izervay",
    normalize_token("Eyservé"): "Izervay",
    normalize_token("Eyserve"): "Izervay",
    normalize_token("EvasenQ"): "Izervay",
    normalize_token("Eye Survey"): "Izervay",
}


def canonicalize_brand(name: str | None) -> str | None:
    """Return a canonical brand name for known ASR/LLM variants.

    Unknown brands are only whitespace-normalized. The alias table is kept
    deliberately small and evidence-driven so unrelated advertisers are not
    merged by fuzzy matching.
    """
    if name is None:
        return None
    cleaned = _clean_brand(name)
    if not cleaned:
        return None
    key = normalize_token(cleaned)
    return _BRAND_ALIASES.get(key, cleaned)
