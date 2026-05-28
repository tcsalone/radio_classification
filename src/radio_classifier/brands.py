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
    normalize_token("Creighton Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Greaten Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Grayton Resort and Casino"): "Graton Resort and Casino",
    normalize_token("Graton Resort and Casino"): "Graton Resort and Casino",
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
