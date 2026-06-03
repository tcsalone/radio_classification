"""Brand name normalization for LLM-derived commercial metadata."""

from __future__ import annotations

import re

from radio_classifier.segments.normalize import normalize_token

_SPACE_RE = re.compile(r"\s+")
# ``Smart & Final`` vs ``Smart and Final`` etc. come in interchangeably from the
# Whisper transcript / LLM-derived brand field. We collapse ``" & "`` (with
# surrounding whitespace) to ``" and "`` before the alias table is consulted so
# unmapped advertisers fold automatically. ``AT&T``-style tokens with no
# surrounding spaces are left alone because the ``&`` there is genuinely part
# of the brand spelling.
_AMP_RE = re.compile(r"\s+&\s+")


def _clean_brand(name: str) -> str:
    collapsed = _SPACE_RE.sub(" ", name.strip())
    return _AMP_RE.sub(" and ", collapsed)


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
    # Common 2026-05-30 advertiser variants / ASR slips.
    normalize_token("Smart & Final"): "Smart and Final",
    normalize_token("Smart and Final"): "Smart and Final",
    normalize_token("Golden State Lumber and Showroom"): "Golden State Lumber",
    normalize_token("Golden State Lumber"): "Golden State Lumber",
    normalize_token("Xfinity Mobile"): "Xfinity",
    normalize_token("Xfinity"): "Xfinity",
    normalize_token("Habes Law"): "Habas Law",
    normalize_token("Habus Law"): "Habas Law",
    normalize_token("Habas Law"): "Habas Law",
    normalize_token("Atco"): "ATCO",
    normalize_token("Atko"): "ATCO",
    normalize_token("ATCO"): "ATCO",
    normalize_token("Ambutra"): "Ambetter",
    normalize_token("Mbutra"): "Ambetter",
    normalize_token("Big Lou Insurance"): "Big Lou",
    normalize_token("Big Lou's Life Insurance"): "Big Lou",
    normalize_token("Big Lou"): "Big Lou",
    normalize_token("PF Chang's"): "P.F. Chang's",
    normalize_token("P.F. Chang's"): "P.F. Chang's",
    normalize_token("WOC Fire"): "Wokfire",
    normalize_token("Wokfire"): "Wokfire",
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
    # 2026-06-02 20h-run advertiser naming variants. Same advertiser emitted by
    # the LLM under two spellings, which fragmented the dashboard's commercial
    # rollup. Folding the variant into the more common canonical form.
    normalize_token("Ethos"): "Ethos",
    normalize_token("Ethos Insurance"): "Ethos",
    normalize_token("The Home Depot"): "The Home Depot",
    normalize_token("Home Depot"): "The Home Depot",
    normalize_token("Easy Cater"): "Easy Cater",
    normalize_token("EasyCater"): "Easy Cater",
    normalize_token("ezCater"): "Easy Cater",
    normalize_token("SelectQuote"): "SelectQuote",
    normalize_token("Select Quote"): "SelectQuote",
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
