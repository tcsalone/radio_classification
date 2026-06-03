"""Deterministic brand extraction from commercial transcripts.

The Tier 3 LLM frequently returns ``brand=null`` on short ad fragments (10s
windows that catch only a tail or head of a spot), which leaves the event in
the dashboard's "Unbranded / unidentified" bucket. This module recovers a
brand from the transcript text *without* an LLM, using only high-precision
signals so false positives stay near zero:

1. A small, evidence-driven **known-phrase table** (catchphrases / spoken brand
   names that are unambiguous).
2. A **URL / domain** extractor (``foo.com`` → ``Foo``), with a stoplist for
   generic or PSA/news domains that are not advertisers.

Everything is routed through :func:`canonicalize_brand` so recovered brands
fold into the existing alias table automatically. Anything ambiguous returns
``None`` (the event stays unbranded) — recall is intentionally sacrificed for
precision; the optional LLM tier in :mod:`.backfill` handles the remainder.
"""

from __future__ import annotations

import re

from radio_classifier.brands import canonicalize_brand

# Known spoken phrases → canonical brand. Kept deliberately small and specific
# so a phrase can only belong to one advertiser. Seeded from the 2026-06-02
# 20h run's unbranded bucket. Matched case-insensitively as substrings of the
# raw transcript.
_KNOWN_PHRASES: list[tuple[str, str]] = [
    ("national debt relief", "National Debt Relief"),
    ("community tax", "Community Tax"),
    ("billshappen", "BillsHappen.com"),
    ("bills happen", "BillsHappen.com"),
    ("river rock casino", "River Rock Casino"),
    ("advertise with live 105", "Live 105"),
    ("you spray, they play", "Spruce"),
    ("you spray they play", "Spruce"),
    ("big lou", "Big Lou"),
    ("biglou", "Big Lou"),
    ("select quote", "SelectQuote"),
    ("selectquote", "SelectQuote"),
    ("ziprecruiter", "ZipRecruiter"),
    ("jumpstart md", "JumpStartMD"),
    ("jumpstartmd", "JumpStartMD"),
]

# URL/domain stem → canonical brand. Curated on purpose: a concatenated domain
# like ``primemalemedical`` cannot be reliably re-spaced into the real brand
# (``Prime Male Medical``), so blindly title-casing every domain would mint new
# brand rows that never merge with the existing multi-word ones — re-creating
# the very duplication the rollup/dedupe work removed. We therefore only accept
# domains we have an explicit, clean mapping for; the LLM tier handles the rest
# with correctly-spaced names.
_DOMAIN_BRANDS: dict[str, str] = {
    "billshappen": "BillsHappen.com",
    "bigloo": "Big Lou",
    "biglou": "Big Lou",
    "audacy": "Live 105",
    "odyssey": "Live 105",
    "mirastarfcu": "MiraStar FCU",
    "selectquote": "SelectQuote",
    "ziprecruiter": "ZipRecruiter",
    "ethos": "Ethos",
}

# Domains that are NOT advertisers (PSA/news/government/charity) or are too
# generic to be a reliable brand signal. These never produce a brand.
_DOMAIN_STOPLIST: frozenset[str] = frozenset(
    {
        "calmatters",
        "usps",
        "standuptocancer",
        "children",
        "medical",
        "www",
        "google",
        "youtube",
        "facebook",
        "instagram",
        "gov",
    }
)

_URL_RE = re.compile(r"\b([a-z0-9][a-z0-9\-]{2,30})\.(?:com|org|net)\b", re.IGNORECASE)


def _domain_to_brand(stem: str) -> str | None:
    """Map a bare domain stem to a *curated* brand, or ``None``.

    Only domains with an explicit mapping in :data:`_DOMAIN_BRANDS` produce a
    brand. Generic title-casing is deliberately avoided (see the note on that
    table): concatenated domains cannot be re-spaced reliably and would
    fragment existing multi-word brands.
    """
    stem = stem.lower()
    if stem in _DOMAIN_STOPLIST:
        return None
    return _DOMAIN_BRANDS.get(stem)


def extract_brand_from_text(text: str | None) -> str | None:
    """Return a canonical brand recovered from ``text``, or ``None``.

    High precision by design: only known phrases and explicit URLs/domains
    yield a brand. The result is passed through :func:`canonicalize_brand` so
    it folds into the existing alias table.
    """
    if not text or not text.strip():
        return None
    lowered = text.casefold()

    for needle, brand in _KNOWN_PHRASES:
        if needle in lowered:
            return canonicalize_brand(brand)

    for match in _URL_RE.finditer(text):
        candidate = _domain_to_brand(match.group(1))
        if candidate is not None:
            return canonicalize_brand(candidate)

    return None
