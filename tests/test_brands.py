"""Brand-name canonicalization tests."""

from __future__ import annotations

from radio_classifier.brands import canonicalize_brand


def test_canonicalize_brand_passes_unknown_brands_through_unchanged() -> None:
    assert canonicalize_brand("Mystery Sponsor") == "Mystery Sponsor"


def test_canonicalize_brand_returns_none_for_blank_input() -> None:
    assert canonicalize_brand(None) is None
    assert canonicalize_brand("") is None
    assert canonicalize_brand("   ") is None


def test_canonicalize_brand_collapses_whitespace_and_trims() -> None:
    assert canonicalize_brand("  Smart   and   Final  ") == "Smart and Final"


def test_canonicalize_brand_normalizes_spaced_ampersand_to_and_for_known_alias() -> None:
    """``" & "`` should canonicalize to ``" and "`` before alias lookup."""
    assert canonicalize_brand("Smart & Final") == "Smart and Final"
    assert canonicalize_brand("Graton Resort & Casino") == "Graton Resort and Casino"


def test_canonicalize_brand_generic_ampersand_for_unmapped_brand() -> None:
    """Unmapped brands with `` & `` still benefit from the generic rule.

    Without the rule a brand like ``"Bob & Sue's Diner"`` would compete with
    ``"Bob and Sue's Diner"`` for the same canonical bucket.
    """
    assert canonicalize_brand("Bob & Sue's Diner") == "Bob and Sue's Diner"


def test_canonicalize_brand_preserves_inline_ampersand_brand_spelling() -> None:
    """``AT&T``-style tokens keep their ampersand because the rule only fires
    when ``&`` has whitespace on both sides.
    """
    assert canonicalize_brand("AT&T") == "AT&T"
    assert canonicalize_brand("M&Ms") == "M&Ms"


def test_canonicalize_brand_keeps_existing_alias_entries_working() -> None:
    """Pre-existing Whisper-mishearing aliases must still resolve."""
    assert canonicalize_brand("Habes Law") == "Habas Law"
    assert canonicalize_brand("Wokfire") == "Wokfire"
    assert canonicalize_brand("WOC Fire") == "Wokfire"
