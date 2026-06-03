"""Deterministic brand extraction from commercial transcripts."""

from __future__ import annotations

from radio_classifier.commercials.brand_extract import extract_brand_from_text


def test_extracts_brand_from_url() -> None:
    text = "See website for details, billshappen.com is not a lender."
    assert extract_brand_from_text(text) == "BillsHappen.com"


def test_known_phrase_beats_no_url() -> None:
    text = "There's a solution. National Debt Relief could put you on the fast track."
    assert extract_brand_from_text(text) == "National Debt Relief"


def test_url_brand_folds_through_alias_table() -> None:
    # selectquote.com → SelectQuote via the brand alias table.
    text = "Details on example rate at selectquote.com slash terms."
    assert extract_brand_from_text(text) == "SelectQuote"


def test_uncurated_domain_is_not_guessed() -> None:
    # Precision-first: a concatenated/unknown domain is left for the LLM tier
    # rather than minting a junk brand like "Acmewidgets" that would never
    # merge with the real, properly-spaced advertiser.
    text = "Get a quote in seconds at acmewidgets.com today."
    assert extract_brand_from_text(text) is None
    assert extract_brand_from_text("at primemalemedical.com now") is None


def test_psa_and_generic_domains_are_ignored() -> None:
    assert extract_brand_from_text("a CalMatters.org investigation found") is None
    assert extract_brand_from_text("learn more at standuptocancer.org") is None
    assert extract_brand_from_text("track it at usps.com for free") is None


def test_boilerplate_only_returns_none() -> None:
    text = (
        "Restrictions apply. Must be 18 years or older. Please play responsibly. "
        "See store for details. Offer ends soon."
    )
    assert extract_brand_from_text(text) is None


def test_blank_input_returns_none() -> None:
    assert extract_brand_from_text(None) is None
    assert extract_brand_from_text("") is None
    assert extract_brand_from_text("   ") is None


def test_catchphrase_maps_to_brand() -> None:
    assert extract_brand_from_text("Pet friendly. You spray, they play.") == "Spruce"
