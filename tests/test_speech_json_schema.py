"""Pydantic validation + LLM-to-domain mapping for the 5-class schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from radio_classifier.segments.types import BroadcastCategory
from radio_classifier.speech.json_schema import LlmClassificationJson, speech_kind_from_llm


def _commercial_payload() -> dict:
    return {
        "class": "COMMERCIAL",
        "brand": "Geico",
        "brand_mentions": [{"name": "Geico", "type": "paid_ad"}],
        "commercial_signature": {
            "key_phrases": ["save 15%", "car insurance"],
            "duration_bucket_seconds": 15,
        },
        "confidence": 0.92,
        "rationale": "Direct ad with brand and CTA.",
    }


def test_validates_commercial_payload() -> None:
    obj = LlmClassificationJson.model_validate(_commercial_payload())
    cat, brand, mentions, sig = speech_kind_from_llm(obj)
    assert cat is BroadcastCategory.COMMERCIAL
    assert brand == "Geico"
    assert [(m.name, m.mention_type) for m in mentions] == [("Geico", "paid_ad")]
    assert sig is not None
    assert sig.key_phrases == ["save 15%", "car insurance"]
    assert sig.duration_bucket_seconds == 15


def test_dj_payload_without_brand() -> None:
    obj = LlmClassificationJson.model_validate(
        {
            "class": "DJ",
            "brand": "",  # empty -> normalized to None
            "brand_mentions": [],
            "commercial_signature": None,
            "confidence": 0.8,
            "rationale": "DJ banter.",
        }
    )
    cat, brand, _, sig = speech_kind_from_llm(obj)
    assert cat is BroadcastCategory.DJ
    assert brand is None
    assert sig is None


def test_strips_unexpected_signature_on_non_commercial() -> None:
    obj = LlmClassificationJson.model_validate(
        {
            "class": "DJ",
            "brand": None,
            "brand_mentions": [],
            "commercial_signature": {
                "key_phrases": ["should not be here"],
                "duration_bucket_seconds": 20,
            },
            "confidence": 0.5,
            "rationale": "...",
        }
    )
    assert obj.commercial_signature is None


def test_rejects_invalid_class() -> None:
    with pytest.raises(ValidationError):
        LlmClassificationJson.model_validate(
            {"class": "BOGUS", "brand": None, "brand_mentions": []}
        )


def test_blank_key_phrases_filtered() -> None:
    obj = LlmClassificationJson.model_validate(
        {
            "class": "COMMERCIAL",
            "brand": "Toyota",
            "brand_mentions": [],
            "commercial_signature": {
                "key_phrases": ["lets go places", "", "  ", "downtown"],
                "duration_bucket_seconds": 10,
            },
        }
    )
    assert obj.commercial_signature.key_phrases == ["lets go places", "downtown"]
