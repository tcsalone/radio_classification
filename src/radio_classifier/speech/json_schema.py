"""Pydantic models for the Ollama 5-class JSON response."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from radio_classifier.segments.types import BroadcastCategory
from radio_classifier.speech.types import BrandMention, CommercialSignature

__all__ = [
    "BrandMentionJson",
    "CommercialSignatureJson",
    "LlmClassificationJson",
    "speech_kind_from_llm",
]


class BrandMentionJson(BaseModel):
    """One LLM-emitted brand mention."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: Literal["paid_ad", "dj_shoutout", "tag"]


class CommercialSignatureJson(BaseModel):
    """LLM-emitted commercial signature (only for ``class == COMMERCIAL``)."""

    model_config = ConfigDict(extra="ignore")

    key_phrases: list[str] = Field(default_factory=list)
    duration_bucket_seconds: int = Field(ge=5, le=120)

    @field_validator("key_phrases")
    @classmethod
    def strip_blank_phrases(cls, v: list[str]) -> list[str]:
        return [p.strip() for p in v if p and p.strip()]


class LlmClassificationJson(BaseModel):
    """Full LLM output for one transcript."""

    model_config = ConfigDict(extra="ignore")

    # Field is named "class" in the protocol but Python forbids it as an
    # identifier; expose as ``category`` and use a populator-style alias.
    category: Literal["SONG", "DJ", "COMMERCIAL", "STATION", "PSA_NEWS"] = Field(
        alias="class",
    )
    brand: str | None = None
    brand_mentions: list[BrandMentionJson] = Field(default_factory=list)
    commercial_signature: CommercialSignatureJson | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def empty_confidence_to_none(cls, v: object) -> object:
        if v == "" or v is None:
            return None
        return v

    @field_validator("brand", mode="before")
    @classmethod
    def empty_brand_to_none(cls, v: object) -> object:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @model_validator(mode="after")
    def _require_signature_for_commercial(self) -> "LlmClassificationJson":
        if self.category == "COMMERCIAL" and self.commercial_signature is None:
            # Tolerate older models that omit it; the resolver will still treat
            # the segment as a generic, un-resolvable commercial.
            return self
        if self.category != "COMMERCIAL" and self.commercial_signature is not None:
            # Strip an unexpected signature rather than fail validation.
            self.commercial_signature = None
        return self


def speech_kind_from_llm(
    data: LlmClassificationJson,
) -> tuple[BroadcastCategory, str | None, list[BrandMention], CommercialSignature | None]:
    """Project the validated LLM JSON onto our domain types."""
    cat = BroadcastCategory(data.category)
    brand = data.brand
    mentions = [BrandMention(name=m.name, mention_type=m.type) for m in data.brand_mentions]
    sig = (
        CommercialSignature(
            key_phrases=list(data.commercial_signature.key_phrases),
            duration_bucket_seconds=int(data.commercial_signature.duration_bucket_seconds),
        )
        if data.commercial_signature is not None
        else None
    )
    return cat, brand, mentions, sig
