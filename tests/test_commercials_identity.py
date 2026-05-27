"""Commercial identity resolver — MinHash + cosine over LLM signature + transcript."""

from __future__ import annotations

import pytest

datasketch = pytest.importorskip("datasketch")

from pathlib import Path

from radio_classifier.commercials.identity import (
    CommercialIdentityResolver,
    CommercialResolverConfig,
    bucket_duration,
)
from radio_classifier.persistence import BroadcastStore
from radio_classifier.speech.types import CommercialSignature


def _resolver(tmp_path: Path) -> tuple[CommercialIdentityResolver, BroadcastStore]:
    store = BroadcastStore(tmp_path / "rc.db")
    return CommercialIdentityResolver(store=store), store


def test_bucket_duration_rounds_to_nearest_five() -> None:
    assert bucket_duration(12.4) == 10
    assert bucket_duration(12.6) == 15
    assert bucket_duration(30.0) == 30


def test_resolver_inserts_then_matches(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        signature = CommercialSignature(
            key_phrases=["save fifteen percent", "car insurance", "1-800-947-AUTO"],
            duration_bucket_seconds=15,
        )
        transcript_a = (
            "Tired of high insurance rates Switch to Geico and save fifteen percent "
            "or more on car insurance Call 1-800-947-AUTO today"
        )
        r1 = resolver.resolve(
            brand="Geico",
            transcript=transcript_a,
            duration_seconds=15.0,
            signature=signature,
        )
        assert r1.was_new is True
        assert r1.commercial_id is not None
        assert r1.reason == "inserted"

        # Small ASR variation of the SAME ad recording — only a couple words
        # differ — should still match the primary Jaccard threshold.
        transcript_b = (
            "Tired of high insurance rates Switch to Geico and save fifteen percent "
            "or more on car insurance Call 1-800-947-AUTO right now"
        )
        r2 = resolver.resolve(
            brand="Geico",
            transcript=transcript_b,
            duration_seconds=15.0,
            signature=signature,
        )
        assert r2.was_new is False
        assert r2.commercial_id == r1.commercial_id
        assert r2.reason == "matched"

        play_count = store.connection.execute(
            "SELECT play_count FROM commercials WHERE id = ?", (r1.commercial_id,)
        ).fetchone()[0]
        assert play_count == 2  # 1 from insert, 1 from increment
    finally:
        store.close()


def test_resolver_distinguishes_different_brands(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        sig = CommercialSignature(
            key_phrases=["lets go places", "new lineup"],
            duration_bucket_seconds=30,
        )
        r1 = resolver.resolve(
            brand="Toyota",
            transcript="Visit your local Toyota dealer Lets go places new lineup",
            duration_seconds=30.0,
            signature=sig,
        )
        r2 = resolver.resolve(
            brand="Honda",
            transcript="Visit your local Honda dealer Lets go places new lineup",
            duration_seconds=30.0,
            signature=sig,
        )
        assert r1.commercial_id != r2.commercial_id
    finally:
        store.close()


def test_resolver_skips_when_brand_missing(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        r = resolver.resolve(
            brand=None,
            transcript="some text",
            duration_seconds=15.0,
            signature=None,
        )
        assert r.commercial_id is None
        assert r.reason == "skipped_no_brand"
    finally:
        store.close()


def test_resolver_canonicalizes_known_brand_variants(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        r1 = resolver.resolve(
            brand="Rolaid",
            transcript="For fast heartburn relief choose Rolaid tablets today",
            duration_seconds=20.0,
            signature=CommercialSignature(
                key_phrases=["heartburn relief", "Rolaid tablets"],
                duration_bucket_seconds=20,
            ),
        )
        r2 = resolver.resolve(
            brand="Rolaids",
            transcript="For fast heartburn relief choose Rolaids tablets today",
            duration_seconds=20.0,
            signature=CommercialSignature(
                key_phrases=["heartburn relief", "Rolaids tablets"],
                duration_bucket_seconds=20,
            ),
        )

        assert r1.brand_id == r2.brand_id
        row = store.connection.execute(
            "SELECT canonical_name FROM brands WHERE id = ?", (r1.brand_id,)
        ).fetchone()
        assert row[0] == "Rolaids"
    finally:
        store.close()


def test_resolver_canonicalizes_graton_resort_variants(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        r1 = resolver.resolve(
            brand="Creighton Resort and Casino",
            transcript="Visit Graton Resort and Casino for gaming dining and entertainment",
            duration_seconds=20.0,
            signature=None,
        )
        r2 = resolver.resolve(
            brand="Greaten Resort and Casino",
            transcript="Visit Graton Resort and Casino for gaming dining and entertainment",
            duration_seconds=20.0,
            signature=None,
        )

        assert r1.brand_id == r2.brand_id
        row = store.connection.execute(
            "SELECT canonical_name FROM brands WHERE id = ?", (r1.brand_id,)
        ).fetchone()
        assert row[0] == "Graton Resort and Casino"
    finally:
        store.close()


def test_resolver_skips_short_or_long_segments(tmp_path: Path) -> None:
    resolver, store = _resolver(tmp_path)
    try:
        too_short = resolver.resolve(
            brand="Toyota",
            transcript="x",
            duration_seconds=5.0,
            signature=None,
        )
        too_long = resolver.resolve(
            brand="Toyota",
            transcript="x",
            duration_seconds=200.0,
            signature=None,
        )
        assert too_short.reason == "skipped_duration"
        assert too_long.reason == "skipped_duration"
    finally:
        store.close()


def test_resolver_secondary_threshold_kicks_in(tmp_path: Path) -> None:
    """Two commercials with moderate Jaccard but high cosine should match."""
    resolver, store = _resolver(
        tmp_path,
    )
    try:
        config = CommercialResolverConfig(
            jaccard_primary=0.95,  # force secondary path
            jaccard_secondary=0.20,
            cosine_secondary=0.50,
        )
        resolver.config = config
        transcript_a = (
            "Toyota of Springfield is having a sale this weekend new and pre owned"
        )
        transcript_b = (
            "Toyota of Springfield sale this weekend on new and pre owned vehicles"
        )
        sig = CommercialSignature(
            key_phrases=["Toyota of Springfield", "weekend sale"],
            duration_bucket_seconds=30,
        )
        r1 = resolver.resolve(
            brand="Toyota",
            transcript=transcript_a,
            duration_seconds=30.0,
            signature=sig,
        )
        r2 = resolver.resolve(
            brand="Toyota",
            transcript=transcript_b,
            duration_seconds=30.0,
            signature=sig,
        )
        assert r1.commercial_id is not None
        assert r2.commercial_id == r1.commercial_id
    finally:
        store.close()
