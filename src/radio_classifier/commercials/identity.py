"""Text-derived commercial identity resolver.

Tier 1 deliberately does not fingerprint commercials (see SPEC.md §1.2). When
the LLM labels a window as ``COMMERCIAL`` we resolve "is this the same ad we
heard before?" purely from text:

1. Normalize the brand and look it up in / insert into ``brands``.
2. Bucket the segment's duration to the nearest 5 seconds.
3. Build a MinHash over ``key_phrases`` ∪ word-3-shingles(transcript).
4. Query ``commercials`` for ``(brand_id, duration_bucket_seconds)`` candidates;
   pick a match if MinHash Jaccard ≥ 0.70, OR (Jaccard ≥ 0.55 AND transcript
   word-cosine ≥ 0.85).
5. If no match, insert a fresh ``commercials`` row.

Steps 3–4 use the deterministic ``datasketch`` MinHash so unit tests can pin
golden vectors. ``num_perm = 128`` per SPEC.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.normalize import normalize_token
from radio_classifier.speech.types import CommercialSignature


_DEFAULT_NUM_PERM = 128
_BUCKET_SECONDS = 5
_MIN_SEGMENT_SECONDS = 10.0
_MAX_SEGMENT_SECONDS = 90.0
_DEFAULT_JACCARD_PRIMARY = 0.70
_DEFAULT_JACCARD_SECONDARY = 0.55
_DEFAULT_COSINE_SECONDARY = 0.85


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def _word_shingles(tokens: Sequence[str], n: int = 3) -> set[str]:
    if len(tokens) < n:
        return set(tokens)
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _cosine_similarity_words(a: Sequence[str], b: Sequence[str]) -> float:
    """Bag-of-words cosine on raw token counts; tolerant to ordering."""
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    shared = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in shared)
    da = math.sqrt(sum(v * v for v in ca.values()))
    db = math.sqrt(sum(v * v for v in cb.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _build_minhash(features: Iterable[str], *, num_perm: int = _DEFAULT_NUM_PERM):
    from datasketch import MinHash  # type: ignore

    mh = MinHash(num_perm=num_perm)
    for f in features:
        mh.update(f.encode("utf-8"))
    return mh


def _minhash_to_hex(mh, *, num_perm: int = _DEFAULT_NUM_PERM) -> str:
    arr = mh.hashvalues  # numpy uint64 array
    if len(arr) != num_perm:
        raise ValueError("MinHash permutation count mismatch")
    return arr.tobytes().hex()


def _minhash_from_hex(blob: str, *, num_perm: int = _DEFAULT_NUM_PERM):
    import numpy as np
    from datasketch import MinHash  # type: ignore

    raw = bytes.fromhex(blob)
    arr = np.frombuffer(raw, dtype=np.uint64).copy()
    if arr.shape[0] != num_perm:
        raise ValueError("hex blob does not match num_perm")
    mh = MinHash(num_perm=num_perm)
    mh.hashvalues = arr
    return mh


def bucket_duration(duration_seconds: float, *, bucket: int = _BUCKET_SECONDS) -> int:
    """Round to the nearest multiple of ``bucket`` seconds."""
    return int(round(duration_seconds / bucket) * bucket)


@dataclass
class CommercialResolution:
    """Outcome of a single resolver call."""

    commercial_id: int | None
    brand_id: int | None
    duration_bucket_seconds: int | None
    was_new: bool
    reason: str  # 'matched' | 'inserted' | 'skipped_no_brand' | 'skipped_duration' | 'disabled'


@dataclass
class CommercialResolverConfig:
    """Tunable thresholds (see SPEC §5)."""

    num_perm: int = _DEFAULT_NUM_PERM
    jaccard_primary: float = _DEFAULT_JACCARD_PRIMARY
    jaccard_secondary: float = _DEFAULT_JACCARD_SECONDARY
    cosine_secondary: float = _DEFAULT_COSINE_SECONDARY
    bucket_seconds: int = _BUCKET_SECONDS
    min_segment_seconds: float = _MIN_SEGMENT_SECONDS
    max_segment_seconds: float = _MAX_SEGMENT_SECONDS


class CommercialIdentityResolver:
    """Resolve a commercial transcript to a stable ``commercial_id``.

    Construction is cheap; the resolver holds no per-call state. ``datasketch``
    is imported lazily so the rest of the package works without it installed,
    as long as no resolver call is made.
    """

    def __init__(
        self,
        store: BroadcastStore,
        config: CommercialResolverConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or CommercialResolverConfig()

    def resolve(
        self,
        *,
        brand: str | None,
        transcript: str,
        duration_seconds: float,
        signature: CommercialSignature | None,
    ) -> CommercialResolution:
        cfg = self.config

        if not brand or not brand.strip():
            return CommercialResolution(
                commercial_id=None,
                brand_id=None,
                duration_bucket_seconds=None,
                was_new=False,
                reason="skipped_no_brand",
            )
        if duration_seconds < cfg.min_segment_seconds or duration_seconds > cfg.max_segment_seconds:
            return CommercialResolution(
                commercial_id=None,
                brand_id=None,
                duration_bucket_seconds=None,
                was_new=False,
                reason="skipped_duration",
            )

        canonical_brand = brand.strip()
        brand_id = self.store.upsert_brand(canonical_brand)

        bucket = bucket_duration(duration_seconds, bucket=cfg.bucket_seconds)
        tokens = _tokenize(transcript)
        # Lowercase key_phrases so they collide with the 3-shingle tokens
        # produced from the transcript (which is also lowercased).
        signature_phrases = (
            [p.lower() for p in signature.key_phrases] if signature else []
        )
        features = set(signature_phrases) | _word_shingles(tokens)
        # Add the brand itself as a feature — it anchors the signature to the
        # canonical advertiser even when the LLM omits key_phrases.
        features.add(
            f"__brand__:{normalize_token(canonical_brand) or canonical_brand.lower()}"
        )
        if not features:
            features = {f"__brand__:{normalize_token(canonical_brand) or canonical_brand.lower()}"}
        new_mh = _build_minhash(features, num_perm=cfg.num_perm)
        new_hex = _minhash_to_hex(new_mh, num_perm=cfg.num_perm)

        candidates = self.store.find_commercials_for_brand(brand_id, bucket)
        for c_id, cand_hex, cand_transcript in candidates:
            try:
                cand_mh = _minhash_from_hex(cand_hex, num_perm=cfg.num_perm)
            except ValueError:
                continue
            jaccard = float(new_mh.jaccard(cand_mh))
            if jaccard >= cfg.jaccard_primary:
                self.store.increment_commercial_play_count(c_id)
                return CommercialResolution(
                    commercial_id=c_id,
                    brand_id=brand_id,
                    duration_bucket_seconds=bucket,
                    was_new=False,
                    reason="matched",
                )
            if jaccard >= cfg.jaccard_secondary:
                cosine = _cosine_similarity_words(tokens, _tokenize(cand_transcript))
                if cosine >= cfg.cosine_secondary:
                    self.store.increment_commercial_play_count(c_id)
                    return CommercialResolution(
                        commercial_id=c_id,
                        brand_id=brand_id,
                        duration_bucket_seconds=bucket,
                        was_new=False,
                        reason="matched",
                    )

        new_id = self.store.insert_commercial(
            brand_id=brand_id,
            duration_bucket_seconds=bucket,
            minhash_hex=new_hex,
            reference_transcript=transcript or "",
        )
        return CommercialResolution(
            commercial_id=new_id,
            brand_id=brand_id,
            duration_bucket_seconds=bucket,
            was_new=True,
            reason="inserted",
        )
