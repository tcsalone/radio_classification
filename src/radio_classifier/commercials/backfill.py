"""Recover brands for unbranded commercial events.

COMMERCIAL events whose Tier 3 pass returned ``brand=null`` end up with no
``commercial_id`` and no ``brand_id``; the dashboard buckets them as
"Unbranded / unidentified". This module makes a second pass over those events
and tries to attribute a brand from the stored transcript:

1. A deterministic extractor (:func:`extract_brand_from_text`) — high
   precision, offline, no dependencies.
2. An optional LLM tier — re-classify the transcript text with the existing
   :class:`OllamaSpeechClassifier` and accept its ``brand`` when it labels the
   text ``COMMERCIAL``. This carries most of the recall for fragments where the
   brand is spoken in plain words (no URL / phone).

When a brand is recovered (and not a dry run) the event is re-pointed through
the standard :class:`CommercialIdentityResolver`, so it folds into an existing
ad for the same brand when the transcripts match — giving dedup for free — and
a ``paid_ad`` brand mention is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from radio_classifier.brands import canonicalize_brand
from radio_classifier.commercials.brand_extract import extract_brand_from_text
from radio_classifier.commercials.identity import CommercialIdentityResolver
from radio_classifier.persistence.broadcast_store import BroadcastStore


class _TranscriptClassifier(Protocol):
    """Minimal interface satisfied by ``OllamaSpeechClassifier``."""

    def classify_transcript(self, text: str):  # pragma: no cover - structural
        ...


@dataclass
class BrandBackfillItem:
    event_id: int
    brand: str
    source: str  # 'deterministic' | 'llm'
    commercial_id: int | None
    timestamp_start: str


@dataclass
class BrandBackfillReport:
    items: list[BrandBackfillItem]
    events_scanned: int
    deterministic_hits: int
    llm_hits: int
    llm_attempts: int
    llm_errors: int
    dry_run: bool

    @property
    def events_branded(self) -> int:
        return len(self.items)


def _load_unbranded(
    store: BroadcastStore,
    *,
    since_utc: str | None,
    until_utc: str | None,
    limit: int | None,
) -> list[tuple[int, float, str, str]]:
    clauses = [
        "category = 'COMMERCIAL'",
        "commercial_id IS NULL",
        "brand_id IS NULL",
        "(brand_name IS NULL OR brand_name = '')",
        "transcript_excerpt IS NOT NULL",
        "TRIM(transcript_excerpt) <> ''",
    ]
    args: list[object] = []
    if since_utc is not None:
        clauses.append("timestamp_start >= ?")
        args.append(since_utc)
    if until_utc is not None:
        clauses.append("timestamp_start < ?")
        args.append(until_utc)
    sql = (
        "SELECT id, COALESCE(duration, 0.0), transcript_excerpt, timestamp_start "
        "FROM broadcast_events WHERE " + " AND ".join(clauses) + " ORDER BY timestamp_start ASC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    rows = store.connection.execute(sql, args).fetchall()
    return [(int(r[0]), float(r[1] or 0.0), str(r[2] or ""), str(r[3])) for r in rows]


def _llm_brand(classifier: _TranscriptClassifier, transcript: str) -> str | None:
    """Return a canonical brand from the LLM, only for COMMERCIAL output."""
    result = classifier.classify_transcript(transcript)
    if getattr(result, "category", None) != "COMMERCIAL":
        return None
    return canonicalize_brand(getattr(result, "brand", None))


def backfill_unbranded_commercials(
    store: BroadcastStore,
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    dry_run: bool = True,
    classifier: _TranscriptClassifier | None = None,
    limit: int | None = None,
) -> BrandBackfillReport:
    """Attribute brands to unbranded commercial events.

    ``classifier`` enables the LLM tier; when ``None`` only the deterministic
    extractor runs. The LLM is only consulted for events the deterministic
    pass could not brand, and is safe to call during a dry run (read-only).
    """
    rows = _load_unbranded(store, since_utc=since_utc, until_utc=until_utc, limit=limit)
    resolver = CommercialIdentityResolver(store)

    items: list[BrandBackfillItem] = []
    deterministic_hits = 0
    llm_hits = 0
    llm_attempts = 0
    llm_errors = 0

    for event_id, duration, transcript, ts_start in rows:
        brand = extract_brand_from_text(transcript)
        source = "deterministic"
        if brand is not None:
            deterministic_hits += 1
        elif classifier is not None:
            llm_attempts += 1
            try:
                brand = _llm_brand(classifier, transcript)
            except Exception:  # noqa: BLE001 - any LLM/transport error is non-fatal
                llm_errors += 1
                brand = None
            if brand is not None:
                source = "llm"
                llm_hits += 1

        if brand is None:
            continue

        commercial_id: int | None = None
        if not dry_run:
            resolution = resolver.resolve(
                brand=brand,
                transcript=transcript,
                duration_seconds=duration,
                signature=None,
            )
            brand_id = resolution.brand_id
            if brand_id is None:
                # Duration out of the resolver's range: still attribute the
                # brand so the event leaves the unbranded bucket, just without
                # a stable commercial identity.
                brand_id = store.upsert_brand(brand)
            commercial_id = resolution.commercial_id
            store.connection.execute(
                "UPDATE broadcast_events "
                "SET brand_id = ?, brand_name = COALESCE(brand_name, ?), commercial_id = ? "
                "WHERE id = ?",
                (brand_id, brand, commercial_id, event_id),
            )
            store.insert_brand_mention(
                segment_id=event_id,
                brand_id=brand_id,
                mention_type="paid_ad",
                heard_utc=ts_start,
            )
            store.connection.commit()

        items.append(
            BrandBackfillItem(
                event_id=event_id,
                brand=brand,
                source=source,
                commercial_id=commercial_id,
                timestamp_start=ts_start,
            )
        )

    return BrandBackfillReport(
        items=items,
        events_scanned=len(rows),
        deterministic_hits=deterministic_hits,
        llm_hits=llm_hits,
        llm_attempts=llm_attempts,
        llm_errors=llm_errors,
        dry_run=dry_run,
    )
