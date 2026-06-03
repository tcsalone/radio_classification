"""Absorb orphan unbranded commercial fragments into their branded neighbor.

The classifier emits fixed (often overlapping) analysis windows, so a single
ad routinely lands across consecutive ``broadcast_events`` rows. When the LLM
extracts the brand on one window but returns ``brand=null`` on the adjacent,
heavily-overlapping window, we get a *branded* commercial event next to an
*unbranded* fragment of the very same ad.

These don't fold via :mod:`.dedupe` because that requires matching brand
identity (the fragment's is empty). This pass closes that gap: an unbranded
COMMERCIAL event inherits the brand + identity of an adjacent branded
COMMERCIAL when their transcripts are similar enough to be the same ad.

The similarity gate is the safety mechanism. Measured on the 2026-06-02 run,
true same-ad overlaps cluster at similarity ≥ 0.55 (median 0.62), while
straddle/pod-boundary cases — where the fragment is actually a *different*
ad than the neighbor's label — fall in the low-similarity tail and are
correctly rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from radio_classifier.brands import canonicalize_brand
from radio_classifier.commercials.identity import CommercialIdentityResolver
from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.text import text_similarity

_DEFAULT_MIN_SIMILARITY = 0.55
# Gaps are ~0 in practice (contiguous windows), but guard future data where a
# station ID or song might sit between two ads.
_DEFAULT_MAX_GAP_SECONDS = 12.0


@dataclass
class BoundaryMergeItem:
    event_id: int
    brand: str
    neighbor_event_id: int
    neighbor_side: str  # 'prev' | 'next'
    similarity: float
    commercial_id: int | None


@dataclass
class BoundaryMergeReport:
    items: list[BoundaryMergeItem]
    events_scanned: int
    dry_run: bool

    @property
    def events_merged(self) -> int:
        return len(self.items)


def _parse_utc(value: str):
    from datetime import datetime, timezone

    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _gap_seconds(prev_end: str | None, next_start: str | None) -> float:
    if not prev_end or not next_start:
        return 0.0
    return max(0.0, (_parse_utc(next_start) - _parse_utc(prev_end)).total_seconds())


@dataclass
class _Event:
    rowid: int
    category: str
    start: str
    end: str | None
    duration: float
    commercial_id: int | None
    brand_id: int | None
    brand_name: str
    transcript: str


def _load_events(store: BroadcastStore, since_utc: str | None, until_utc: str | None) -> list[_Event]:
    clauses = ["1=1"]
    args: list[object] = []
    if since_utc is not None:
        clauses.append("timestamp_start >= ?")
        args.append(since_utc)
    if until_utc is not None:
        clauses.append("timestamp_start < ?")
        args.append(until_utc)
    rows = store.connection.execute(
        "SELECT id, category, timestamp_start, timestamp_end, COALESCE(duration,0.0), "
        "commercial_id, brand_id, COALESCE(brand_name,''), COALESCE(transcript_excerpt,'') "
        "FROM broadcast_events WHERE " + " AND ".join(clauses) + " ORDER BY timestamp_start ASC",
        args,
    ).fetchall()
    return [
        _Event(
            rowid=int(r[0]),
            category=str(r[1]),
            start=str(r[2]),
            end=r[3],
            duration=float(r[4] or 0.0),
            commercial_id=r[5],
            brand_id=r[6],
            brand_name=str(r[7] or ""),
            transcript=str(r[8] or ""),
        )
        for r in rows
    ]


def _is_branded_commercial(e: _Event) -> bool:
    return e.category == "COMMERCIAL" and (e.brand_id is not None or bool(e.brand_name.strip()))


def _is_unbranded_commercial(e: _Event) -> bool:
    return (
        e.category == "COMMERCIAL"
        and e.commercial_id is None
        and e.brand_id is None
        and not e.brand_name.strip()
    )


def _neighbor_brand(store: BroadcastStore, neighbor: _Event) -> str | None:
    """Resolve a neighbor's canonical brand from its name or brand_id row."""
    if neighbor.brand_name.strip():
        return canonicalize_brand(neighbor.brand_name)
    if neighbor.brand_id is not None:
        row = store.connection.execute(
            "SELECT canonical_name FROM brands WHERE id = ?", (neighbor.brand_id,)
        ).fetchone()
        if row and row[0]:
            return canonicalize_brand(str(row[0]))
    return None


def merge_boundary_commercials(
    store: BroadcastStore,
    *,
    since_utc: str | None = None,
    until_utc: str | None = None,
    min_similarity: float = _DEFAULT_MIN_SIMILARITY,
    max_gap_seconds: float = _DEFAULT_MAX_GAP_SECONDS,
    dry_run: bool = True,
) -> BoundaryMergeReport:
    """Attribute unbranded commercial fragments to an adjacent branded ad.

    For each unbranded COMMERCIAL event, the immediate previous/next events are
    considered; the branded-commercial neighbor with the highest transcript
    similarity is chosen. When that similarity meets ``min_similarity`` (and the
    gap is within ``max_gap_seconds``), the fragment inherits the neighbor's
    brand and commercial identity.
    """
    events = _load_events(store, since_utc, until_utc)
    resolver = CommercialIdentityResolver(store)
    items: list[BoundaryMergeItem] = []
    scanned = 0

    for i, ev in enumerate(events):
        if not _is_unbranded_commercial(ev):
            continue
        scanned += 1

        best: tuple[float, str, _Event] | None = None  # (sim, side, neighbor)
        for side, nidx in (("prev", i - 1), ("next", i + 1)):
            if not (0 <= nidx < len(events)):
                continue
            neighbor = events[nidx]
            if not _is_branded_commercial(neighbor):
                continue
            if side == "prev":
                gap = _gap_seconds(neighbor.end, ev.start)
            else:
                gap = _gap_seconds(ev.end, neighbor.start)
            if gap > max_gap_seconds:
                continue
            sim = text_similarity(ev.transcript, neighbor.transcript)
            if best is None or sim > best[0]:
                best = (sim, side, neighbor)

        if best is None or best[0] < min_similarity:
            continue
        sim, side, neighbor = best
        brand = _neighbor_brand(store, neighbor)
        if brand is None:
            continue

        commercial_id: int | None = neighbor.commercial_id
        if not dry_run:
            brand_id: int | None
            if neighbor.commercial_id is not None:
                # Same ad as the neighbor: attach to its existing identity and
                # bump the play count so the rollup stays accurate.
                commercial_id = neighbor.commercial_id
                brand_id = neighbor.brand_id if neighbor.brand_id is not None else store.upsert_brand(brand)
                store.increment_commercial_play_count(commercial_id)
            else:
                # Neighbor carries a brand but no resolved identity; resolve the
                # fragment's own identity under that brand.
                resolution = resolver.resolve(
                    brand=brand,
                    transcript=ev.transcript,
                    duration_seconds=ev.duration,
                    signature=None,
                )
                commercial_id = resolution.commercial_id
                brand_id = resolution.brand_id if resolution.brand_id is not None else store.upsert_brand(brand)

            store.connection.execute(
                "UPDATE broadcast_events "
                "SET brand_id = ?, brand_name = COALESCE(NULLIF(brand_name,''), ?), commercial_id = ? "
                "WHERE id = ?",
                (brand_id, brand, commercial_id, ev.rowid),
            )
            store.insert_brand_mention(
                segment_id=ev.rowid,
                brand_id=brand_id,
                mention_type="paid_ad",
                heard_utc=ev.start,
            )
            store.connection.commit()

        items.append(
            BoundaryMergeItem(
                event_id=ev.rowid,
                brand=brand,
                neighbor_event_id=neighbor.rowid,
                neighbor_side=side,
                similarity=round(sim, 3),
                commercial_id=commercial_id,
            )
        )

    return BoundaryMergeReport(items=items, events_scanned=scanned, dry_run=dry_run)
