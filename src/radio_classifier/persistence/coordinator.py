"""Feed :class:`SegmentInput` into the reducer and persist closed transitions.

This module is the bridge between :class:`SegmentReducer` (pure) and
:class:`BroadcastStore` (I/O). It also resolves transient ``brand_name``
strings into ``brand_id`` foreign keys at persistence time.
"""

from __future__ import annotations

from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.reducer import SegmentReducer
from radio_classifier.segments.types import SegmentInput, SegmentTransition


def _resolve_brand_id(store: BroadcastStore, brand_name: str | None) -> int | None:
    if brand_name is None or not brand_name.strip():
        return None
    return store.upsert_brand(brand_name.strip())


def persist_input(
    reducer: SegmentReducer,
    store: BroadcastStore | None,
    inp: SegmentInput | None,
) -> list[int]:
    """Apply one window's :class:`SegmentInput` to the reducer; persist closures.

    Returns the list of inserted ``broadcast_events.id`` values (may be empty
    when no segment closed).
    """
    if inp is None or store is None:
        # Reducer still advances even without a store, so callers can run
        # dry-run pipelines and inspect emitted transitions via reducer state.
        if inp is not None:
            reducer.feed(inp)
        return []
    new_ids: list[int] = []
    for t in reducer.feed(inp):
        new_ids.append(_persist_one(store, t))
    return new_ids


def persist_finalize(
    reducer: SegmentReducer,
    store: BroadcastStore | None,
    *,
    last_window_start_utc: str,
    window_seconds: float,
) -> list[int]:
    """Close any final open segment after processing all windows."""
    if store is None:
        return []
    new_ids: list[int] = []
    for t in reducer.finalize(last_window_start_utc, window_seconds):
        new_ids.append(_persist_one(store, t))
    return new_ids


def _persist_one(store: BroadcastStore, t: SegmentTransition) -> int:
    """Resolve foreign keys then insert."""
    if t.brand_id is None and t.brand_name:
        t.brand_id = _resolve_brand_id(store, t.brand_name)
    return store.apply_transition(t)
