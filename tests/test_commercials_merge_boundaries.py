"""Boundary merge: unbranded fragments absorbed into a branded neighbor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from radio_classifier.commercials import merge_boundary_commercials
from radio_classifier.commercials.identity import CommercialIdentityResolver
from radio_classifier.persistence import BroadcastStore
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _branded(store: BroadcastStore, *, start: datetime, seconds: int, brand: str, text: str) -> int:
    """Insert a branded commercial with a resolved identity."""
    resolver = CommercialIdentityResolver(store)
    res = resolver.resolve(brand=brand, transcript=text, duration_seconds=float(seconds), signature=None)
    brand_id = res.brand_id or store.upsert_brand(brand)
    return store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(start + timedelta(seconds=seconds)),
            category=BroadcastCategory.COMMERCIAL,
            brand_name=brand,
            brand_id=brand_id,
            commercial_id=res.commercial_id,
            transcript_excerpt=text,
        )
    )


def _unbranded(store: BroadcastStore, *, start: datetime, seconds: int, text: str) -> int:
    return store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(start + timedelta(seconds=seconds)),
            category=BroadcastCategory.COMMERCIAL,
            transcript_excerpt=text,
        )
    )


def test_overlapping_fragment_inherits_neighbor_brand_and_identity(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    full = "Now through June 15th get gig wifi for just 50 a month for five years no strings no commitment"
    overlap = "get gig wifi for just 50 a month for five years no strings no commitment included online in minutes"
    with BroadcastStore(tmp_path / "rc.db") as store:
        nbr = _branded(store, start=base, seconds=20, brand="Xfinity", text=full)
        frag = _unbranded(store, start=base + timedelta(seconds=20), seconds=10, text=overlap)

        report = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55)
        assert report.events_merged == 1
        item = report.items[0]
        assert item.event_id == frag
        assert item.brand == "Xfinity"
        assert item.neighbor_side == "prev"

        row = store.connection.execute(
            "SELECT brand_name, brand_id, commercial_id FROM broadcast_events WHERE id = ?",
            (frag,),
        ).fetchone()
        assert row[0] == "Xfinity"
        assert row[1] is not None
        # Inherits the neighbor's resolved identity.
        nbr_cid = store.connection.execute(
            "SELECT commercial_id FROM broadcast_events WHERE id = ?", (nbr,)
        ).fetchone()[0]
        assert row[2] == nbr_cid
        # paid_ad mention recorded.
        assert store.connection.execute(
            "SELECT mention_type FROM brand_mentions WHERE segment_id = ?", (frag,)
        ).fetchall() == [("paid_ad",)]


def test_low_similarity_straddle_is_rejected(tmp_path: Path) -> None:
    """A fragment that is a *different* ad than the branded neighbor stays unbranded."""
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    tmobile = "taxes and fees apply see T-Mobile dot com unlimited plan for the whole family"
    living_spaces = "get ready for summer with exciting styles during the Living Spaces Memorial Day event"
    with BroadcastStore(tmp_path / "rc.db") as store:
        _branded(store, start=base, seconds=20, brand="T-Mobile", text=tmobile)
        frag = _unbranded(store, start=base + timedelta(seconds=20), seconds=10, text=living_spaces)

        report = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55)
        assert report.events_merged == 0
        row = store.connection.execute(
            "SELECT brand_id, commercial_id FROM broadcast_events WHERE id = ?", (frag,)
        ).fetchone()
        assert row == (None, None)


def test_dry_run_does_not_mutate(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    full = "save big at the store this weekend only with your rewards card and free delivery"
    overlap = "save big at the store this weekend only with your rewards card and free delivery today"
    with BroadcastStore(tmp_path / "rc.db") as store:
        _branded(store, start=base, seconds=20, brand="Acme", text=full)
        frag = _unbranded(store, start=base + timedelta(seconds=20), seconds=10, text=overlap)

        report = merge_boundary_commercials(store, dry_run=True, min_similarity=0.55)
        assert report.events_merged == 1
        row = store.connection.execute(
            "SELECT brand_id, commercial_id, brand_name FROM broadcast_events WHERE id = ?", (frag,)
        ).fetchone()
        assert row == (None, None, None)


def test_no_branded_neighbor_leaves_fragment_alone(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    with BroadcastStore(tmp_path / "rc.db") as store:
        # Fragment flanked by SONGs only.
        song = store.upsert_song(artist="Nirvana", title="Lithium")
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(base),
                timestamp_end=_iso(base + timedelta(seconds=180)),
                category=BroadcastCategory.SONG,
                artist="Nirvana",
                track_title="Lithium",
                song_id=song,
            )
        )
        frag = _unbranded(
            store, start=base + timedelta(seconds=180), seconds=10, text="restrictions apply see store"
        )
        report = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55)
        assert report.events_merged == 0
        assert store.connection.execute(
            "SELECT brand_id FROM broadcast_events WHERE id = ?", (frag,)
        ).fetchone()[0] is None


def test_gap_guard_blocks_distant_neighbor(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    full = "save big at the store this weekend only with your rewards card and free delivery"
    overlap = "save big at the store this weekend only with your rewards card and free delivery today"
    with BroadcastStore(tmp_path / "rc.db") as store:
        _branded(store, start=base, seconds=20, brand="Acme", text=full)
        # Fragment starts 60s after the neighbor ends -> beyond max_gap.
        frag = _unbranded(store, start=base + timedelta(seconds=80), seconds=10, text=overlap)
        report = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55, max_gap_seconds=12.0)
        assert report.events_merged == 0
        assert store.connection.execute(
            "SELECT brand_id FROM broadcast_events WHERE id = ?", (frag,)
        ).fetchone()[0] is None


def test_idempotent_second_run_is_noop(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    full = "Now through June 15th get gig wifi for just 50 a month for five years no strings no commitment"
    overlap = "get gig wifi for just 50 a month for five years no strings no commitment included online in minutes"
    with BroadcastStore(tmp_path / "rc.db") as store:
        _branded(store, start=base, seconds=20, brand="Xfinity", text=full)
        _unbranded(store, start=base + timedelta(seconds=20), seconds=10, text=overlap)
        first = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55)
        assert first.events_merged == 1
        second = merge_boundary_commercials(store, dry_run=False, min_similarity=0.55)
        assert second.events_scanned == 0
        assert second.events_merged == 0
