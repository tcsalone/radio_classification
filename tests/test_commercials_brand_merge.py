"""Duplicate-brand identity merge (case + alias variants)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from radio_classifier.commercials import merge_brands
from radio_classifier.persistence import BroadcastStore
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _put_commercial_event(
    store: BroadcastStore,
    *,
    start: datetime,
    brand_id: int,
    brand_name: str,
    commercial_id: int | None = None,
    duration: int = 20,
) -> int:
    return store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(start + timedelta(seconds=duration)),
            category=BroadcastCategory.COMMERCIAL,
            brand_id=brand_id,
            brand_name=brand_name,
            commercial_id=commercial_id,
        )
    )


def test_merge_brands_folds_case_variant_and_keeps_proper_noun_spelling(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "brands.db")
    try:
        upper = store.upsert_brand("GEICO")
        lower = store.upsert_brand("Geico")
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=30)

        # More references on the lowercase row so the survivor pick (most refs)
        # and the display pick (proper-noun casing) diverge — display must still
        # be "GEICO".
        ev_upper = _put_commercial_event(store, start=base, brand_id=upper, brand_name="GEICO")
        ev_lower1 = _put_commercial_event(
            store, start=base + timedelta(minutes=5), brand_id=lower, brand_name="Geico"
        )
        ev_lower2 = _put_commercial_event(
            store, start=base + timedelta(minutes=10), brand_id=lower, brand_name="Geico"
        )
        for ev, bid in ((ev_upper, upper), (ev_lower1, lower), (ev_lower2, lower)):
            store.insert_brand_mention(
                segment_id=ev, brand_id=bid, mention_type="paid_ad", heard_utc=_iso(base)
            )

        # dry-run changes nothing.
        preview = merge_brands(store, dry_run=True)
        assert preview.collapsed_pairs == 1
        assert preview.brands_deleted == 0
        assert (
            store.connection.execute("SELECT COUNT(*) FROM brands").fetchone()[0] == 2
        )

        report = merge_brands(store)
        assert report.collapsed_pairs == 1
        assert report.brands_deleted == 1
        # survivor is the most-referenced row ("Geico", 2 refs); only the single
        # "GEICO"-id event/mention has to be re-pointed onto it.
        assert report.events_repointed == 1
        assert report.mentions_repointed == 1

        brands = store.connection.execute(
            "SELECT id, canonical_name FROM brands"
        ).fetchall()
        assert len(brands) == 1
        surviving_id, surviving_name = int(brands[0][0]), brands[0][1]
        assert surviving_name == "GEICO"
        # survivor is the most-referenced row (lowercase had 2 vs 1).
        assert surviving_id == lower

        # every event + mention now points at the survivor and reads "GEICO".
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM broadcast_events WHERE brand_id = ?", (surviving_id,)
            ).fetchone()[0]
            == 3
        )
        assert (
            store.connection.execute(
                "SELECT COUNT(DISTINCT brand_name) FROM broadcast_events"
            ).fetchone()[0]
            == 1
        )
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM brand_mentions WHERE brand_id = ?", (surviving_id,)
            ).fetchone()[0]
            == 3
        )
        # loser spelling preserved as an alias.
        aliases = store.connection.execute(
            "SELECT aliases_json FROM brands WHERE id = ?", (surviving_id,)
        ).fetchone()[0]
        assert "Geico" in aliases
    finally:
        store.close()


def test_merge_brands_folds_known_alias_to_canonical(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "brands.db")
    try:
        proper = store.upsert_brand("Graton Resort and Casino")
        variant = store.upsert_brand("GreatOn.com")  # alias-table variant
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        _put_commercial_event(store, start=base, brand_id=proper, brand_name="Graton Resort and Casino")
        _put_commercial_event(
            store, start=base + timedelta(minutes=5), brand_id=variant, brand_name="GreatOn.com"
        )

        report = merge_brands(store)
        assert report.collapsed_pairs == 1
        names = [r[0] for r in store.connection.execute("SELECT canonical_name FROM brands")]
        assert names == ["Graton Resort and Casino"]
    finally:
        store.close()


def test_merge_brands_folds_duplicate_commercial_playcounts(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "brands.db")
    try:
        upper = store.upsert_brand("IKEA")
        lower = store.upsert_brand("Ikea")
        # Identical ad (same duration bucket + minhash) recorded under both
        # brand rows -> must fold into one commercial with summed play_count,
        # not trip the UNIQUE (brand_id, duration_bucket, minhash) constraint.
        c_upper = store.insert_commercial(
            brand_id=upper,
            duration_bucket_seconds=30,
            minhash_hex="deadbeef",
            reference_transcript="Assemble happiness at IKEA this weekend",
        )
        c_lower = store.insert_commercial(
            brand_id=lower,
            duration_bucket_seconds=30,
            minhash_hex="deadbeef",
            reference_transcript="Assemble happiness at IKEA this weekend",
        )
        store.connection.execute(
            "UPDATE commercials SET play_count = 3 WHERE id = ?", (c_upper,)
        )
        store.connection.execute(
            "UPDATE commercials SET play_count = 4 WHERE id = ?", (c_lower,)
        )
        store.connection.commit()

        report = merge_brands(store)
        assert report.collapsed_pairs == 1
        assert report.commercials_merged == 1

        rows = store.connection.execute(
            "SELECT brand_id, play_count FROM commercials"
        ).fetchall()
        assert len(rows) == 1  # duplicate ad folded away
        assert int(rows[0][1]) == 7  # 3 + 4 play counts summed
        brand_rows = store.connection.execute("SELECT COUNT(*) FROM brands").fetchone()[0]
        assert brand_rows == 1
    finally:
        store.close()


def test_merge_brands_leaves_distinct_advertisers_alone(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "brands.db")
    try:
        store.upsert_brand("Xfinity")
        store.upsert_brand("Pepsi")
        report = merge_brands(store, dry_run=True)
        assert report.groups == []
        assert report.collapsed_pairs == 0
    finally:
        store.close()
