"""Commercial identity dedupe helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from radio_classifier.commercials import dedupe_commercials
from radio_classifier.persistence import BroadcastStore
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _put_commercial_event(
    store: BroadcastStore,
    *,
    start: datetime,
    duration: int,
    brand_id: int,
    brand_name: str,
    commercial_id: int,
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


def test_dedupe_commercials_folds_brand_variants_with_similar_transcript(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "commercials.db")
    try:
        smart_amp = store.upsert_brand("Smart & Final")
        smart_and = store.upsert_brand("Smart and Final")
        c1 = store.insert_commercial(
            brand_id=smart_amp,
            duration_bucket_seconds=20,
            minhash_hex="aa",
            reference_transcript="Free breakfast is back at Smart and Final with bacon and eggs",
        )
        c2 = store.insert_commercial(
            brand_id=smart_and,
            duration_bucket_seconds=20,
            minhash_hex="bb",
            reference_transcript="Clip the Smart and Final digital coupon for free bacon and eggs",
        )
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        ev1 = _put_commercial_event(
            store,
            start=base,
            duration=20,
            brand_id=smart_amp,
            brand_name="Smart & Final",
            commercial_id=c1,
        )
        ev2 = _put_commercial_event(
            store,
            start=base + timedelta(minutes=5),
            duration=20,
            brand_id=smart_and,
            brand_name="Smart and Final",
            commercial_id=c2,
        )
        store.insert_brand_mention(
            segment_id=ev1,
            brand_id=smart_amp,
            mention_type="paid_ad",
            heard_utc=_iso(base),
        )
        store.insert_brand_mention(
            segment_id=ev2,
            brand_id=smart_and,
            mention_type="paid_ad",
            heard_utc=_iso(base + timedelta(minutes=5)),
        )

        report = dedupe_commercials(store)
        assert report.collapsed_pairs == 1
        assert report.events_repointed == 1
        assert report.rows_deleted == 1

        rows = store.connection.execute("SELECT id, brand_id FROM commercials ORDER BY id").fetchall()
        assert len(rows) == 1
        brand_name = store.connection.execute(
            "SELECT canonical_name FROM brands WHERE id = ?", (rows[0][1],)
        ).fetchone()[0]
        assert brand_name == "Smart and Final"
        event_commercial_ids = {
            r[0] for r in store.connection.execute("SELECT commercial_id FROM broadcast_events")
        }
        assert event_commercial_ids == {rows[0][0]}
    finally:
        store.close()


def test_dedupe_commercials_folds_adjacent_split_windows(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "commercials.db")
    try:
        toyota = store.upsert_brand("Toyota")
        c1 = store.insert_commercial(
            brand_id=toyota,
            duration_bucket_seconds=20,
            minhash_hex="11",
            reference_transcript="Toyota Memorial Day offers are extended on Corolla Hybrid and Camry",
        )
        c2 = store.insert_commercial(
            brand_id=toyota,
            duration_bucket_seconds=20,
            minhash_hex="22",
            reference_transcript="Save at Toyota on Corolla Hybrid Camry and electrified models",
        )
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        _put_commercial_event(
            store,
            start=base,
            duration=10,
            brand_id=toyota,
            brand_name="Toyota",
            commercial_id=c1,
        )
        _put_commercial_event(
            store,
            start=base + timedelta(seconds=10),
            duration=10,
            brand_id=toyota,
            brand_name="Toyota",
            commercial_id=c2,
        )

        report = dedupe_commercials(store)
        assert report.collapsed_pairs == 1
        assert report.events_repointed == 1
        assert store.connection.execute("SELECT COUNT(*) FROM commercials").fetchone()[0] == 1
    finally:
        store.close()


def test_dedupe_commercials_folds_split_across_classifier_window_gap(tmp_path: Path) -> None:
    """A single ad split across two ~10s classifier windows must fold.

    Real-world reproduction (from the 2026-05-31 16h run): a 30s Toyota ad
    landed in window N (10s body, transcript ending ``"...hurry in whi-"``)
    and window N+1 (20s body, transcript starting ``"in while there's still
    time to save..."``) with a ~9s wall-clock gap because the classifier
    emitted them as separate rows. The 2s adjacent-split gap used previously
    missed it; the new 10s default catches it.
    """
    store = BroadcastStore(tmp_path / "commercials.db")
    try:
        toyota = store.upsert_brand("Toyota")
        # The two transcripts share the window-overlap region (5s of audio
        # appears in both windows) so they retain the same anchor phrase.
        c1 = store.insert_commercial(
            brand_id=toyota,
            duration_bucket_seconds=10,
            minhash_hex="aa",
            reference_transcript=(
                "Toyota Memorial Day offers are extended hurry in while "
                "there is still time to save"
            ),
        )
        c2 = store.insert_commercial(
            brand_id=toyota,
            duration_bucket_seconds=20,
            minhash_hex="bb",
            reference_transcript=(
                "in while there is still time to save on Toyota Camry "
                "Corolla and electrified models"
            ),
        )
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=15)
        _put_commercial_event(
            store,
            start=base,
            duration=10,
            brand_id=toyota,
            brand_name="Toyota",
            commercial_id=c1,
        )
        _put_commercial_event(
            store,
            start=base + timedelta(seconds=19),
            duration=20,
            brand_id=toyota,
            brand_name="Toyota",
            commercial_id=c2,
        )

        report = dedupe_commercials(store)
        assert report.collapsed_pairs == 1
        assert report.rows_deleted == 1
    finally:
        store.close()


def test_dedupe_commercials_does_not_fold_distant_same_brand_events(tmp_path: Path) -> None:
    """Two ads for the same brand 60s apart with different transcripts stay
    separate. Guards against the new 10s adjacent gap silently merging
    distinct airings of similarly-worded ads.
    """
    store = BroadcastStore(tmp_path / "commercials.db")
    try:
        brand = store.upsert_brand("Toyota")
        c1 = store.insert_commercial(
            brand_id=brand,
            duration_bucket_seconds=20,
            minhash_hex="cc",
            reference_transcript="Test drive a Toyota Tacoma at your local dealer",
        )
        c2 = store.insert_commercial(
            brand_id=brand,
            duration_bucket_seconds=20,
            minhash_hex="dd",
            reference_transcript="Lease a Toyota Camry for two ninety nine a month",
        )
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        _put_commercial_event(
            store,
            start=base,
            duration=20,
            brand_id=brand,
            brand_name="Toyota",
            commercial_id=c1,
        )
        _put_commercial_event(
            store,
            start=base + timedelta(seconds=80),
            duration=20,
            brand_id=brand,
            brand_name="Toyota",
            commercial_id=c2,
        )

        report = dedupe_commercials(store)
        assert report.collapsed_pairs == 0
        assert store.connection.execute("SELECT COUNT(*) FROM commercials").fetchone()[0] == 2
    finally:
        store.close()


def test_dedupe_commercials_dry_run_is_pure(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "commercials.db")
    try:
        brand = store.upsert_brand("Xfinity")
        c1 = store.insert_commercial(
            brand_id=brand,
            duration_bucket_seconds=20,
            minhash_hex="33",
            reference_transcript="Xfinity summer savings event lock in your price",
        )
        c2 = store.insert_commercial(
            brand_id=brand,
            duration_bucket_seconds=20,
            minhash_hex="44",
            reference_transcript="Lock in your price during the Xfinity summer savings event",
        )
        base = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
        _put_commercial_event(
            store,
            start=base,
            duration=20,
            brand_id=brand,
            brand_name="Xfinity",
            commercial_id=c1,
        )
        _put_commercial_event(
            store,
            start=base + timedelta(minutes=5),
            duration=20,
            brand_id=brand,
            brand_name="Xfinity",
            commercial_id=c2,
        )

        report = dedupe_commercials(store, dry_run=True)
        assert report.collapsed_pairs == 1
        assert report.rows_deleted == 0
        assert store.connection.execute("SELECT COUNT(*) FROM commercials").fetchone()[0] == 2
    finally:
        store.close()
