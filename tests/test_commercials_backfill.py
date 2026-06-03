"""Brand backfill for unbranded commercial events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from radio_classifier.commercials import backfill_unbranded_commercials
from radio_classifier.persistence import BroadcastStore
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _add_unbranded_commercial(store: BroadcastStore, *, start: datetime, seconds: int, text: str) -> int:
    return store.apply_transition(
        SegmentTransition(
            timestamp_start=_iso(start),
            timestamp_end=_iso(start + timedelta(seconds=seconds)),
            category=BroadcastCategory.COMMERCIAL,
            transcript_excerpt=text,
        )
    )


@dataclass
class _FakeLlmResult:
    category: str
    brand: str | None


class _FakeClassifier:
    """Returns a canned result per transcript; records calls."""

    def __init__(self, mapping: dict[str, _FakeLlmResult]) -> None:
        self.mapping = mapping
        self.calls: list[str] = []

    def classify_transcript(self, text: str) -> _FakeLlmResult:
        self.calls.append(text)
        return self.mapping.get(text, _FakeLlmResult(category="DJ", brand=None))


def test_backfill_dry_run_does_not_mutate(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    with BroadcastStore(tmp_path / "rc.db") as store:
        eid = _add_unbranded_commercial(
            store, start=base, seconds=20, text="apply now at billshappen.com today"
        )
        report = backfill_unbranded_commercials(store, dry_run=True)
        assert report.events_branded == 1
        assert report.items[0].brand == "BillsHappen.com"
        assert report.items[0].commercial_id is None  # dry run resolves nothing
        # DB untouched.
        row = store.connection.execute(
            "SELECT brand_id, commercial_id, brand_name FROM broadcast_events WHERE id = ?",
            (eid,),
        ).fetchone()
        assert row == (None, None, None)


def test_backfill_apply_brands_and_repoints(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    with BroadcastStore(tmp_path / "rc.db") as store:
        eid = _add_unbranded_commercial(
            store, start=base, seconds=20, text="apply now at billshappen.com today"
        )
        report = backfill_unbranded_commercials(store, dry_run=False)
        assert report.events_branded == 1
        assert report.deterministic_hits == 1

        row = store.connection.execute(
            "SELECT brand_id, commercial_id, brand_name FROM broadcast_events WHERE id = ?",
            (eid,),
        ).fetchone()
        assert row[0] is not None  # brand_id set
        assert row[1] is not None  # commercial_id set (20s is in resolver range)
        assert row[2] == "BillsHappen.com"

        # A paid_ad brand mention was recorded.
        mentions = store.connection.execute(
            "SELECT mention_type FROM brand_mentions WHERE segment_id = ?", (eid,)
        ).fetchall()
        assert mentions == [("paid_ad",)]


def test_backfill_llm_tier_used_when_deterministic_misses(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    spoken = "Pet safe and weed deadly, find it wherever weed killers are sold."
    classifier = _FakeClassifier({spoken: _FakeLlmResult(category="COMMERCIAL", brand="Spruce")})
    with BroadcastStore(tmp_path / "rc.db") as store:
        eid = _add_unbranded_commercial(store, start=base, seconds=20, text=spoken)
        report = backfill_unbranded_commercials(store, dry_run=False, classifier=classifier)
        assert report.events_branded == 1
        assert report.llm_hits == 1
        assert report.deterministic_hits == 0
        assert classifier.calls == [spoken]
        brand_name = store.connection.execute(
            "SELECT brand_name FROM broadcast_events WHERE id = ?", (eid,)
        ).fetchone()[0]
        assert brand_name == "Spruce"


def test_backfill_skips_llm_when_deterministic_hits(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    text = "apply now at billshappen.com today"
    classifier = _FakeClassifier({})
    with BroadcastStore(tmp_path / "rc.db") as store:
        _add_unbranded_commercial(store, start=base, seconds=20, text=text)
        report = backfill_unbranded_commercials(store, dry_run=False, classifier=classifier)
        assert report.deterministic_hits == 1
        assert classifier.calls == []  # never consulted the LLM


def test_backfill_leaves_boilerplate_unbranded(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    text = "Restrictions apply. Must be 18 or older. See store for details."
    with BroadcastStore(tmp_path / "rc.db") as store:
        eid = _add_unbranded_commercial(store, start=base, seconds=20, text=text)
        report = backfill_unbranded_commercials(store, dry_run=False)
        assert report.events_branded == 0
        row = store.connection.execute(
            "SELECT brand_id, commercial_id FROM broadcast_events WHERE id = ?", (eid,)
        ).fetchone()
        assert row == (None, None)


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    base = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    with BroadcastStore(tmp_path / "rc.db") as store:
        _add_unbranded_commercial(
            store, start=base, seconds=20, text="apply now at billshappen.com today"
        )
        first = backfill_unbranded_commercials(store, dry_run=False)
        assert first.events_branded == 1
        # Re-run: nothing left unbranded, so nothing to do.
        second = backfill_unbranded_commercials(store, dry_run=False)
        assert second.events_scanned == 0
        assert second.events_branded == 0
