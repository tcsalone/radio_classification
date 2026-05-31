"""Post-hoc cleanup for duplicate commercial identity rows."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from radio_classifier.brands import canonicalize_brand
from radio_classifier.persistence.broadcast_store import BroadcastStore


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass
class CommercialDedupeMember:
    commercial_id: int
    brand_id: int
    brand: str
    canonical_brand: str
    duration_bucket_seconds: int
    reference_transcript: str
    event_count: int


@dataclass
class CommercialDedupeGroup:
    """Commercial rows that should fold into one survivor."""

    reason: str
    members: list[CommercialDedupeMember]

    @property
    def survivor(self) -> CommercialDedupeMember:
        return self.members[0]

    @property
    def losers(self) -> list[CommercialDedupeMember]:
        return self.members[1:]


@dataclass
class CommercialDedupeReport:
    groups: list[CommercialDedupeGroup]
    events_repointed: int
    brand_mentions_repointed: int
    rows_deleted: int
    dry_run: bool

    @property
    def collapsed_pairs(self) -> int:
        return sum(len(g.losers) for g in self.groups)


def dedupe_commercials(
    store: BroadcastStore,
    *,
    dry_run: bool = False,
    transcript_similarity_threshold: float = 0.55,
    adjacent_similarity_threshold: float = 0.35,
    max_adjacent_gap_seconds: float = 2.0,
    max_adjacent_segment_seconds: float = 30.0,
    max_adjacent_combined_seconds: float = 90.0,
) -> CommercialDedupeReport:
    """Fold duplicate ``commercials`` rows and repoint dependent events.

    The grouping is intentionally conservative:

    * same canonical brand and same duration bucket with similar transcripts;
    * or adjacent short commercial events for the same canonical brand with at
      least modest transcript overlap, which catches one ad split across
      sequential 10-second classifier windows.
    """
    members = _load_members(store)
    if not members:
        return CommercialDedupeReport([], 0, 0, 0, dry_run)

    union = _UnionFind(members)
    _union_similar_transcripts(
        union,
        members,
        transcript_similarity_threshold=transcript_similarity_threshold,
    )
    _union_adjacent_splits(
        store,
        union,
        members,
        adjacent_similarity_threshold=adjacent_similarity_threshold,
        max_adjacent_gap_seconds=max_adjacent_gap_seconds,
        max_adjacent_segment_seconds=max_adjacent_segment_seconds,
        max_adjacent_combined_seconds=max_adjacent_combined_seconds,
    )

    groups = _build_groups(union)
    events_repointed = 0
    mentions_repointed = 0
    rows_deleted = 0
    if not dry_run and groups:
        conn = store.connection
        canonical_brand_ids = {
            group.survivor.canonical_brand: store.upsert_brand(group.survivor.canonical_brand)
            for group in groups
        }
        conn.execute("BEGIN")
        try:
            for group in groups:
                canonical_brand_id = canonical_brand_ids[group.survivor.canonical_brand]
                all_ids = [m.commercial_id for m in group.members]
                loser_ids = [m.commercial_id for m in group.losers]
                placeholders_all = ",".join("?" * len(all_ids))
                placeholders_losers = ",".join("?" * len(loser_ids))

                event_ids = [
                    int(r[0])
                    for r in conn.execute(
                        f"SELECT id FROM broadcast_events WHERE commercial_id IN ({placeholders_all})",
                        all_ids,
                    ).fetchall()
                ]

                conn.execute(
                    "UPDATE commercials SET brand_id = ? WHERE id = ?",
                    (canonical_brand_id, group.survivor.commercial_id),
                )
                cur = conn.execute(
                    f"""
                    UPDATE broadcast_events
                       SET commercial_id = ?,
                           brand_id = ?,
                           brand_name = COALESCE(brand_name, ?)
                     WHERE commercial_id IN ({placeholders_losers})
                    """,
                    [
                        group.survivor.commercial_id,
                        canonical_brand_id,
                        group.survivor.canonical_brand,
                        *loser_ids,
                    ],
                )
                events_repointed += cur.rowcount or 0
                conn.execute(
                    f"UPDATE broadcast_events SET brand_id = ? WHERE commercial_id = ?",
                    (canonical_brand_id, group.survivor.commercial_id),
                )
                if event_ids:
                    placeholders_events = ",".join("?" * len(event_ids))
                    cur = conn.execute(
                        f"""
                        UPDATE brand_mentions
                           SET brand_id = ?
                         WHERE mention_type = 'paid_ad'
                           AND segment_id IN ({placeholders_events})
                        """,
                        [canonical_brand_id, *event_ids],
                    )
                    mentions_repointed += cur.rowcount or 0
                cur = conn.execute(
                    f"DELETE FROM commercials WHERE id IN ({placeholders_losers})",
                    loser_ids,
                )
                rows_deleted += cur.rowcount or 0
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return CommercialDedupeReport(
        groups=groups,
        events_repointed=events_repointed,
        brand_mentions_repointed=mentions_repointed,
        rows_deleted=rows_deleted,
        dry_run=dry_run,
    )


def _load_members(store: BroadcastStore) -> dict[int, CommercialDedupeMember]:
    rows = store.connection.execute(
        """
        SELECT
            c.id,
            c.brand_id,
            b.canonical_name,
            c.duration_bucket_seconds,
            c.reference_transcript,
            (SELECT COUNT(*) FROM broadcast_events e WHERE e.commercial_id = c.id) AS event_count
        FROM commercials c
        JOIN brands b ON b.id = c.brand_id
        ORDER BY c.id ASC
        """
    ).fetchall()
    out: dict[int, CommercialDedupeMember] = {}
    for r in rows:
        brand = str(r[2] or "")
        canonical = canonicalize_brand(brand) or brand
        commercial_id = int(r[0])
        out[commercial_id] = CommercialDedupeMember(
            commercial_id=commercial_id,
            brand_id=int(r[1]),
            brand=brand,
            canonical_brand=canonical,
            duration_bucket_seconds=int(r[3]),
            reference_transcript=str(r[4] or ""),
            event_count=int(r[5] or 0),
        )
    return out


def _union_similar_transcripts(
    union: _UnionFind,
    members: dict[int, CommercialDedupeMember],
    *,
    transcript_similarity_threshold: float,
) -> None:
    by_key: dict[tuple[str, int], list[CommercialDedupeMember]] = {}
    for member in members.values():
        by_key.setdefault((member.canonical_brand, member.duration_bucket_seconds), []).append(member)

    for group in by_key.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                similarity = _text_similarity(left.reference_transcript, right.reference_transcript)
                if similarity >= transcript_similarity_threshold:
                    union.union(left.commercial_id, right.commercial_id, "similar-transcript")


def _union_adjacent_splits(
    store: BroadcastStore,
    union: _UnionFind,
    members: dict[int, CommercialDedupeMember],
    *,
    adjacent_similarity_threshold: float,
    max_adjacent_gap_seconds: float,
    max_adjacent_segment_seconds: float,
    max_adjacent_combined_seconds: float,
) -> None:
    rows = store.connection.execute(
        """
        SELECT
            id,
            timestamp_start,
            timestamp_end,
            duration,
            commercial_id
        FROM broadcast_events
        WHERE category = 'COMMERCIAL'
          AND commercial_id IS NOT NULL
        ORDER BY timestamp_start ASC
        """
    ).fetchall()
    for prev, cur in zip(rows, rows[1:]):
        prev_id = int(prev[4])
        cur_id = int(cur[4])
        if prev_id == cur_id or prev_id not in members or cur_id not in members:
            continue
        prev_member = members[prev_id]
        cur_member = members[cur_id]
        if prev_member.canonical_brand != cur_member.canonical_brand:
            continue
        prev_duration = float(prev[3] or 0.0)
        cur_duration = float(cur[3] or 0.0)
        if prev_duration > max_adjacent_segment_seconds or cur_duration > max_adjacent_segment_seconds:
            continue
        if prev_duration + cur_duration > max_adjacent_combined_seconds:
            continue
        if _gap_seconds(prev[2], cur[1]) > max_adjacent_gap_seconds:
            continue
        similarity = _text_similarity(prev_member.reference_transcript, cur_member.reference_transcript)
        if similarity >= adjacent_similarity_threshold:
            union.union(prev_id, cur_id, "adjacent-split")


def _build_groups(union: _UnionFind) -> list[CommercialDedupeGroup]:
    by_root: dict[int, list[CommercialDedupeMember]] = {}
    reasons: dict[int, set[str]] = {}
    for commercial_id, member in union.members.items():
        root = union.find(commercial_id)
        by_root.setdefault(root, []).append(member)
    for child, reason in union.reasons.items():
        reasons.setdefault(union.find(child), set()).add(reason)

    groups: list[CommercialDedupeGroup] = []
    for root, members in by_root.items():
        if len(members) < 2:
            continue
        ranked = sorted(
            members,
            key=lambda m: (-m.event_count, m.canonical_brand.casefold(), m.commercial_id),
        )
        groups.append(
            CommercialDedupeGroup(
                reason="+".join(sorted(reasons.get(root, {"unknown"}))),
                members=ranked,
            )
        )
    groups.sort(key=lambda g: g.survivor.commercial_id)
    return groups


def _tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _WORD_RE.findall(text or "")]


def _text_similarity(a: str, b: str) -> float:
    """Combined token Jaccard / cosine similarity for ASR transcript text."""
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    jaccard = len(a_set & b_set) / len(a_set | b_set)
    cosine = _cosine(a_tokens, b_tokens)
    return max(jaccard, cosine)


def _cosine(a: Iterable[str], b: Iterable[str]) -> float:
    from collections import Counter

    ca = Counter(a)
    cb = Counter(b)
    shared = set(ca) & set(cb)
    num = sum(ca[t] * cb[t] for t in shared)
    da = math.sqrt(sum(v * v for v in ca.values()))
    db = math.sqrt(sum(v * v for v in cb.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _gap_seconds(prev_end: str | None, next_start: str | None) -> float:
    if not prev_end or not next_start:
        return 0.0
    end_dt = _parse_utc(prev_end)
    start_dt = _parse_utc(next_start)
    return max(0.0, (start_dt - end_dt).total_seconds())


def _parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class _UnionFind:
    def __init__(self, members: dict[int, CommercialDedupeMember]) -> None:
        self.members = members
        self.parent = {commercial_id: commercial_id for commercial_id in members}
        self.reasons: dict[int, str] = {}

    def find(self, commercial_id: int) -> int:
        parent = self.parent[commercial_id]
        if parent != commercial_id:
            self.parent[commercial_id] = self.find(parent)
        return self.parent[commercial_id]

    def union(self, left: int, right: int, reason: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            self.reasons[right] = reason
            return
        root = min(root_left, root_right)
        child = max(root_left, root_right)
        self.parent[child] = root
        self.reasons[child] = reason
