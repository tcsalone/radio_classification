"""Merge duplicate ``brands`` identity rows that are the same advertiser.

Over a long collection run the ``brands`` table accumulates near-duplicate rows
for a single advertiser:

* **Case collisions** — ``GEICO``/``Geico``, ``IKEA``/``Ikea``,
  ``ZipRecruiter``/``Ziprecruiter``. ``upsert_brand`` keys on the *exact*
  ``canonical_name`` (``UNIQUE (canonical_name)``), so a casing drift in the
  LLM-derived brand field creates a brand-new row + ``brand_id``.
* **Known ASR/LLM spelling variants** that :func:`canonicalize_brand` already
  folds to one canonical name (e.g. ``GreatOn.com`` → ``Graton Resort and
  Casino``) but which slipped into the table as separate rows before the alias
  table caught up.

Every report groups by ``brands.canonical_name`` (joined through ``brand_id``),
so each duplicate row silently splits that advertiser's airings / mentions
across two rankings. This module folds each such group down to a single
survivor ``brand_id`` — re-pointing ``broadcast_events``, ``commercials`` and
``brand_mentions``, merging ``aliases_json``, then deleting the loser rows.

The fold key is deliberately conservative: two brands only merge when
``normalize_token(canonicalize_brand(name))`` collides. That is case + the
evidence-driven alias table only — no fuzzy matching — so unrelated advertisers
are never merged.

Mirrors the shape of :mod:`radio_classifier.commercials.dedupe`: pure grouping,
a single transaction guarded by ``dry_run``, and a report dataclass the CLI
prints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from radio_classifier.brands import canonicalize_brand
from radio_classifier.persistence.broadcast_store import BroadcastStore
from radio_classifier.segments.normalize import normalize_token


@dataclass
class BrandMergeMember:
    """One ``brands`` row inside a merge group."""

    brand_id: int
    canonical_name: str
    aliases: list[str]
    event_count: int
    mention_count: int
    commercial_count: int

    @property
    def total_refs(self) -> int:
        return self.event_count + self.mention_count + self.commercial_count


@dataclass
class BrandMergeGroup:
    """Same-advertiser ``brands`` rows that fold into one survivor.

    ``members`` is ranked: the **first** member is the survivor whose
    ``brand_id`` is kept (the row with the most existing references, so the
    fewest rows have to be re-pointed). ``display_name`` is the spelling the
    survivor row is renamed to — the authoritative alias-table canonical when
    the group is a known-variant fold, otherwise the "proper-noun" spelling
    (most uppercase letters) among the case variants.
    """

    key: str
    display_name: str
    members: list[BrandMergeMember]

    @property
    def survivor(self) -> BrandMergeMember:
        return self.members[0]

    @property
    def losers(self) -> list[BrandMergeMember]:
        return self.members[1:]


@dataclass
class BrandMergeReport:
    """Outcome of one :func:`merge_brands` invocation."""

    groups: list[BrandMergeGroup]
    events_repointed: int = 0
    mentions_repointed: int = 0
    commercials_repointed: int = 0
    commercials_merged: int = 0
    brands_deleted: int = 0
    dry_run: bool = True

    @property
    def collapsed_pairs(self) -> int:
        """How many ``brands`` rows would disappear (sum of group sizes - 1)."""
        return sum(len(g.losers) for g in self.groups)


def _fold_key(name: str) -> str:
    """Case-insensitive + alias-folded key for a brand name.

    ``canonicalize_brand`` collapses whitespace, folds ``" & "`` → ``" and "``
    and applies the evidence-driven alias table; ``normalize_token`` then
    case-folds and trims stray punctuation. The result is the identity two
    brand rows must share to be considered the same advertiser.
    """
    canonical = canonicalize_brand(name) or name
    return normalize_token(canonical) or normalize_token(name) or name.strip().casefold()


def _uppercase_count(name: str) -> int:
    return sum(1 for ch in name if ch.isupper())


def _load_members(store: BroadcastStore) -> list[BrandMergeMember]:
    rows = store.connection.execute(
        """
        SELECT
            b.id,
            b.canonical_name,
            b.aliases_json,
            (SELECT COUNT(*) FROM broadcast_events e WHERE e.brand_id = b.id) AS event_count,
            (SELECT COUNT(*) FROM brand_mentions m WHERE m.brand_id = b.id) AS mention_count,
            (SELECT COUNT(*) FROM commercials c WHERE c.brand_id = b.id) AS commercial_count
        FROM brands b
        ORDER BY b.id ASC
        """
    ).fetchall()
    members: list[BrandMergeMember] = []
    for r in rows:
        try:
            aliases = list(json.loads(r[2] or "[]"))
        except (TypeError, ValueError):
            aliases = []
        members.append(
            BrandMergeMember(
                brand_id=int(r[0]),
                canonical_name=str(r[1] or ""),
                aliases=[str(a) for a in aliases],
                event_count=int(r[3] or 0),
                mention_count=int(r[4] or 0),
                commercial_count=int(r[5] or 0),
            )
        )
    return members


def _pick_display_name(key: str, members: list[BrandMergeMember]) -> str:
    """Choose the canonical spelling the survivor row is renamed to.

    Priority:

    1.  A member whose name maps to an authoritative alias-table canonical
        (``canonicalize_brand(name)`` differs from the raw name) — emit that
        canonical target. This is how ``GreatOn.com`` folds to
        ``Graton Resort and Casino`` even if no member row literally stored the
        proper spelling.
    2.  Otherwise the spelling with the most uppercase letters (proper-noun
        forms like ``GEICO``/``IKEA``/``ZipRecruiter`` beat ``Geico``/``Ikea``/
        ``Ziprecruiter``), tie-broken by most references then lowest id.
    """
    for m in members:
        canonical = canonicalize_brand(m.canonical_name)
        if (
            canonical
            and (normalize_token(canonical) or "") == key
            and normalize_token(canonical) != normalize_token(m.canonical_name)
        ):
            return canonical
    best = max(
        members,
        key=lambda m: (_uppercase_count(m.canonical_name), m.total_refs, -m.brand_id),
    )
    return best.canonical_name


def _iter_groups(store: BroadcastStore) -> list[BrandMergeGroup]:
    buckets: dict[str, list[BrandMergeMember]] = {}
    for member in _load_members(store):
        buckets.setdefault(_fold_key(member.canonical_name), []).append(member)

    groups: list[BrandMergeGroup] = []
    for key, members in buckets.items():
        if len(members) < 2:
            continue
        ranked = sorted(
            members,
            key=lambda m: (-m.total_refs, m.brand_id),
        )
        groups.append(
            BrandMergeGroup(
                key=key,
                display_name=_pick_display_name(key, members),
                members=ranked,
            )
        )
    groups.sort(key=lambda g: g.survivor.brand_id)
    return groups


def merge_brands(store: BroadcastStore, *, dry_run: bool = False) -> BrandMergeReport:
    """Fold duplicate ``brands`` rows for the same advertiser.

    Transactional order per group:

    1.  Re-point ``broadcast_events.brand_id`` from every loser to the survivor
        and set ``brand_name`` to the chosen display spelling.
    2.  Re-point ``brand_mentions.brand_id`` from losers to the survivor.
    3.  Re-point ``commercials.brand_id`` from losers to the survivor. The
        ``commercials`` table has ``UNIQUE (brand_id, duration_bucket_seconds,
        minhash_hex)``: when a loser commercial would collide with an identical
        survivor commercial, its ``play_count`` and dependent
        ``broadcast_events.commercial_id`` are folded into the survivor
        commercial and the loser commercial row is deleted instead.
    4.  Delete the loser ``brands`` rows.
    5.  Rename the survivor to ``display_name`` and merge ``aliases_json``
        (survivor + loser names + loser aliases). Done last so the
        ``UNIQUE (canonical_name)`` constraint can't collide with a loser that
        still holds that spelling.

    Pass ``dry_run=True`` to compute the report without writing.
    """
    groups = _iter_groups(store)
    report = BrandMergeReport(groups=groups, dry_run=dry_run)
    if dry_run or not groups:
        return report

    conn = store.connection
    conn.execute("BEGIN")
    try:
        for group in groups:
            survivor = group.survivor
            loser_ids = [m.brand_id for m in group.losers]
            placeholders = ",".join("?" * len(loser_ids))

            # 1. broadcast_events -> survivor, normalize brand_name text.
            cur = conn.execute(
                f"UPDATE broadcast_events SET brand_id = ?, brand_name = ? "
                f"WHERE brand_id IN ({placeholders})",
                [survivor.brand_id, group.display_name, *loser_ids],
            )
            report.events_repointed += cur.rowcount or 0
            conn.execute(
                "UPDATE broadcast_events SET brand_name = ? WHERE brand_id = ?",
                (group.display_name, survivor.brand_id),
            )

            # 2. brand_mentions -> survivor.
            cur = conn.execute(
                f"UPDATE brand_mentions SET brand_id = ? WHERE brand_id IN ({placeholders})",
                [survivor.brand_id, *loser_ids],
            )
            report.mentions_repointed += cur.rowcount or 0

            # 3. commercials -> survivor, honoring the UNIQUE identity constraint.
            # Map (duration_bucket, minhash) -> survivor commercial id so a loser
            # commercial that duplicates an existing survivor ad is folded, not
            # re-pointed into a UNIQUE violation.
            survivor_keys: dict[tuple[int, str], int] = {}
            for r in conn.execute(
                "SELECT id, duration_bucket_seconds, minhash_hex "
                "FROM commercials WHERE brand_id = ?",
                (survivor.brand_id,),
            ).fetchall():
                survivor_keys[(int(r[1]), str(r[2]))] = int(r[0])

            for r in conn.execute(
                f"SELECT id, duration_bucket_seconds, minhash_hex, play_count "
                f"FROM commercials WHERE brand_id IN ({placeholders})",
                loser_ids,
            ).fetchall():
                loser_cid = int(r[0])
                ckey = (int(r[1]), str(r[2]))
                loser_plays = int(r[3] or 0)
                twin = survivor_keys.get(ckey)
                if twin is not None:
                    # Identical ad already on the survivor: fold play_count +
                    # dependent events, drop the loser commercial row.
                    conn.execute(
                        "UPDATE commercials SET play_count = play_count + ? WHERE id = ?",
                        (loser_plays, twin),
                    )
                    conn.execute(
                        "UPDATE broadcast_events SET commercial_id = ? WHERE commercial_id = ?",
                        (twin, loser_cid),
                    )
                    conn.execute("DELETE FROM commercials WHERE id = ?", (loser_cid,))
                    report.commercials_merged += 1
                else:
                    conn.execute(
                        "UPDATE commercials SET brand_id = ? WHERE id = ?",
                        (survivor.brand_id, loser_cid),
                    )
                    survivor_keys[ckey] = loser_cid
                    report.commercials_repointed += 1

            # 4. delete loser brand rows (now unreferenced).
            cur = conn.execute(
                f"DELETE FROM brands WHERE id IN ({placeholders})",
                loser_ids,
            )
            report.brands_deleted += cur.rowcount or 0

            # 5. rename survivor + merge aliases (last, to dodge UNIQUE collisions).
            merged_aliases = _merge_aliases(group)
            conn.execute(
                "UPDATE brands SET canonical_name = ?, aliases_json = ? WHERE id = ?",
                (group.display_name, json.dumps(merged_aliases), survivor.brand_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return report


def _merge_aliases(group: BrandMergeGroup) -> list[str]:
    """Union of every member's exact spelling + aliases, minus the display name.

    De-duplicated by exact string (not normalized token) so provenance spellings
    that differ only by case — ``Geico`` under a surviving ``GEICO`` — are kept
    as aliases rather than silently dropped.
    """
    seen: set[str] = set()
    for m in group.members:
        for value in [m.canonical_name, *m.aliases]:
            if value and value != group.display_name:
                seen.add(value)
    return sorted(seen)
