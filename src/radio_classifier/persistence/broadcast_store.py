"""SQLite persistence (schema v2) for broadcast segments, brands, songs, commercials."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from radio_classifier.segments.reducer import duration_seconds
from radio_classifier.segments.types import SegmentTransition


_ARTIST_DISPLAY_ALIASES = {
    "linkin park": "Linkin Park",
}


def _display_key(value: str | None) -> str:
    return " ".join((value or "").strip().split()).casefold()


def _display_value(value: str | None, *, aliases: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return value
    if aliases:
        alias = aliases.get(_display_key(cleaned))
        if alias is not None:
            return alias
    return cleaned


def _has_lowercase_letter(value: str | None) -> bool:
    return any(ch.isalpha() and ch.islower() for ch in value or "")


def _prefer_display_value(
    existing: str | None,
    incoming: str | None,
    *,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Pick the least-noisy display value without changing song identity.

    ``upsert_song`` matches case-insensitively, so this helper only decides
    what text to keep on the canonical row. Explicit aliases fix known
    artifacts such as Shazam's ``LINKIN PARK``. Otherwise, a mixed-case
    incoming reference value can replace an all-caps existing value, while
    legitimate acronyms like ``AFI`` stay untouched unless an alias says
    otherwise.
    """

    existing_clean = _display_value(existing, aliases=aliases)
    incoming_clean = _display_value(incoming, aliases=aliases)
    if existing_clean is None or existing_clean == "":
        return incoming_clean
    if incoming_clean is None or incoming_clean == "":
        return existing_clean
    if existing_clean == incoming_clean:
        return existing_clean
    if _display_key(existing_clean) != _display_key(incoming_clean):
        return existing_clean
    if aliases and _display_key(existing_clean) in aliases:
        return incoming_clean
    if not _has_lowercase_letter(existing_clean) and _has_lowercase_letter(incoming_clean):
        return incoming_clean
    return existing_clean


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_schema_path() -> Path:
    return _repo_root() / "db" / "schema.sql"


def _ensure_database(db_path: Path, schema_path: Path) -> None:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    schema_sql = schema_path.read_text(encoding="utf-8")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


class BroadcastStore:
    """Single-writer SQLite store for schema v2."""

    def __init__(
        self,
        db_path: Path,
        *,
        schema_path: Path | None = None,
        use_wal: bool = True,
    ) -> None:
        self._db_path = Path(db_path).resolve()
        self._schema_path = schema_path or _default_schema_path()
        _ensure_database(self._db_path, self._schema_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if use_wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> BroadcastStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def apply_transition(self, t: SegmentTransition) -> int:
        """Insert one closed segment row. Returns ``broadcast_events.id``."""
        dur = duration_seconds(t.timestamp_start, t.timestamp_end)
        cur = self._conn.execute(
            """
            INSERT INTO broadcast_events (
                timestamp_start, timestamp_end, duration,
                category, song_id, commercial_id, brand_id,
                artist, track_title, brand_name,
                transcript_excerpt, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.timestamp_start,
                t.timestamp_end,
                dur,
                t.category.value,
                t.song_id,
                t.commercial_id,
                t.brand_id,
                t.artist,
                t.track_title,
                t.brand_name,
                t.transcript_excerpt,
                t.confidence,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ----------------------------------------------------------------- brands
    def upsert_brand(self, canonical_name: str, aliases: list[str] | None = None) -> int:
        """Insert brand if missing, return its id."""
        row = self._conn.execute(
            "SELECT id FROM brands WHERE canonical_name = ?",
            (canonical_name,),
        ).fetchone()
        if row is not None:
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO brands (canonical_name, aliases_json) VALUES (?, ?)",
            (canonical_name, json.dumps(aliases or [])),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def find_brand_by_name(self, canonical_name: str) -> int | None:
        row = self._conn.execute(
            "SELECT id FROM brands WHERE canonical_name = ?",
            (canonical_name,),
        ).fetchone()
        return int(row[0]) if row else None

    # ------------------------------------------------------------------ songs
    def upsert_song(
        self,
        *,
        artist: str | None,
        title: str | None,
        audfprint_track_id: str | None = None,
        source: str = "audfprint",
    ) -> int:
        """Insert-or-update a song row keyed on (case-folded artist, case-folded title).

        Prior to 2026-05-30 the lookup also included ``source``, which meant a
        Shazam discovery row and a later audfprint match for the *same* song
        coexisted as two rows. That, plus loose casing (e.g. "Bring Me to Life"
        vs "Bring Me To Life"), polluted every report.

        New behaviour:

        *  Look up by ``LOWER(TRIM(artist))`` / ``LOWER(TRIM(title))`` —
           source-agnostic and case-insensitive — so a Shazam row found later
           by audfprint resolves to the same row.
        *  If the existing row is missing ``audfprint_track_id`` and the new
           call supplies one, fill it in and upgrade the source to
           ``audfprint`` (a confirmed reference recording is the strongest
           signal we can record).
        *  Otherwise leave the existing row untouched and return its id.
        """

        display_artist = _display_value(artist, aliases=_ARTIST_DISPLAY_ALIASES)
        display_title = _display_value(title)
        norm_artist = _display_key(display_artist)
        norm_title = _display_key(display_title)
        existing = self._conn.execute(
            """
            SELECT id, audfprint_track_id, source, artist, title
            FROM songs
            WHERE LOWER(TRIM(COALESCE(artist, ''))) = ?
              AND LOWER(TRIM(COALESCE(title,  ''))) = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (norm_artist, norm_title),
        ).fetchone()
        if existing is not None:
            song_id = int(existing[0])
            existing_track_id = existing[1]
            existing_source = existing[2]
            existing_artist = existing[3]
            existing_title = existing[4]

            next_track_id = existing_track_id
            next_source = existing_source
            if existing_track_id is None and audfprint_track_id is not None:
                next_track_id = audfprint_track_id
                next_source = "audfprint"
            elif existing_source != source and source == "audfprint" and existing_source == "shazam":
                # No track id either way (caller didn't supply one and we
                # don't have one), but a deterministic audfprint match still
                # outranks the prior Shazam guess for "source".
                next_source = "audfprint"

            next_artist = _prefer_display_value(
                existing_artist,
                display_artist,
                aliases=_ARTIST_DISPLAY_ALIASES,
            )
            next_title = _prefer_display_value(existing_title, display_title)

            if (
                next_track_id != existing_track_id
                or next_source != existing_source
                or next_artist != existing_artist
                or next_title != existing_title
            ):
                self._conn.execute(
                    """
                    UPDATE songs
                       SET audfprint_track_id = ?,
                           source = ?,
                           artist = ?,
                           title = ?
                     WHERE id = ?
                    """,
                    (next_track_id, next_source, next_artist, next_title, song_id),
                )
                self._conn.commit()
            return song_id
        cur = self._conn.execute(
            """
            INSERT INTO songs (audfprint_track_id, artist, title, source)
            VALUES (?, ?, ?, ?)
            """,
            (audfprint_track_id, display_artist, display_title, source),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ------------------------------------------------------------ commercials
    def find_commercials_for_brand(
        self,
        brand_id: int,
        duration_bucket_seconds: int,
    ) -> list[tuple[int, str, str]]:
        """Return ``[(commercial_id, minhash_hex, reference_transcript)]`` candidates."""
        rows = self._conn.execute(
            """
            SELECT id, minhash_hex, reference_transcript
            FROM commercials
            WHERE brand_id = ? AND duration_bucket_seconds = ?
            """,
            (brand_id, duration_bucket_seconds),
        ).fetchall()
        return [(int(r[0]), str(r[1]), str(r[2])) for r in rows]

    def insert_commercial(
        self,
        *,
        brand_id: int,
        duration_bucket_seconds: int,
        minhash_hex: str,
        reference_transcript: str,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO commercials (
                brand_id, duration_bucket_seconds, minhash_hex,
                reference_transcript, play_count
            ) VALUES (?, ?, ?, ?, 1)
            """,
            (brand_id, duration_bucket_seconds, minhash_hex, reference_transcript),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def increment_commercial_play_count(self, commercial_id: int) -> None:
        self._conn.execute(
            "UPDATE commercials SET play_count = play_count + 1 WHERE id = ?",
            (commercial_id,),
        )
        self._conn.commit()

    # --------------------------------------------------------- brand_mentions
    def insert_brand_mention(
        self,
        *,
        segment_id: int,
        brand_id: int,
        mention_type: str,
        heard_utc: str,
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO brand_mentions (segment_id, brand_id, mention_type, heard_utc)
            VALUES (?, ?, ?, ?)
            """,
            (segment_id, brand_id, mention_type, heard_utc),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ------------------------------------------------------------ schema meta
    def schema_version(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'",
        ).fetchone()
        return str(row[0]) if row else None
