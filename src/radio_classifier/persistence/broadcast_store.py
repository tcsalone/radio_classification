"""SQLite persistence for broadcast segments, brands, songs, commercials, and runs."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

from radio_classifier.segments.reducer import duration_seconds
from radio_classifier.segments.types import SegmentTransition


_ARTIST_DISPLAY_ALIASES = {
    "linkin park": "Linkin Park",
}


# Unicode apostrophe variants we routinely see on the wire. Shazam returns
# the typographic right single quote U+2019 ("Picking Dragons’ Pockets")
# while curated tracklists and Whisper outputs tend toward the ASCII form
# ("Picking Dragons' Pockets"). U+02BC (modifier letter apostrophe) shows
# up in some international band names, and U+2018 / U+201B for completeness.
_TYPOGRAPHIC_APOSTROPHES = "\u2018\u2019\u02BC\u201B"

# Wrapping/sentence punctuation that the two detectors disagree on for the same
# recording. Shazam returns "Bedroom Posters (feat. Good Charlotte)" while the
# audfprint reference filename surfaces "Bedroom Posters _feat. Good Charlotte".
# After dropping apostrophes/underscores the remaining difference is the parens
# and the period, so neutralize those too — without removing the feature words,
# which keeps a genuine "Song" distinct from "Song (feat. X)".
_SONG_TITLE_PUNCT_RE = re.compile(r"[()\[\]{}.,]+")


def _display_key(value: str | None) -> str:
    """Normalize an artist/title field for case-insensitive identity matching.

    Folds three classes of differences that have collapsed real song
    identities into duplicate ``songs`` rows in production:

    * Whitespace and case (long-standing behaviour).
    * Typographic apostrophes (U+2019, U+2018, U+02BC, U+201B) versus ASCII
      ``'``. Shazam → tracklist drift, observed on "Picking Dragons' Pockets".
    * Filename-sanitizer underscores. Reference audio filenames replace
      ``'`` with ``_`` (``Picking Dragons_ Pockets.mp3``); the audfprint
      ``track_id`` parser surfaces those underscores in the title, while the
      Shazam transcript keeps the original punctuation. Treating both as
      empty in the key means they collapse onto the same song.

    Apostrophes and underscores are *dropped* (not converted) because we
    cannot reliably recover the original punctuation, and songs almost
    never carry a semantically-meaningful underscore anyway.
    """
    raw = value or ""
    for ch in _TYPOGRAPHIC_APOSTROPHES:
        raw = raw.replace(ch, "'")
    raw = raw.replace("'", "").replace("_", "")
    return " ".join(raw.strip().split()).casefold()


def _song_title_key(value: str | None) -> str:
    """Identity key for song titles: ``_display_key`` plus punctuation folding.

    Extends :func:`_display_key` by neutralizing wrapping/sentence punctuation
    (parentheses, brackets, braces, periods, commas) so format drift on the
    *same* recording collapses — e.g. Shazam's ``Bedroom Posters (feat. Good
    Charlotte)`` and the audfprint filename's ``Bedroom Posters _feat. Good
    Charlotte`` resolve to one ``songs`` row, while a plain ``Bedroom Posters``
    (no feature credit) stays distinct.
    """
    raw = value or ""
    for ch in _TYPOGRAPHIC_APOSTROPHES:
        raw = raw.replace(ch, "'")
    raw = raw.replace("'", "").replace("_", "")
    raw = _SONG_TITLE_PUNCT_RE.sub(" ", raw)
    return " ".join(raw.strip().split()).casefold()


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


_UNDERSCORE_BETWEEN_LETTERS_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]")


def _has_filename_artifact_underscore(value: str | None) -> bool:
    """Detect underscores wedged between alphanumerics (filename sanitizer signature).

    The seeding downloader replaces ``'`` with ``_`` in reference audio
    filenames; the audfprint track-id parser then surfaces those underscores
    in the song title (``Can_t Stop``). When a Shazam or curated-tracklist
    title for the same song exists with the original apostrophe, we want to
    prefer the cleaner display text on the canonical row.
    """
    if not value:
        return False
    return _UNDERSCORE_BETWEEN_LETTERS_RE.search(value) is not None


def _apostrophe_quality(value: str | None) -> int:
    """Rank how cleanly a title carries its apostrophe punctuation.

    Used to break display-only ties between variants that already share a
    ``_display_key``. Higher is better:

    * ``2`` — ASCII ``'`` (the curated form we prefer for reports).
    * ``1`` — any typographic apostrophe (``’``, ``‘``, ``ʼ``, ``’``). Still
      a real apostrophe, just an inferior glyph for our text reports.
    * ``0`` — no apostrophe at all (e.g. ``Adams Song`` for a song that
      should be ``Adam's Song``). The normalizer drops apostrophes so
      identity matches, but for display we'd rather keep them.
    """
    if not value:
        return 0
    if "'" in value:
        return 2
    if any(ch in value for ch in _TYPOGRAPHIC_APOSTROPHES):
        return 1
    return 0


def _prefer_display_value(
    existing: str | None,
    incoming: str | None,
    *,
    aliases: dict[str, str] | None = None,
    key_func: Callable[[str | None], str] = _display_key,
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
    if key_func(existing_clean) != key_func(incoming_clean):
        return existing_clean
    if aliases and _display_key(existing_clean) in aliases:
        return incoming_clean
    existing_has_artifact = _has_filename_artifact_underscore(existing_clean)
    incoming_has_artifact = _has_filename_artifact_underscore(incoming_clean)
    if existing_has_artifact and not incoming_has_artifact:
        return incoming_clean
    if incoming_has_artifact and not existing_has_artifact:
        return existing_clean
    existing_apos = _apostrophe_quality(existing_clean)
    incoming_apos = _apostrophe_quality(incoming_clean)
    if incoming_apos > existing_apos:
        return incoming_clean
    if existing_apos > incoming_apos:
        return existing_clean
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
        has_schema_meta = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'schema_meta'
            """
        ).fetchone()
        if has_schema_meta is not None:
            return
        conn.executescript(schema_sql)
        conn.commit()


class BroadcastStore:
    """Single-writer SQLite store."""

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
        self._conn.create_function("display_key", 1, _display_key, deterministic=True)
        self._conn.create_function("song_title_key", 1, _song_title_key, deterministic=True)
        self._conn.execute("PRAGMA foreign_keys = ON")
        if use_wal:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate_schema()
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

    def apply_transition(self, t: SegmentTransition, *, capture_run_id: int | None = None) -> int:
        """Insert one closed segment row. Returns ``broadcast_events.id``."""
        dur = duration_seconds(t.timestamp_start, t.timestamp_end)
        cur = self._conn.execute(
            """
            INSERT INTO broadcast_events (
                timestamp_start, timestamp_end, duration,
                category, song_id, commercial_id, brand_id,
                artist, track_title, brand_name,
                transcript_excerpt, confidence, capture_run_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                capture_run_id,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    # ------------------------------------------------------------ capture runs
    def open_capture_run(
        self,
        *,
        run_id: str,
        started_utc: str,
        pipeline_version: str,
        host: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Create or return a capture run row.

        The operation is idempotent by ``run_id`` so restart wrappers can safely
        retry after a crash without creating duplicate provenance rows.
        """

        self._conn.execute(
            """
            INSERT OR IGNORE INTO capture_runs (
                run_id, started_utc, pipeline_version, host, notes
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, started_utc, pipeline_version, host, notes),
        )
        row = self._conn.execute(
            "SELECT id FROM capture_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        self._conn.commit()
        if row is None:
            raise RuntimeError(f"failed to create capture_run {run_id!r}")
        return int(row[0])

    def close_capture_run(
        self,
        *,
        run_id: str,
        ended_utc: str,
        notes: str | None = None,
    ) -> None:
        """Mark a capture run as ended."""

        if notes is None:
            self._conn.execute(
                "UPDATE capture_runs SET ended_utc = ? WHERE run_id = ?",
                (ended_utc, run_id),
            )
        else:
            self._conn.execute(
                "UPDATE capture_runs SET ended_utc = ?, notes = ? WHERE run_id = ?",
                (ended_utc, notes, run_id),
            )
        self._conn.commit()

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

        *  Look up by the same Unicode-aware display key used in Python —
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
        norm_title = _song_title_key(display_title)
        existing = self._conn.execute(
            """
            SELECT id, audfprint_track_id, source, artist, title
            FROM songs
            WHERE display_key(artist) = ?
              AND song_title_key(title) = ?
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
            next_title = _prefer_display_value(
                existing_title, display_title, key_func=_song_title_key
            )

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

    def set_song_release_date(self, song_id: int, release_date: str | None) -> None:
        """Persist a MusicBrainz-derived ``YYYY-MM-DD`` on a song row."""
        self._conn.execute(
            "UPDATE songs SET release_date = ? WHERE id = ?",
            (release_date, song_id),
        )

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
    def _migrate_schema(self) -> None:
        """Bring an existing database up to the latest schema.

        ``db/schema.sql`` uses ``CREATE TABLE IF NOT EXISTS`` and
        ``INSERT OR IGNORE`` for the version marker so opening an old v2 DB does
        not accidentally mark it as v3 before the ALTER/backfill work runs.
        """

        version = self.schema_version()
        if version == "2":
            self._migrate_v2_to_v3()
            version = self.schema_version()
        if version in (None, "3", "4"):
            self._migrate_v3_to_v4()
            return
        raise RuntimeError(f"unsupported broadcast store schema version: {version}")

    def _migrate_v3_to_v4(self) -> None:
        """Add optional MusicBrainz-derived release dates on ``songs`` rows."""
        # Some old tests/fixtures (and potentially hand-built legacy DBs) have
        # schema_meta but not every table from the current schema. Replaying the
        # idempotent schema first creates any missing tables before ALTERs run.
        schema_sql = self._schema_path.read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)
        if "release_date" not in self._table_columns("songs"):
            self._conn.execute("ALTER TABLE songs ADD COLUMN release_date TEXT")
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '4')"
        )

    def _migrate_v2_to_v3(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS capture_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                started_utc TEXT NOT NULL,
                ended_utc TEXT,
                pipeline_version TEXT NOT NULL,
                host TEXT,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            """
        )
        if "capture_run_id" not in self._table_columns("broadcast_events"):
            self._conn.execute(
                "ALTER TABLE broadcast_events "
                "ADD COLUMN capture_run_id INTEGER REFERENCES capture_runs(id)"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_capture_run "
            "ON broadcast_events (capture_run_id)"
        )
        self._backfill_legacy_capture_run()
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', '3')"
        )

    def _backfill_legacy_capture_run(self) -> None:
        row = self._conn.execute(
            """
            SELECT MIN(timestamp_start), MAX(COALESCE(timestamp_end, timestamp_start)), COUNT(*)
            FROM broadcast_events
            """
        ).fetchone()
        if row is None or int(row[2] or 0) == 0:
            return
        started_utc = str(row[0])
        ended_utc = str(row[1])
        self._conn.execute(
            """
            INSERT OR IGNORE INTO capture_runs (
                run_id, started_utc, ended_utc, pipeline_version, notes
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "legacy_pre_v3",
                started_utc,
                ended_utc,
                "unknown",
                "Backfilled during schema v3 migration",
            ),
        )
        legacy_id = self._conn.execute(
            "SELECT id FROM capture_runs WHERE run_id = 'legacy_pre_v3'"
        ).fetchone()[0]
        self._conn.execute(
            "UPDATE broadcast_events SET capture_run_id = ? WHERE capture_run_id IS NULL",
            (int(legacy_id),),
        )

    def _table_columns(self, table_name: str) -> set[str]:
        return {
            str(row[1])
            for row in self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }

    def schema_version(self) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'",
        ).fetchone()
        return str(row[0]) if row else None
