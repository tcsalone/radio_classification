"""Tests for Shazam discovery listing and tracklist promotion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radio_classifier.discovery import (
    dedupe_songs,
    list_shazam_discoveries,
    promote_to_tracklist,
)
from radio_classifier.persistence import BroadcastStore
from radio_classifier.reports import format_discoveries
from radio_classifier.segments.types import BroadcastCategory, SegmentTransition


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _seed_shazam_db(tmp_path: Path) -> BroadcastStore:
    store = BroadcastStore(tmp_path / "rc.db")
    now = datetime.now(tz=timezone.utc)

    djo = store.upsert_song(
        artist="Djo", title="End of Beginning", source="shazam"
    )
    float_on = store.upsert_song(
        artist="Modest Mouse", title="Float On", source="shazam"
    )
    guns = store.upsert_song(
        artist="Green Day", title="21 Guns", source="shazam"
    )
    audfprint = store.upsert_song(
        artist="Nirvana", title="Smells Like Teen Spirit", source="audfprint"
    )

    def song_play(song_id: int, artist: str, title: str, *, minutes_ago: int) -> None:
        start = now - timedelta(minutes=minutes_ago)
        end = start + timedelta(seconds=180)
        store.apply_transition(
            SegmentTransition(
                timestamp_start=_iso(start),
                timestamp_end=_iso(end),
                category=BroadcastCategory.SONG,
                artist=artist,
                track_title=title,
                song_id=song_id,
            )
        )

    song_play(djo, "Djo", "End of Beginning", minutes_ago=30)
    song_play(djo, "Djo", "End of Beginning", minutes_ago=25)
    song_play(djo, "Djo", "End of Beginning", minutes_ago=20)
    song_play(float_on, "Modest Mouse", "Float On", minutes_ago=15)
    song_play(guns, "Green Day", "21 Guns", minutes_ago=10)
    song_play(audfprint, "Nirvana", "Smells Like Teen Spirit", minutes_ago=5)
    return store


def _write_tracklist(path: Path, lines: list[str]) -> None:
    path.write_text(
        "# test tracklist\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def test_list_shazam_discoveries_ranks_by_plays(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            tracklist_path=tracklist,
        )
        assert len(rows) == 2
        assert rows[0].artist == "Djo"
        assert rows[0].play_count == 3
        assert rows[0].in_tracklist is False
        assert rows[0].needs_review is False
        assert rows[1].artist == "Modest Mouse"
        assert rows[1].play_count == 1
        assert rows[1].needs_review is True
    finally:
        store.close()


def test_list_shazam_discoveries_hides_indexed_by_default(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            include_indexed=True,
            tracklist_path=tracklist,
        )
        assert any(r.artist == "Green Day" and r.in_tracklist for r in rows)
        hidden = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            include_indexed=False,
            tracklist_path=tracklist,
        )
        assert all(r.artist != "Green Day" for r in hidden)
    finally:
        store.close()


def test_list_shazam_discoveries_min_plays_filter(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(
            store,
            since_utc=since,
            top_n=10,
            min_plays=2,
            tracklist_path=None,
        )
        assert len(rows) == 1
        assert rows[0].artist == "Djo"
    finally:
        store.close()


def test_format_discoveries_renders_table(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    since = _iso(datetime.now(tz=timezone.utc) - timedelta(hours=24))
    try:
        rows = list_shazam_discoveries(store, since_utc=since, top_n=5, tracklist_path=None)
        text = format_discoveries(rows)
        assert "Djo" in text
        assert "missing" in text
        assert "manual" in text
        assert "review" in text
        assert "plays" in text
    finally:
        store.close()


def test_promote_appends_new_tracks(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    djo_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Djo'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert result.appended_count == 1
        body = tracklist.read_text(encoding="utf-8")
        assert "Djo | End of Beginning" in body
        assert "promoted from Shazam discoveries on 2026-05-25" in body
    finally:
        store.close()


def test_promote_skips_already_in_tracklist(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Green Day | 21 Guns"])
    guns_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Green Day'"
    ).fetchone()[0]
    before = tracklist.read_text(encoding="utf-8")
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[guns_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert result.appended_count == 0
        assert result.skipped_count == 1
        assert result.promoted[0].reason == "already in tracklist"
        assert tracklist.read_text(encoding="utf-8") == before
    finally:
        store.close()


def test_promote_skips_normalized_duplicate(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["green day | 21 guns"])
    guns_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Green Day'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[guns_id],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "already in tracklist" in result.promoted[0].reason
    finally:
        store.close()


def test_upsert_song_collapses_unicode_apostrophe_and_filename_underscore(
    tmp_path: Path,
) -> None:
    """Three forms of the same song must resolve to one ``songs`` row.

    Real-world drift seen in ``data/store/broadcast.db`` on 2026-06-01:

    * Shazam returned ``Picking Dragons’ Pockets`` (U+2019).
    * audfprint registered the reference recording as
      ``Picking Dragons_ Pockets`` because the filename sanitizer
      replaced the ASCII apostrophe with an underscore.
    * The curated tracklist file used a plain ASCII apostrophe.

    The store keyed identity on a casefold-only ``display_key``, so the
    Shazam row and the audfprint row coexisted as separate songs and the
    operator's discovery listing showed a "new" track that was already on
    the tracklist. The fix normalizes typographic apostrophes to ASCII and
    drops both apostrophes and filename-artifact underscores from the key.
    """
    store = BroadcastStore(tmp_path / "apostrophe.db")
    try:
        shazam_id = store.upsert_song(
            artist="Modest Mouse",
            title="Picking Dragons\u2019 Pockets",  # curly
            source="shazam",
        )
        audfprint_id = store.upsert_song(
            artist="Modest Mouse",
            title="Picking Dragons_ Pockets",  # filename-sanitizer underscore
            audfprint_track_id="data/reference/songs/Modest Mouse - Picking Dragons_ Pockets.mp3",
            source="audfprint",
        )
        ascii_id = store.upsert_song(
            artist="Modest Mouse",
            title="Picking Dragons' Pockets",  # ASCII apostrophe
            source="shazam",
        )
        assert shazam_id == audfprint_id == ascii_id, (
            "three apostrophe/underscore variants should resolve to one song"
        )
        row = store.connection.execute(
            "SELECT source, audfprint_track_id FROM songs WHERE id = ?",
            (shazam_id,),
        ).fetchone()
        assert row[0] == "audfprint", "audfprint upgrade should stick"
        assert row[1] is not None
    finally:
        store.close()


def test_dedupe_folds_existing_apostrophe_underscore_duplicates(tmp_path: Path) -> None:
    """A pre-fix DB with two duplicate rows should fold on dedupe.

    Simulates the live row state captured on 2026-06-01: id=56 was the
    Shazam row with U+2019, id=57 was the audfprint row with the filename
    underscore. ``dedupe_songs`` must surface them as one group and pick
    the audfprint row as the survivor.
    """
    store = BroadcastStore(tmp_path / "legacy-dupes.db")
    conn = store.connection
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) "
        "VALUES (?, ?, ?, ?)",
        ("Modest Mouse", "Picking Dragons\u2019 Pockets", "shazam", None),
    )
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) "
        "VALUES (?, ?, ?, ?)",
        (
            "Modest Mouse",
            "Picking Dragons_ Pockets",
            "audfprint",
            "data/reference/songs/Modest Mouse - Picking Dragons_ Pockets.mp3",
        ),
    )
    conn.commit()
    try:
        report = dedupe_songs(store, dry_run=False)
        assert report.collapsed_pairs == 1
        assert report.rows_deleted == 1
        survivors = conn.execute(
            "SELECT artist, title, source, audfprint_track_id FROM songs "
            "WHERE artist = 'Modest Mouse'"
        ).fetchall()
        assert len(survivors) == 1
        assert survivors[0][2] == "audfprint", "audfprint row should survive"
    finally:
        store.close()


def test_dedupe_upgrades_survivor_title_from_filename_underscore_to_apostrophe(
    tmp_path: Path,
) -> None:
    """Even when the audfprint row wins on identity, dedupe should adopt the
    cleaner apostrophe spelling from the Shazam loser for the displayed
    title — the underscore is a filename-sanitizer artifact, not the way an
    operator wants to see the song in reports."""
    store = BroadcastStore(tmp_path / "display-upgrade.db")
    conn = store.connection
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) "
        "VALUES (?, ?, ?, ?)",
        ("Blink-182", "Adams Song", "shazam", None),
    )
    # Shazam loser with cleanest apostrophe spelling.
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) "
        "VALUES (?, ?, ?, ?)",
        ("Blink-182", "Adam's Song", "shazam", None),
    )
    # audfprint survivor with filename-sanitizer underscore.
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) "
        "VALUES (?, ?, ?, ?)",
        (
            "Blink-182",
            "Adam_s Song",
            "audfprint",
            "data/reference/songs/Blink-182 - Adam_s Song.mp3",
        ),
    )
    conn.commit()
    try:
        report = dedupe_songs(store, dry_run=False)
        assert report.collapsed_pairs == 2
        survivors = conn.execute(
            "SELECT artist, title, source, audfprint_track_id FROM songs "
            "WHERE artist = 'Blink-182'"
        ).fetchall()
        assert len(survivors) == 1
        artist, title, source, track_id = survivors[0]
        assert source == "audfprint"
        assert track_id is not None
        assert title == "Adam's Song", (
            "survivor should keep audfprint identity but adopt clean apostrophe spelling"
        )
    finally:
        store.close()


def test_promote_recognises_underscore_filename_artifact_as_already_in_tracklist(
    tmp_path: Path,
) -> None:
    """A Shazam discovery whose title comes back with a filename underscore
    must still be detected as already on the curated tracklist."""
    store = BroadcastStore(tmp_path / "underscore.db")
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, ["Modest Mouse | Picking Dragons' Pockets"])
    try:
        song_id = store.upsert_song(
            artist="Modest Mouse",
            title="Picking Dragons_ Pockets",
            source="shazam",
        )
        result = promote_to_tracklist(
            store,
            song_ids=[song_id],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert result.skipped_count == 1
        assert result.promoted[0].reason == "already in tracklist"
    finally:
        store.close()


def test_promote_skips_tracklist_match_with_apostrophe_and_feature_suffix(tmp_path: Path) -> None:
    store = BroadcastStore(tmp_path / "tracklist-normalize.db")
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(
        tracklist,
        [
            "Royel Otis | Who's Your Boyfriend",
            "Weezer | Go Away",
            "Dexter And The Moonrocks | Freakin' Out",
        ],
    )
    try:
        royel = store.upsert_song(artist="Royel Otis", title="who’s your boyfriend", source="shazam")
        weezer = store.upsert_song(artist="Weezer", title="Go Away (feat. Best Coast)", source="shazam")
        dexter = store.upsert_song(
            artist="Dexter and The Moonrocks",
            title="Freakin’ Out",
            source="shazam",
        )

        result = promote_to_tracklist(
            store,
            song_ids=[royel, weezer, dexter],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert result.skipped_count == 3
        assert all(p.reason == "already in tracklist" for p in result.promoted)
    finally:
        store.close()


def test_promote_refuses_non_shazam_source(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    nirvana_id = store.connection.execute(
        "SELECT id FROM songs WHERE source = 'audfprint'"
    ).fetchone()[0]
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[nirvana_id],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "source='audfprint'" in result.promoted[0].reason
    finally:
        store.close()


def test_promote_is_idempotent(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    djo_id = store.connection.execute(
        "SELECT id FROM songs WHERE artist = 'Djo'"
    ).fetchone()[0]
    try:
        first = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        second = promote_to_tracklist(
            store,
            song_ids=[djo_id],
            tracklist_path=tracklist,
            now_iso_date="2026-05-25",
        )
        assert first.appended_count == 1
        assert second.appended_count == 0
        assert second.promoted[0].reason == "already in tracklist"
        assert tracklist.read_text(encoding="utf-8").count("Djo | End of Beginning") == 1
    finally:
        store.close()


def test_promote_unknown_song_id(tmp_path: Path) -> None:
    store = _seed_shazam_db(tmp_path)
    tracklist = tmp_path / "tracklist.txt"
    _write_tracklist(tracklist, [])
    try:
        result = promote_to_tracklist(
            store,
            song_ids=[9999],
            tracklist_path=tracklist,
        )
        assert result.appended_count == 0
        assert "not found" in result.promoted[0].reason
    finally:
        store.close()


# ----------------------------------------------------------------- dedupe tests
def _seed_dupe_db(tmp_path: Path) -> tuple[BroadcastStore, dict[str, int]]:
    """Build a DB that simulates the pre-fix pollution: Shazam + audfprint
    rows for the same songs, plus already-clean rows for control.

    Returns the open store and a mapping of human-readable labels to the
    expected surviving ``songs.id`` for each test scenario. The seed has to
    bypass the normalised upsert (we want to demonstrate cleanup of old
    state), so it writes rows directly via SQL.
    """
    from datetime import datetime, timedelta, timezone

    store = BroadcastStore(tmp_path / "dupe.db")
    conn = store.connection

    # Pair 1: shazam row first (id will be lower), audfprint row second.
    # Survivor MUST be the audfprint row because it has a track id.
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) VALUES (?, ?, ?, ?)",
        ("Foo Fighters", "Times Like These", "shazam", None),
    )
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) VALUES (?, ?, ?, ?)",
        ("Foo Fighters", "Times Like These", "audfprint", "ref/foo.mp3"),
    )

    # Pair 2: casing variation only — pick survivor by event count.
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) VALUES (?, ?, ?, ?)",
        ("Evanescence", "Bring Me to Life", "shazam", None),  # lowercase 'to'
    )
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) VALUES (?, ?, ?, ?)",
        ("Evanescence", "Bring Me To Life", "audfprint", "ref/evan.mp3"),  # caps 'To'
    )

    # Clean control: a single-row song should be untouched.
    conn.execute(
        "INSERT INTO songs (artist, title, source, audfprint_track_id) VALUES (?, ?, ?, ?)",
        ("Nirvana", "Lithium", "audfprint", "ref/nirvana.mp3"),
    )

    conn.commit()

    ids = {row[2]: row[0] for row in conn.execute("SELECT id, artist, title FROM songs")}
    label_to_id = {
        "foo_shazam": conn.execute(
            "SELECT id FROM songs WHERE artist = 'Foo Fighters' AND source = 'shazam'"
        ).fetchone()[0],
        "foo_audfprint": conn.execute(
            "SELECT id FROM songs WHERE artist = 'Foo Fighters' AND source = 'audfprint'"
        ).fetchone()[0],
        "evan_shazam": conn.execute(
            "SELECT id FROM songs WHERE artist = 'Evanescence' AND source = 'shazam'"
        ).fetchone()[0],
        "evan_audfprint": conn.execute(
            "SELECT id FROM songs WHERE artist = 'Evanescence' AND source = 'audfprint'"
        ).fetchone()[0],
        "nirvana": conn.execute(
            "SELECT id FROM songs WHERE artist = 'Nirvana'"
        ).fetchone()[0],
    }

    # A few broadcast_events on each side so we can verify repointing.
    now = datetime.now(tz=timezone.utc)

    def _evt(song_id: int, offset_min: int) -> None:
        start = now - timedelta(minutes=offset_min)
        end = start + timedelta(seconds=180)
        conn.execute(
            "INSERT INTO broadcast_events "
            "(timestamp_start, timestamp_end, duration, category, song_id, artist, track_title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _iso(start),
                _iso(end),
                180.0,
                "SONG",
                song_id,
                "x",
                "y",
            ),
        )

    _evt(label_to_id["foo_shazam"], 60)       # 1 event on Foo Shazam
    _evt(label_to_id["foo_audfprint"], 50)    # 1 event on Foo audfprint
    _evt(label_to_id["evan_shazam"], 40)      # 2 events on Evan shazam (will win on event count tie if both lacked track_id, but audfprint should still win)
    _evt(label_to_id["evan_shazam"], 35)
    _evt(label_to_id["evan_audfprint"], 30)
    _evt(label_to_id["nirvana"], 20)
    conn.commit()
    return store, label_to_id


def test_dedupe_folds_shazam_into_audfprint_when_track_id_present(tmp_path: Path) -> None:
    """Foo Fighters has a Shazam row (no track id) and an audfprint row (with
    track id). The audfprint row must win regardless of insertion order or
    event count."""
    store, ids = _seed_dupe_db(tmp_path)
    try:
        report = dedupe_songs(store, dry_run=False)
        assert report.dry_run is False
        assert report.collapsed_pairs == 2  # Foo + Evanescence groups, 1 loser each
        assert report.events_repointed == 3  # 1 Foo shazam + 2 Evan shazam events
        assert report.rows_deleted == 2

        # Survivor for Foo is the audfprint row.
        survivor_id = ids["foo_audfprint"]
        row = store.connection.execute(
            "SELECT id, source, audfprint_track_id FROM songs WHERE artist = 'Foo Fighters'"
        ).fetchall()
        assert len(row) == 1
        assert row[0] == (survivor_id, "audfprint", "ref/foo.mp3")

        # Both Foo events now point at the survivor.
        cnt = store.connection.execute(
            "SELECT COUNT(*) FROM broadcast_events WHERE song_id = ?", (survivor_id,)
        ).fetchone()[0]
        assert cnt == 2
    finally:
        store.close()


def test_dedupe_dry_run_is_pure(tmp_path: Path) -> None:
    """A dry run reports what would happen without touching the DB."""
    store, _ = _seed_dupe_db(tmp_path)
    try:
        report = dedupe_songs(store, dry_run=True)
        assert report.dry_run is True
        assert report.collapsed_pairs == 2
        assert report.events_repointed == 0  # nothing actually written
        assert report.rows_deleted == 0

        # The pollution is still there for a real run later.
        n_songs = store.connection.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        assert n_songs == 5
    finally:
        store.close()


def test_dedupe_leaves_singletons_alone(tmp_path: Path) -> None:
    """A song with only one row must not appear in any dedupe group."""
    store, ids = _seed_dupe_db(tmp_path)
    try:
        report = dedupe_songs(store, dry_run=False)
        nirvana_id = ids["nirvana"]
        # No group should mention Nirvana.
        assert all("nirvana" not in g.key[0] for g in report.groups)
        # And the row still exists with its track id intact.
        row = store.connection.execute(
            "SELECT id, audfprint_track_id FROM songs WHERE id = ?", (nirvana_id,)
        ).fetchone()
        assert row == (nirvana_id, "ref/nirvana.mp3")
    finally:
        store.close()


def test_dedupe_skips_rows_with_blank_artist_or_title(tmp_path: Path) -> None:
    """Two ``(None, None)`` rows must NOT be auto-merged — they have no usable
    identity and need manual inspection."""
    store = BroadcastStore(tmp_path / "blanks.db")
    try:
        conn = store.connection
        conn.execute("INSERT INTO songs (artist, title, source) VALUES (NULL, NULL, 'shazam')")
        conn.execute("INSERT INTO songs (artist, title, source) VALUES (NULL, NULL, 'shazam')")
        conn.execute("INSERT INTO songs (artist, title, source) VALUES ('', '', 'shazam')")
        conn.commit()

        report = dedupe_songs(store, dry_run=False)
        assert report.groups == []
        assert report.rows_deleted == 0
    finally:
        store.close()
