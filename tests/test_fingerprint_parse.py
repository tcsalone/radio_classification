"""Parser tests for audfprint match output."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from radio_classifier.fingerprint.audfprint_engine import (
    AudfprintConfig,
    AudfprintIndex,
    _audfprint_argv,
    _default_index_ncores,
    _split_track_id,
    parse_audfprint_batch_output,
    parse_audfprint_candidates,
    parse_audfprint_match_output,
)
from radio_classifier.fingerprint.types import FingerprintStatus


def test_default_audfprint_candidate_floor_is_below_strong_acceptance_floor() -> None:
    """The CLI surfaces low-score candidates for downstream confirmation.

    They are not accepted directly by the funnel; scores below the strong
    acceptance threshold (~60) need extra adjacent same-track support before
    Tier 1 wins. The 2026-05-31 validated-unknowns eval showed lowering this
    floor from 45 to 30 recovered Bad Omens (score 60) without surfacing any
    new false positive from the known Linkin Park / Temper City collision
    (score 67) which sits above either threshold.
    """
    assert AudfprintConfig().min_count == 30


def test_default_index_ncores_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADIO_CLASSIFIER_AUDFPRINT_INDEX_NCORES", "3")
    assert _default_index_ncores() == 3


def test_default_index_ncores_ignores_bogus_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADIO_CLASSIFIER_AUDFPRINT_INDEX_NCORES", "not-a-number")
    value = _default_index_ncores()
    assert 1 <= value <= 6


def test_build_or_extend_passes_ncores_when_configured(tmp_path: Path) -> None:
    """Parallel hashing via ``--ncores`` cuts rebuild wall time substantially.
    The wrapper must forward the configured worker count to audfprint.
    """
    idx_path = tmp_path / "songs.pklz"
    config = AudfprintConfig(index_ncores=4)
    index = AudfprintIndex(index_path=idx_path, config=config)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00")

    with mock.patch(
        "radio_classifier.fingerprint.audfprint_engine.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
    ) as run:
        index.build_or_extend([audio])

    cmd = run.call_args.args[0]
    assert "new" in cmd
    assert "--ncores" in cmd
    ncores_index = cmd.index("--ncores")
    assert cmd[ncores_index + 1] == "4"


def test_build_or_extend_omits_ncores_when_single_threaded(tmp_path: Path) -> None:
    """Single-threaded mode must not pass ``--ncores`` so we stay compatible
    with audfprint builds that predate the flag."""
    idx_path = tmp_path / "songs.pklz"
    config = AudfprintConfig(index_ncores=1)
    index = AudfprintIndex(index_path=idx_path, config=config)
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00")

    with mock.patch(
        "radio_classifier.fingerprint.audfprint_engine.subprocess.run",
        return_value=mock.Mock(returncode=0, stdout="", stderr=""),
    ) as run:
        index.build_or_extend([audio])

    cmd = run.call_args.args[0]
    assert "--ncores" not in cmd


def test_parse_no_match_explicit() -> None:
    r = parse_audfprint_match_output("NOMATCH query.wav\n", "ts")
    assert r.status is FingerprintStatus.no_match
    assert r.window_start_utc == "ts"


def test_parse_no_match_empty() -> None:
    r = parse_audfprint_match_output("", "ts")
    assert r.status is FingerprintStatus.no_match


def test_parse_match_format_a() -> None:
    line = "Matched query.wav as Taylor Swift - Anti-Hero.mp3 with 13 of 24 common hashes\n"
    r = parse_audfprint_match_output(line, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "Taylor Swift"
    assert r.title == "Anti-Hero"
    assert r.match_score == 13.0
    assert r.track_id == "Taylor Swift - Anti-Hero.mp3"


def test_parse_match_format_b_path() -> None:
    line = (
        "Matched   /tmp/win.wav 5.0 sec  starting at  0.0 sec in  reference/Singer_-_Song.wav "
        "as reference/Singer_-_Song.wav with  9 common hashes at rank 0\n"
    )
    r = parse_audfprint_match_output(line, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "Singer"
    assert r.title == "Song"


def test_parse_strips_offset_suffix_from_track_id() -> None:
    """Live audfprint emits ``... as REF.mp3 at  104.4 s with 12 ...``.

    Regression for a parser bug where ``at 104.4 s`` was captured into the
    track id, which broke ``_split_track_id`` (the ``.mp3`` extension stripper
    only matched at end-of-string), producing titles like
    ``Smells Like Teen Spirit.mp3 at  104.4 s`` in the persisted songs table.
    """
    line = (
        "Matched /tmp/win.wav 20.0 sec as "
        "data/reference/songs/Nirvana - Smells Like Teen Spirit.mp3 "
        "at  104.4 s with 12 of 80 common hashes at rank 0\n"
    )
    r = parse_audfprint_match_output(line, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "Nirvana"
    assert r.title == "Smells Like Teen Spirit"
    assert r.track_id == "data/reference/songs/Nirvana - Smells Like Teen Spirit.mp3"
    assert r.match_score == 12.0


def test_parse_strips_negative_offset_suffix_from_track_id() -> None:
    """audfprint can emit negative offsets for boundary windows."""
    line = (
        "Matched /tmp/win.wav 20.0 sec as "
        "data/reference/songs/Yellowcard - Bedroom Posters.mp3 "
        "at  -13.1 s with 17 of 80 common hashes at rank 0\n"
    )
    r = parse_audfprint_match_output(line, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "Yellowcard"
    assert r.title == "Bedroom Posters"
    assert r.track_id == "data/reference/songs/Yellowcard - Bedroom Posters.mp3"
    assert r.match_score == 17.0


def test_parse_picks_best_match_by_score() -> None:
    """We must pick the highest-score line, not the first.

    Regression for an index-poisoning bug where a noisy reference (a poor
    quality opus/webm rip) won audfprint's ``rank 0`` slot with a small
    score and crowded out the real, much higher-score match listed below
    it as ``rank 1+``. With ``--max-matches 1`` this caused real matches
    to be reported as no-match.
    """
    stdout = (
        "Matched q.wav as Noisy - Reference.webm at 12.0 s "
        "with 15 of 500 common hashes at rank 0\n"
        "Matched q.wav as Real - Match.mp3 at 216.0 s "
        "with 60 of 246 common hashes at rank 1\n"
        "Matched q.wav as Noisy - Reference.webm at 18.0 s "
        "with 10 of 500 common hashes at rank 0\n"
    )
    r = parse_audfprint_match_output(stdout, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "Real"
    assert r.title == "Match"
    assert r.match_score == 60.0


def test_parse_batch_picks_best_match_per_query(tmp_path) -> None:
    """Same best-by-score rule must apply per-query in batch mode."""
    q1 = tmp_path / "window_000001.wav"
    q2 = tmp_path / "window_000002.wav"
    stdout = "\n".join(
        [
            f"Matched {q1} as Spurious - Rip.webm at 1.0 s with 5 of 300 common hashes at rank 0",
            f"Matched {q1} as Correct - Track.mp3 at 60.0 s with 50 of 200 common hashes at rank 1",
            f"NOMATCH {q2}",
        ]
    )
    results = parse_audfprint_batch_output(stdout, [q1, q2], ["ts1", "ts2"])
    assert results[0].status is FingerprintStatus.match
    assert results[0].artist == "Correct"
    assert results[0].title == "Track"
    assert results[0].match_score == 50.0
    assert results[1].status is FingerprintStatus.no_match


def test_parse_audfprint_candidates_returns_all_matches_in_score_order() -> None:
    """`fingerprint explain` relies on every candidate, not just the winner."""
    stdout = (
        "Matched q.wav as Noisy - Reference.webm at 12.0 s "
        "with 15 of 500 common hashes at rank 0\n"
        "Matched q.wav as Real - Match.mp3 at 216.0 s "
        "with 60 of 246 common hashes at rank 1\n"
        "Matched q.wav as Another - Weak.mp3 at 4.0 s "
        "with 7 of 120 common hashes at rank 2\n"
    )
    candidates = parse_audfprint_candidates(stdout)
    assert candidates == [
        ("Real - Match.mp3", 60),
        ("Noisy - Reference.webm", 15),
        ("Another - Weak.mp3", 7),
    ]


def test_parse_audfprint_candidates_handles_no_match() -> None:
    assert parse_audfprint_candidates("") == []
    assert parse_audfprint_candidates("NOMATCH q.wav\n") == []


def test_default_max_matches_is_above_one() -> None:
    """We need top-N candidates from audfprint so the parser can pick the
    best-scoring one (audfprint's ``rank 0`` is not score-ordered)."""
    assert AudfprintConfig().max_matches >= 5


def test_parse_batch_output_maps_matches_to_queries(tmp_path) -> None:
    q1 = tmp_path / "window_000001.wav"
    q2 = tmp_path / "window_000002.wav"
    q3 = tmp_path / "window_000003.wav"
    stdout = "\n".join(
        [
            f"Matched {q1} as MGMT - Kids.mp3 with 42 of 80 common hashes",
            f"NOMATCH {q2}",
            "Matched window_000003.wav as AFI - Miss Murder.mp3 at  -3.2 s with 17 of 80 common hashes",
        ]
    )

    results = parse_audfprint_batch_output(
        stdout,
        [q1, q2, q3],
        ["ts1", "ts2", "ts3"],
    )

    assert [r.window_start_utc for r in results] == ["ts1", "ts2", "ts3"]
    assert results[0].status is FingerprintStatus.match
    assert results[0].artist == "MGMT"
    assert results[0].title == "Kids"
    assert results[1].status is FingerprintStatus.no_match
    assert results[2].status is FingerprintStatus.match
    assert results[2].artist == "AFI"
    assert results[2].title == "Miss Murder"


def test_split_track_id_handles_extensions_and_paths() -> None:
    assert _split_track_id("Foo - Bar.mp3") == ("Foo", "Bar")
    assert _split_track_id("/refs/Foo - Bar.flac") == ("Foo", "Bar")
    assert _split_track_id("Foo - Bar.webm") == ("Foo", "Bar")
    assert _split_track_id("OnlyTitle.wav") == (None, "OnlyTitle")
    assert _split_track_id("weird name") == (None, "weird name")


def test_split_track_id_collapses_alternate_reference_suffix() -> None:
    """Multiple reference recordings for the same song must share an identity.

    Lets the operator drop a second Wonderwall reference (e.g. the original
    1995 master alongside the 2024 stereo remaster we already have) into
    ``data/reference/songs/`` and have both audfprint matches resolve to
    the same ``songs.id`` after upsert.
    """
    assert _split_track_id("Oasis - Wonderwall (alt).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (alt 2).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (alternate).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (ref).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (ref 2).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (reference).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (source).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id("Oasis - Wonderwall (v2).mp3") == ("Oasis", "Wonderwall")
    assert _split_track_id(
        "data/reference/songs/Oasis - Wonderwall (alt).mp3"
    ) == ("Oasis", "Wonderwall")


def test_split_track_id_preserves_genuine_parenthetical_variants() -> None:
    """We only strip narrow operator-supplied disambiguators.

    Legitimate variants — live cuts, remixes, acoustic versions, feature
    credits — must stay distinguishable so they remain separate songs.
    """
    assert _split_track_id("Oasis - Wonderwall (MTV Unplugged).mp3") == (
        "Oasis",
        "Wonderwall (MTV Unplugged)",
    )
    assert _split_track_id("Julia Wolf - In My Room (Acoustic).mp3") == (
        "Julia Wolf",
        "In My Room (Acoustic)",
    )
    assert _split_track_id("Weezer - Go Away (feat. Best Coast).mp3") == (
        "Weezer",
        "Go Away (feat. Best Coast)",
    )
    assert _split_track_id("Linkin Park - Crawling (Reanimation).mp3") == (
        "Linkin Park",
        "Crawling (Reanimation)",
    )


def test_audfprint_argv_expands_tilde_and_envvars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``RADIO_CLASSIFIER_AUDFPRINT_BIN`` must expand ``~`` and ``$VAR`` tokens.

    Regression for the previous behavior where ``python ~/dev/audfprint/audfprint.py``
    was passed to subprocess verbatim and Python resolved ``~`` as a relative path
    against ``cwd``.
    """
    monkeypatch.setenv("HOME", "/home/example")
    monkeypatch.setenv("RC_TEST_PREFIX", "/opt/rc")
    monkeypatch.setenv(
        "RADIO_CLASSIFIER_AUDFPRINT_BIN",
        "python ~/dev/audfprint/audfprint.py --extra $RC_TEST_PREFIX/x",
    )
    argv = _audfprint_argv()
    assert argv[0] == "python"
    assert argv[1] == "/home/example/dev/audfprint/audfprint.py"
    assert argv[-1] == "/opt/rc/x"
    # No raw ``~`` survives anywhere in the resolved argv.
    assert not any("~" in tok for tok in argv)


def test_audfprint_argv_falls_back_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RADIO_CLASSIFIER_AUDFPRINT_BIN", raising=False)
    argv = _audfprint_argv()
    assert argv  # never empty
    # Either ``audfprint`` on PATH, local clone auto-discovery, or
    # ``python -m audfprint`` fallback.
    assert argv[0] == "audfprint" or (
        argv[0] == sys.executable
        and (argv[1:] == ["-m", "audfprint"] or argv[1].endswith("/audfprint.py"))
    )


def test_audfprint_argv_discovers_common_local_clone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("RADIO_CLASSIFIER_AUDFPRINT_BIN", raising=False)
    monkeypatch.setattr(
        "radio_classifier.fingerprint.audfprint_engine.shutil.which",
        lambda _name: None,
    )
    home = tmp_path / "home"
    clone = home / "dev" / "audfprint" / "audfprint.py"
    clone.parent.mkdir(parents=True)
    clone.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    assert _audfprint_argv() == [sys.executable, str(clone)]
