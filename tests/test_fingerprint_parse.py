"""Parser tests for audfprint match output."""

from __future__ import annotations

import sys

import pytest

from radio_classifier.fingerprint.audfprint_engine import (
    _audfprint_argv,
    _split_track_id,
    parse_audfprint_match_output,
)
from radio_classifier.fingerprint.types import FingerprintStatus


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


def test_parse_picks_first_match() -> None:
    stdout = (
        "Matched q.wav as A - B.mp3 with 10 of 30 common hashes\n"
        "Matched q.wav as C - D.mp3 with 4 of 30 common hashes\n"
    )
    r = parse_audfprint_match_output(stdout, "ts")
    assert r.status is FingerprintStatus.match
    assert r.artist == "A"
    assert r.title == "B"


def test_split_track_id_handles_extensions_and_paths() -> None:
    assert _split_track_id("Foo - Bar.mp3") == ("Foo", "Bar")
    assert _split_track_id("/refs/Foo - Bar.flac") == ("Foo", "Bar")
    assert _split_track_id("Foo - Bar.webm") == ("Foo", "Bar")
    assert _split_track_id("OnlyTitle.wav") == (None, "OnlyTitle")
    assert _split_track_id("weird name") == (None, "weird name")


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
