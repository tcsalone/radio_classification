"""Tier-2 routing logic (the pure function — no TF Hub required)."""

from __future__ import annotations

from radio_classifier.acoustic.yamnet_backend import route_label
from radio_classifier.acoustic.types import AcousticLabel


def test_route_argmax_simple() -> None:
    assert route_label(0.6, 0.3, 0.1) is AcousticLabel.MUSIC
    assert route_label(0.1, 0.6, 0.3) is AcousticLabel.SPEECH
    assert route_label(0.1, 0.3, 0.6) is AcousticLabel.OTHER


def test_route_below_min_prob_defaults_to_speech() -> None:
    assert route_label(0.1, 0.1, 0.1, min_prob=0.25) is AcousticLabel.SPEECH


def test_route_dj_over_music_prefers_speech_with_bias() -> None:
    # Music slightly higher but speech non-trivial → SPEECH wins with bias on.
    assert route_label(0.55, 0.40, 0.05, speech_bias=True) is AcousticLabel.SPEECH
    # Same input without bias → MUSIC.
    assert route_label(0.55, 0.40, 0.05, speech_bias=False) is AcousticLabel.MUSIC


def test_route_pure_music_unaffected_by_speech_bias() -> None:
    assert route_label(0.9, 0.05, 0.05, speech_bias=True) is AcousticLabel.MUSIC


def test_route_pure_speech_unchanged() -> None:
    assert route_label(0.05, 0.9, 0.05) is AcousticLabel.SPEECH
