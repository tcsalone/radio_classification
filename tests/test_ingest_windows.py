"""Sliding-window tests inherited from live105sux (adapted to defaults)."""

from __future__ import annotations

import numpy as np
import pytest

from radio_classifier.ingest.windows import iter_overlapping_windows


def _make_pcm(n_samples: int, rate: int = 16_000) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.int16)


def test_partial_tail_dropped() -> None:
    rate = 16_000
    seconds = 22.0  # one 20s window then 2s tail (dropped)
    pcm = _make_pcm(int(rate * seconds), rate)
    windows = list(
        iter_overlapping_windows(pcm, rate, window_seconds=20.0, overlap_fraction=0.5)
    )
    # 20s window, 10s hop: starts at 0s and 10s, but window at 10s would need
    # samples up to 30s — only 22s available, so it's dropped.
    assert len(windows) == 1


def test_overlapping_windows_count() -> None:
    rate = 16_000
    seconds = 60.0
    pcm = _make_pcm(int(rate * seconds), rate)
    windows = list(
        iter_overlapping_windows(pcm, rate, window_seconds=20.0, overlap_fraction=0.5)
    )
    # 20s windows at 10s hop over 60s: starts at 0,10,20,30,40 (50→70 dropped)
    assert len(windows) == 5
    for w in windows:
        assert w.frame_count == rate * 20


def test_validates_overlap_fraction() -> None:
    pcm = _make_pcm(16_000)
    with pytest.raises(ValueError):
        list(iter_overlapping_windows(pcm, 16_000, window_seconds=1.0, overlap_fraction=1.0))
