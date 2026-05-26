"""Sliding overlapping windows over mono int16 PCM (numpy)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    """One analysis window of PCM samples with UTC timestamp metadata."""

    samples: np.ndarray  # shape (n_frames,), int16
    sample_rate_hz: int
    window_start_utc: str  # ISO-8601 UTC ending with Z
    frame_count: int


def _format_utc_z(ns: int) -> str:
    dt = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def iter_overlapping_windows(
    pcm_int16_1d: np.ndarray,
    sample_rate_hz: int,
    window_seconds: float = 20.0,
    overlap_fraction: float = 0.5,
    clock_start_ns: int | None = None,
) -> Iterator[AudioWindow]:
    """Yield overlapping windows along 1D int16 PCM.

    Windows start at sample indices ``0, hop, 2*hop, ...`` where
    ``hop = int((1 - overlap_fraction) * window_samples)``.

    The default ``window_seconds=20.0`` (with the default 0.5 overlap → 10s hop)
    is tuned for ``audfprint`` recall; ``live105sux`` used 10s. Override per
    deployment as needed.

    **Partial tail:** if fewer than ``window_samples`` samples remain after a
    start index, that remainder is **dropped** (not emitted).
    """
    if pcm_int16_1d.dtype != np.int16:
        pcm_int16_1d = pcm_int16_1d.astype(np.int16, copy=False)
    if pcm_int16_1d.ndim != 1:
        raise ValueError("pcm_int16_1d must be 1-D")

    if clock_start_ns is None:
        clock_start_ns = time.time_ns()

    window_samples = int(window_seconds * sample_rate_hz)
    hop_samples = int((1.0 - overlap_fraction) * window_samples)
    if window_samples <= 0:
        raise ValueError("window too short for sample rate")
    if hop_samples <= 0:
        raise ValueError("overlap_fraction must leave a positive hop")

    n = int(pcm_int16_1d.shape[0])
    i = 0
    while True:
        start = i * hop_samples
        if start + window_samples > n:
            break
        chunk = pcm_int16_1d[start : start + window_samples].copy()
        extra_ns = int(1e9 * start / sample_rate_hz)
        ts_ns = clock_start_ns + extra_ns
        yield AudioWindow(
            samples=chunk,
            sample_rate_hz=sample_rate_hz,
            window_start_utc=_format_utc_z(ts_ns),
            frame_count=window_samples,
        )
        i += 1
