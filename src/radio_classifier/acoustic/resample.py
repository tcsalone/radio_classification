"""Resample mono int16 PCM to a target sample rate (16 kHz for YAMNet/VAD).

Harvested from ``live105sux/src/live105sux/vad/resample.py``. We keep the
generalized name so other tiers (e.g. PANNs at 32 kHz) can reuse it later.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


_UP_DOWN_16K: dict[int, tuple[int, int]] = {
    8_000: (2, 1),
    32_000: (1, 2),
    44_100: (160, 441),
    48_000: (1, 3),
}


def resample_pcm_int16_to_16k(mono_int16: np.ndarray, src_rate: int) -> np.ndarray:
    """Resample mono int16 PCM to **16000 Hz** int16.

    Supported ``src_rate`` values: ``8000``, ``16000``, ``32000``, ``44100``,
    ``48000``. Other rates raise ``ValueError``.
    """
    if mono_int16.ndim != 1:
        raise ValueError("mono_int16 must be 1-D")
    if mono_int16.dtype != np.int16:
        raise ValueError("mono_int16 must be int16")
    if src_rate == 16_000:
        return mono_int16.copy()
    if src_rate not in _UP_DOWN_16K:
        raise ValueError(
            f"unsupported src_rate={src_rate}; supported: 8000, 16000, 32000, 44100, 48000"
        )
    up, down = _UP_DOWN_16K[src_rate]
    x = mono_int16.astype(np.float32)
    y = signal.resample_poly(x, up, down)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y


def pcm_int16_to_float32_normalized(pcm: np.ndarray) -> np.ndarray:
    """Convert mono int16 PCM to float32 in [-1.0, 1.0] (YAMNet input format)."""
    if pcm.dtype != np.int16:
        raise ValueError("pcm must be int16")
    return (pcm.astype(np.float32)) / 32768.0
