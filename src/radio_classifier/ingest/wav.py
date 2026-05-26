"""Read mono 16-bit PCM from WAV files (stdlib :mod:`wave`)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_mono_s16le_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load **mono**, **16-bit PCM** WAV into int16 samples and return ``(pcm, sample_rate)``."""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1:
            raise ValueError(f"expected mono WAV, got nchannels={wf.getnchannels()}")
        if wf.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM (sampwidth=2), got {wf.getsampwidth()}")
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    pcm = np.frombuffer(raw, dtype="<i2")
    return pcm, int(rate)


def write_mono_s16le_wav(path: Path, pcm: np.ndarray, sample_rate_hz: int) -> None:
    """Write mono int16 PCM to a WAV file (utility for capture / testing)."""
    if pcm.dtype != np.int16:
        raise ValueError("pcm must be int16")
    if pcm.ndim != 1:
        raise ValueError("pcm must be 1-D")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm.tobytes())
