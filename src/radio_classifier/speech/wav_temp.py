"""Write :class:`AudioWindow` to a temporary WAV file (Tier-1 + Tier-3 helper)."""

from __future__ import annotations

import os
import tempfile
import wave
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from radio_classifier.ingest.windows import AudioWindow


def write_window_to_temp_wav(window: AudioWindow) -> Path:
    """Write mono s16le PCM to a **named temp** ``.wav`` file. Caller must delete."""
    fd, path_str = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
    except OSError:
        pass
    path = Path(path_str)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(window.sample_rate_hz)
        wf.writeframes(window.samples.tobytes())
    return path


@contextmanager
def temp_wav_for_window(window: AudioWindow) -> Generator[Path, None, None]:
    """Context manager: write window to temp WAV and **unlink** after."""
    path = write_window_to_temp_wav(window)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
