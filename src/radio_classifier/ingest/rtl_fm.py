"""Spawn ``rtl_fm`` and read raw PCM bytes from stdout (s16le mono; rate from ``-r``)."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import IO


# FM broadcast center for product default (105.3 MHz). Override via CLI.
DEFAULT_FREQUENCY_HZ = 105_300_000


class RtlFmExitedError(RuntimeError):
    """Raised when ``rtl_fm`` exits with a non-zero status or stderr indicates failure."""

    def __init__(self, returncode: int, stderr_text: str = "") -> None:
        self.returncode = returncode
        msg = f"rtl_fm exited with code {returncode}"
        if stderr_text.strip():
            msg += f": {stderr_text.strip()[:500]}"
        super().__init__(msg)


def default_rtl_fm_argv(
    frequency_hz: float = DEFAULT_FREQUENCY_HZ,
    device_index: int = 0,
    sample_rate_hz: int = 48_000,
) -> list[str]:
    """Build a reference ``rtl_fm`` argument list.

    Expected output: **mono signed 16-bit PCM** little-endian at ``sample_rate_hz``
    (``-r``), written to stdout when the last argument is ``-``.

    Exact RF parameters (-s, AGC, etc.) are operator-tunable; this is a default.
    """
    return [
        "rtl_fm",
        "-d",
        str(device_index),
        "-f",
        str(int(frequency_hz)),
        "-M",
        "wfm",
        "-s",
        "200k",
        "-r",
        str(sample_rate_hz),
        "-",
    ]


@dataclass
class RtlFmStream:
    """Run ``rtl_fm`` as a subprocess and iterate PCM bytes from stdout."""

    frequency_hz: float = DEFAULT_FREQUENCY_HZ
    device_index: int = 0
    sample_rate_hz: int = 48_000
    argv: list[str] | None = None
    chunk_size: int = 16_384

    _proc: subprocess.Popen[bytes] | None = None
    _ended_by_time_limit: bool = False

    def __post_init__(self) -> None:
        if self.argv is None:
            self.argv = default_rtl_fm_argv(
                self.frequency_hz,
                self.device_index,
                self.sample_rate_hz,
            )

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def iter_stdout_bytes(self, max_wall_seconds: float | None = None) -> Iterator[bytes]:
        """Yield chunks of raw PCM bytes until stdout ends; then wait and check return code.

        If ``max_wall_seconds`` is set, stop after that much wall-clock time and
        ``terminate()`` the child (no error if the process was stopped for this reason).
        """
        if self._proc is None:
            raise RuntimeError("RtlFmStream.start() must be called before reading")
        self._ended_by_time_limit = False
        stdout: IO[bytes] | None = self._proc.stdout
        assert stdout is not None
        t0 = time.monotonic()
        while True:
            if max_wall_seconds is not None and time.monotonic() - t0 >= max_wall_seconds:
                self._ended_by_time_limit = True
                self._proc.terminate()
                break
            chunk = stdout.read(self.chunk_size)
            if not chunk:
                break
            yield chunk
        stderr_data = b""
        stderr = self._proc.stderr
        if stderr is not None:
            stderr_data = stderr.read()
        rc = self._proc.wait()
        if rc != 0 and not self._ended_by_time_limit:
            raise RtlFmExitedError(rc, stderr_data.decode(errors="replace"))
