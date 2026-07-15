"""Whisper transcription for :class:`~radio_classifier.ingest.windows.AudioWindow`.

Two backends share one ``transcribe(window) -> SpeechTranscriptResult`` contract:

* :class:`WhisperTranscriber` — faster-whisper/CTranslate2 (CPU or CUDA). The
  Linux/NVIDIA path; unchanged from the original port.
* :class:`MlxWhisperTranscriber` — mlx-whisper on Apple Metal (GPU/ANE), used on
  Apple Silicon to run larger models (e.g. ``large-v3-turbo``) far faster than
  faster-whisper's CPU-only path on that hardware.

Use :func:`build_transcriber` to pick a backend from config.
"""

from __future__ import annotations

import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.speech.types import SpeechTranscriptResult, TranscribeStatus
from radio_classifier.speech.wav_temp import temp_wav_for_window


def _load_model(model_size: str, device: str, compute_type: str) -> Any:
    # Imported lazily so tests / unrelated CLI subcommands don't pay the cost.
    if device == "cuda":
        from radio_classifier.gpu import preload_nvidia_libs

        preload_nvidia_libs()
    from faster_whisper import WhisperModel  # type: ignore

    return WhisperModel(model_size, device=device, compute_type=compute_type)


class WhisperTranscriber:
    """Stateful wrapper that loads the model ONCE per process.

    Reuse across windows to avoid the ~3s reload cost on GPU.
    """

    def __init__(
        self,
        *,
        model_size: str = "medium.en",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        beam_size: int | None = None,
        vad_filter: bool = False,
    ) -> None:
        self.language = None if language in ("", "auto") else language
        self.beam_size = max(1, int(beam_size)) if beam_size is not None else None
        self.vad_filter = vad_filter
        self._model = _load_model(model_size, device, compute_type)

    def transcribe(self, window: AudioWindow) -> SpeechTranscriptResult:
        try:
            with temp_wav_for_window(window) as wav_path:
                kwargs: dict[str, object] = {"language": self.language}
                if self.beam_size is not None:
                    kwargs["beam_size"] = self.beam_size
                if self.vad_filter:
                    kwargs["vad_filter"] = True
                segments, _info = self._model.transcribe(str(wav_path), **kwargs)
                parts = [s.text for s in segments if s and s.text]
                text = " ".join(p.strip() for p in parts if p).strip()
                return SpeechTranscriptResult(
                    status=TranscribeStatus.ok,
                    window_start_utc=window.window_start_utc,
                    text=text,
                )
        except Exception as exc:  # noqa: BLE001 — surface any STT failure
            return SpeechTranscriptResult(
                status=TranscribeStatus.error,
                window_start_utc=window.window_start_utc,
                text="",
                message=str(exc),
            )


class MlxWhisperTranscriber:
    """Apple-Metal transcriber backed by ``mlx-whisper`` (Apple Silicon only).

    Same public contract as :class:`WhisperTranscriber` — construct once, reuse
    across windows. ``mlx_whisper`` caches the loaded model by ``path_or_hf_repo``
    (an internal lru), so a warm-load in ``__init__`` primes that cache and makes
    first-window latency (and any model download / auth failure) surface at
    startup instead of mid-run.

    ``device``/``compute_type`` are irrelevant on Metal and ignored.
    ``beam_size``/``vad_filter`` are not exposed by ``mlx_whisper.transcribe`` in
    the simple form; they are accepted for a uniform factory signature and ignored
    with a one-line warning. The upstream RMS speech-gate and hallucination
    suppression in :mod:`radio_classifier.speech.pipeline` are backend-agnostic
    and cover what faster-whisper's VAD would.
    """

    def __init__(
        self,
        *,
        model: str = "mlx-community/whisper-large-v3-turbo",
        language: str = "en",
        beam_size: int | None = None,
        vad_filter: bool = False,
        **_ignored: Any,
    ) -> None:
        self.model_ref = model
        self.language = None if language in ("", "auto") else language
        if beam_size is not None or vad_filter:
            print(
                "radio-classifier: mlx-whisper backend ignores --whisper-beam-size "
                "and --whisper-vad-filter (unsupported by mlx_whisper.transcribe).",
                file=sys.stderr,
            )
        # Imported lazily so Linux/CI and unrelated CLI subcommands never require
        # mlx (Apple Silicon only), mirroring the faster-whisper lazy import.
        import mlx_whisper  # type: ignore

        self._mlx = mlx_whisper
        self._warm()

    def _warm(self) -> None:
        """Prime mlx's model cache with a short silent clip (fail-fast on load)."""
        fd, path_str = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd)
        except OSError:
            pass
        path = Path(path_str)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence
            self._mlx.transcribe(
                str(path), path_or_hf_repo=self.model_ref, language=self.language
            )
        finally:
            path.unlink(missing_ok=True)

    def transcribe(self, window: AudioWindow) -> SpeechTranscriptResult:
        try:
            with temp_wav_for_window(window) as wav_path:
                result = self._mlx.transcribe(
                    str(wav_path),
                    path_or_hf_repo=self.model_ref,
                    language=self.language,
                )
                text = (result.get("text") or "").strip()
                return SpeechTranscriptResult(
                    status=TranscribeStatus.ok,
                    window_start_utc=window.window_start_utc,
                    text=text,
                )
        except Exception as exc:  # noqa: BLE001 — surface any STT failure
            return SpeechTranscriptResult(
                status=TranscribeStatus.error,
                window_start_utc=window.window_start_utc,
                text="",
                message=str(exc),
            )


def build_transcriber(
    *,
    backend: str = "faster-whisper",
    model_size: str = "medium.en",
    model: str | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    beam_size: int | None = None,
    vad_filter: bool = False,
) -> Any:
    """Construct the configured transcriber backend (loaded once, reused).

    ``model`` is the mlx HF repo / local path; when omitted it falls back to
    ``model_size`` so a single ``--whisper-model`` flag drives both backends.
    """
    if backend == "mlx":
        return MlxWhisperTranscriber(
            model=model or model_size,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    if backend == "faster-whisper":
        return WhisperTranscriber(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
    raise ValueError(f"unknown whisper backend: {backend!r}")


def transcribe_window(
    window: AudioWindow,
    *,
    model_size: str = "medium.en",
    device: str = "cuda",
    compute_type: str = "float16",
    language: str = "en",
    beam_size: int | None = None,
    vad_filter: bool = False,
) -> SpeechTranscriptResult:
    """One-shot transcription for offline / test paths.

    Production / long-running CLI should construct :class:`WhisperTranscriber`
    once and reuse it across windows.
    """
    t = WhisperTranscriber(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
    return t.transcribe(window)
