"""faster-whisper transcription for :class:`~radio_classifier.ingest.windows.AudioWindow`."""

from __future__ import annotations

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
