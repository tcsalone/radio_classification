"""Tests for Tier 3 speech pipeline routing and cheap pre-gates."""

from __future__ import annotations

import numpy as np
import pytest

from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.segments.types import BroadcastCategory
from radio_classifier.speech.json_schema import LlmClassificationJson
from radio_classifier.speech.ollama import OllamaSpeechClassifier
from radio_classifier.speech.pipeline import run_speech_pipeline
from radio_classifier.speech.transcribe import WhisperTranscriber
from radio_classifier.speech.types import SpeechPipelineStatus, SpeechTranscriptResult, TranscribeStatus


def _window(samples: np.ndarray) -> AudioWindow:
    return AudioWindow(
        samples=samples.astype(np.int16),
        sample_rate_hz=16_000,
        window_start_utc="2020-01-01T00:00:00.000Z",
        frame_count=int(samples.size),
    )


class FakeTranscriber:
    def __init__(self, text: str = "you're listening to 105.3") -> None:
        self.calls = 0
        self.text = text

    def transcribe(self, window: AudioWindow) -> SpeechTranscriptResult:
        self.calls += 1
        return SpeechTranscriptResult(
            status=TranscribeStatus.ok,
            window_start_utc=window.window_start_utc,
            text=self.text,
        )


class FakeClassifier:
    def __init__(self, category: str = "STATION") -> None:
        self.calls = 0
        self.category = category

    def classify_transcript(self, text: str) -> LlmClassificationJson:
        self.calls += 1
        return LlmClassificationJson.model_validate(
            {
                "class": self.category,
                "brand": "105.3" if self.category == "STATION" else None,
                "brand_mentions": (
                    [{"name": "105.3", "type": "tag"}]
                    if self.category == "STATION"
                    else []
                ),
                "confidence": 0.9,
                "rationale": "test classification",
            }
        )


class FakeSegment:
    text = " You're listening to Live 105. "


class FakeWhisperModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def transcribe(self, path: str, **kwargs: object):
        self.kwargs = kwargs
        return [FakeSegment()], object()


class CachedClassifier(OllamaSpeechClassifier):
    def __init__(self) -> None:
        super().__init__(base_url="http://127.0.0.1:9", model="test")
        self.calls = 0

    def _classify_once(self, text: str) -> LlmClassificationJson:
        self.calls += 1
        return LlmClassificationJson.model_validate(
            {
                "class": "DJ",
                "brand": None,
                "brand_mentions": [],
                "confidence": 0.8,
                "rationale": "test",
            }
        )


def test_speech_pipeline_skips_below_rms_gate() -> None:
    transcriber = FakeTranscriber()
    classifier = FakeClassifier()

    result = run_speech_pipeline(
        _window(np.zeros(16_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.skipped
    assert result.category is None
    assert "below speech gate" in (result.message or "")
    assert transcriber.calls == 0
    assert classifier.calls == 0


def test_speech_pipeline_runs_above_rms_gate() -> None:
    transcriber = FakeTranscriber()
    classifier = FakeClassifier()

    result = run_speech_pipeline(
        _window(np.full(16_000, 2_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.ok
    assert result.category is BroadcastCategory.STATION
    assert result.brand == "105.3"
    assert transcriber.calls == 1
    assert classifier.calls == 1


def test_speech_pipeline_skips_known_whisper_hallucination_before_llm() -> None:
    transcriber = FakeTranscriber("Thanks for watching!")
    classifier = FakeClassifier("DJ")

    result = run_speech_pipeline(
        _window(np.full(16_000, 2_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.skipped
    assert "known whisper hallucination" in (result.message or "")
    assert transcriber.calls == 1
    assert classifier.calls == 0


def test_speech_pipeline_skips_see_you_in_the_next_video_hallucination() -> None:
    """Regression: real Live 105 station audio was being transcribed as
    'See you in the next video.' and getting classified as STATION.
    The hallucination filter should drop these before they reach the LLM."""
    transcriber = FakeTranscriber("See you in the next video.")
    classifier = FakeClassifier("STATION")

    result = run_speech_pipeline(
        _window(np.full(16_000, 2_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.skipped
    assert "see you in the next video" in (result.message or "")
    assert classifier.calls == 0


def test_speech_pipeline_skips_short_transcript_before_llm() -> None:
    transcriber = FakeTranscriber("From the start")
    classifier = FakeClassifier("DJ")

    result = run_speech_pipeline(
        _window(np.full(16_000, 2_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.skipped
    assert "short transcript" in (result.message or "")
    assert transcriber.calls == 1
    assert classifier.calls == 0


def test_speech_pipeline_preserves_commercial_like_transcript() -> None:
    transcriber = FakeTranscriber("Toyota of downtown has zero percent financing this weekend")
    classifier = FakeClassifier("COMMERCIAL")

    result = run_speech_pipeline(
        _window(np.full(16_000, 2_000, dtype=np.int16)),
        transcriber=transcriber,  # type: ignore[arg-type]
        ollama_classifier=classifier,  # type: ignore[arg-type]
        min_rms=750.0,
    )

    assert result.status is SpeechPipelineStatus.ok
    assert result.category is BroadcastCategory.COMMERCIAL
    assert transcriber.calls == 1
    assert classifier.calls == 1


def test_whisper_transcriber_can_pass_speed_tuning_options(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeWhisperModel()
    monkeypatch.setattr(
        "radio_classifier.speech.transcribe._load_model",
        lambda *args, **kwargs: model,
    )

    transcriber = WhisperTranscriber(beam_size=1, vad_filter=True)
    result = transcriber.transcribe(_window(np.full(16_000, 2_000, dtype=np.int16)))

    assert result.status is TranscribeStatus.ok
    assert result.text == "You're listening to Live 105."
    assert model.kwargs["beam_size"] == 1
    assert model.kwargs["vad_filter"] is True


def test_ollama_classifier_caches_normalized_duplicate_transcripts() -> None:
    classifier = CachedClassifier()

    first = classifier.classify_transcript("Law Tigers has tickets")
    second = classifier.classify_transcript(" law   tigers HAS tickets ")

    assert first is second
    assert classifier.calls == 1
