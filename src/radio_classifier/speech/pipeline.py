"""Transcribe then classify with Ollama; 5-class output."""

from __future__ import annotations

from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.brands import canonicalize_brand
from radio_classifier.speech.json_schema import speech_kind_from_llm
from radio_classifier.speech.ollama import OllamaClassificationError, OllamaSpeechClassifier
from radio_classifier.speech.transcribe import WhisperTranscriber
from radio_classifier.speech.types import (
    SpeechPipelineResult,
    SpeechPipelineStatus,
    TranscribeStatus,
)

_MIN_TRANSCRIPT_WORDS = 4
# Phrases Whisper hallucinates onto silence / music / station-ID audio because
# its training data included a lot of YouTube outros. Match in lowercase
# substring form so the surrounding punctuation does not affect detection.
_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "subscribe to our channel",
    "see you next time",
    "see you in the next video",
    "see you in the next one",
    "see you next video",
    "see you in the next",
    "see you next",
)


def run_speech_pipeline(
    window: AudioWindow,
    *,
    transcriber: WhisperTranscriber,
    ollama_classifier: OllamaSpeechClassifier,
    min_rms: float = 750.0,
) -> SpeechPipelineResult:
    """Run faster-whisper then Ollama 5-class JSON classification."""
    rms = _window_rms(window)
    if rms < min_rms:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.skipped,
            transcript="",
            message=f"rms {rms:.1f} below speech gate {min_rms:.1f}",
        )

    tr = transcriber.transcribe(window)
    if tr.status != TranscribeStatus.ok:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.error,
            transcript=tr.text,
            message=tr.message or "transcription failed",
        )
    pre_gate = _transcript_quality_message(tr.text)
    if pre_gate is not None:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.skipped,
            transcript=tr.text,
            message=pre_gate,
        )
    try:
        llm = ollama_classifier.classify_transcript(tr.text)
    except OllamaClassificationError as exc:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.error,
            transcript=tr.text,
            message=str(exc),
        )
    cat, brand, mentions, sig = speech_kind_from_llm(llm)
    brand = canonicalize_brand(brand)
    for mention in mentions:
        normalized = canonicalize_brand(mention.name)
        if normalized is not None:
            mention.name = normalized
    post_gate = _classification_quality_message(tr.text, cat.value)
    if post_gate is not None:
        return SpeechPipelineResult(
            window_start_utc=window.window_start_utc,
            status=SpeechPipelineStatus.skipped,
            transcript=tr.text,
            confidence=llm.confidence,
            rationale=llm.rationale,
            message=post_gate,
        )
    return SpeechPipelineResult(
        window_start_utc=window.window_start_utc,
        status=SpeechPipelineStatus.ok,
        transcript=tr.text,
        category=cat,
        brand=brand,
        brand_mentions=mentions,
        commercial_signature=sig,
        confidence=llm.confidence,
        rationale=llm.rationale,
    )


def _window_rms(window: AudioWindow) -> float:
    """Return PCM RMS in int16 sample units."""
    if window.samples.size == 0:
        return 0.0
    # ``float32`` is plenty for an int16 RMS gate and avoids int16 overflow.
    return float((window.samples.astype("float32") ** 2).mean() ** 0.5)


def _transcript_quality_message(text: str) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return "empty transcript"
    hallucination = _matched_hallucination(normalized)
    if hallucination is not None:
        return f"known whisper hallucination: {hallucination!r}"
    words = _word_count(normalized)
    if words < _MIN_TRANSCRIPT_WORDS:
        return f"short transcript ({words} words)"
    return None


def _classification_quality_message(text: str, category: str) -> str | None:
    """Suppress fragile DJ/STATION rows while preserving commercial recall."""
    if category not in {"DJ", "STATION"}:
        return None
    normalized = _normalize_text(text)
    hallucination = _matched_hallucination(normalized)
    if hallucination is not None:
        return f"known whisper hallucination: {hallucination!r}"
    words = _word_count(normalized)
    if words < _MIN_TRANSCRIPT_WORDS:
        return f"short {category} transcript ({words} words)"
    return None


def _matched_hallucination(normalized: str) -> str | None:
    for phrase in _HALLUCINATION_PHRASES:
        if phrase in normalized:
            return phrase
    return None


def _word_count(normalized: str) -> int:
    return len(normalized.split())


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())
