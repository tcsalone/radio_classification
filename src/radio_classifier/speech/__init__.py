"""Tier 3 — faster-whisper transcription and Ollama 5-class classification."""

from radio_classifier.speech.json_schema import (
    LlmClassificationJson,
    speech_kind_from_llm,
)
from radio_classifier.speech.logging_fmt import format_speech_human, format_speech_json
from radio_classifier.speech.ollama import (
    OllamaClassificationError,
    OllamaSpeechClassifier,
)
from radio_classifier.speech.pipeline import run_speech_pipeline
from radio_classifier.speech.transcribe import WhisperTranscriber, transcribe_window
from radio_classifier.speech.types import (
    BrandMention,
    CommercialSignature,
    SpeechPipelineResult,
    SpeechPipelineStatus,
    SpeechTranscriptResult,
    TranscribeStatus,
)

__all__ = [
    "BrandMention",
    "CommercialSignature",
    "LlmClassificationJson",
    "OllamaClassificationError",
    "OllamaSpeechClassifier",
    "SpeechPipelineResult",
    "SpeechPipelineStatus",
    "SpeechTranscriptResult",
    "TranscribeStatus",
    "WhisperTranscriber",
    "format_speech_human",
    "format_speech_json",
    "run_speech_pipeline",
    "speech_kind_from_llm",
    "transcribe_window",
]
