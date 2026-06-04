"""Ollama client request-shaping: keep_alive pinning + env override."""

from __future__ import annotations

import io
import json

import pytest

from radio_classifier.speech import ollama as ollama_mod
from radio_classifier.speech.ollama import OllamaSpeechClassifier


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _patch_urlopen(monkeypatch) -> list[dict]:
    """Capture each request body posted to Ollama; return the captured list."""
    captured: list[dict] = []
    reply = json.dumps(
        {
            "message": {
                "content": json.dumps(
                    {"class": "COMMERCIAL", "brand": "Acme", "confidence": 0.9}
                )
            }
        }
    ).encode("utf-8")

    def fake_urlopen(req, timeout=None):  # noqa: ANN001
        captured.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(reply)

    monkeypatch.setattr(ollama_mod.urllib.request, "urlopen", fake_urlopen)
    return captured


def test_request_pins_model_resident_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE", raising=False)
    captured = _patch_urlopen(monkeypatch)

    clf = OllamaSpeechClassifier(base_url="http://127.0.0.1:9", model="test")
    clf.classify_transcript("save big at acme today")

    assert captured, "no request was sent"
    # -1 keeps the model loaded indefinitely (no 5-minute unload).
    assert captured[0]["keep_alive"] == -1


def test_keep_alive_env_override_accepts_duration_string(monkeypatch) -> None:
    monkeypatch.setenv("RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE", "30m")
    captured = _patch_urlopen(monkeypatch)

    clf = OllamaSpeechClassifier(base_url="http://127.0.0.1:9", model="test")
    clf.classify_transcript("another ad read")

    assert captured[0]["keep_alive"] == "30m"


def test_explicit_keep_alive_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE", "0")
    captured = _patch_urlopen(monkeypatch)

    clf = OllamaSpeechClassifier(base_url="http://127.0.0.1:9", model="test", keep_alive=-1)
    clf.classify_transcript("one more spot")

    assert captured[0]["keep_alive"] == -1


@pytest.mark.parametrize("raw,expected", [("-1", -1), ("0", 0), ("600", 600), ("15m", "15m")])
def test_default_keep_alive_parsing(monkeypatch, raw, expected) -> None:
    monkeypatch.setenv("RADIO_CLASSIFIER_OLLAMA_KEEP_ALIVE", raw)
    assert ollama_mod._default_keep_alive() == expected
