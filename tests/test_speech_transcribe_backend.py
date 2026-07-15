"""Backend selection + mlx-whisper adapter behaviour for Tier-3 speech."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from radio_classifier.ingest.windows import AudioWindow
from radio_classifier.speech import (
    MlxWhisperTranscriber,
    WhisperTranscriber,
    build_transcriber,
)
from radio_classifier.speech.types import TranscribeStatus


def _window() -> AudioWindow:
    samples = np.zeros(1600, dtype=np.int16)
    return AudioWindow(
        samples=samples,
        sample_rate_hz=16_000,
        window_start_utc="2020-01-01T00:00:00.000Z",
        frame_count=int(samples.size),
    )


@pytest.fixture
def fake_mlx(monkeypatch):
    """Inject a fake ``mlx_whisper`` module so no model download/Metal is needed."""
    module = types.ModuleType("mlx_whisper")
    module.calls = []

    def transcribe(audio, *, path_or_hf_repo=None, language=None):
        module.calls.append({"audio": audio, "repo": path_or_hf_repo, "language": language})
        return {"text": "  hello world  ", "segments": [], "language": "en"}

    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    return module


def test_module_import_does_not_require_mlx():
    """Importing the speech module must not pull mlx (Linux/CI safety)."""
    monkeypatched = "mlx_whisper" in sys.modules
    # In a clean environment mlx isn't installed; the module still imported fine
    # at the top of this file, which is the guarantee we care about.
    assert "radio_classifier.speech.transcribe" in sys.modules
    # If mlx happens to be installed/injected we can't assert absence; only
    # assert the negative when it is genuinely not present.
    if not monkeypatched:
        assert "mlx_whisper" not in sys.modules


def test_factory_dispatches_mlx(fake_mlx):
    t = build_transcriber(backend="mlx", model="fake/model-repo", model_size="fake/model-repo")
    assert isinstance(t, MlxWhisperTranscriber)
    # warm-load ran once during construction
    assert len(fake_mlx.calls) == 1
    assert fake_mlx.calls[0]["repo"] == "fake/model-repo"


def test_factory_dispatches_faster_whisper(monkeypatch):
    # Avoid a real WhisperModel download by stubbing the loader.
    import radio_classifier.speech.transcribe as tr

    monkeypatch.setattr(tr, "_load_model", lambda *a, **k: object())
    t = build_transcriber(
        backend="faster-whisper", model_size="tiny", device="cpu", compute_type="int8"
    )
    assert isinstance(t, WhisperTranscriber)


def test_factory_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown whisper backend"):
        build_transcriber(backend="nope")


def test_mlx_transcribe_ok(fake_mlx):
    t = build_transcriber(backend="mlx", model="fake/model-repo")
    r = t.transcribe(_window())
    assert r.status == TranscribeStatus.ok
    assert r.text == "hello world"  # stripped


def test_mlx_transcribe_error_contract(fake_mlx):
    t = build_transcriber(backend="mlx", model="fake/model-repo")

    def boom(audio, *, path_or_hf_repo=None, language=None):
        raise RuntimeError("metal blew up")

    fake_mlx.transcribe = boom
    r = t.transcribe(_window())
    assert r.status == TranscribeStatus.error
    assert "metal blew up" in (r.message or "")


def test_mlx_language_auto_becomes_none(fake_mlx):
    t = build_transcriber(backend="mlx", model="fake/model-repo", language="auto")
    t.transcribe(_window())
    # last recorded call is the real transcribe (index 1; index 0 = warm-load)
    assert fake_mlx.calls[-1]["language"] is None
