"""Tests for platform helpers."""

from __future__ import annotations

import sys
from unittest import mock

from radio_classifier import platform as plat


def test_is_linux_on_linux():
    with mock.patch.object(sys, "platform", "linux"):
        assert plat.is_linux() is True
        assert plat.is_macos() is False


def test_is_macos_on_darwin():
    with mock.patch.object(sys, "platform", "darwin"):
        assert plat.is_macos() is True
        assert plat.is_linux() is False


def test_default_ollama_host_from_env(monkeypatch):
    monkeypatch.setenv("RADIO_CLASSIFIER_OLLAMA_HOST", "http://127.0.0.1:11435")
    assert plat.default_ollama_host() == "http://127.0.0.1:11435"


def test_default_ollama_host_fallback(monkeypatch):
    monkeypatch.delenv("RADIO_CLASSIFIER_OLLAMA_HOST", raising=False)
    assert plat.default_ollama_host() == "http://127.0.0.1:11434"


def test_default_whisper_device_macos():
    with mock.patch.object(plat, "is_macos", return_value=True):
        assert plat.default_whisper_device() == "cpu"


def test_default_whisper_device_linux():
    with mock.patch.object(plat, "is_macos", return_value=False):
        assert plat.default_whisper_device() == "cuda"
