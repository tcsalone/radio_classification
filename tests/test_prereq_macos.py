"""Darwin-specific prereq-check behaviour."""

from __future__ import annotations

from unittest import mock

from radio_classifier import prereq


def test_run_checks_macos_skips_gpu():
    with mock.patch.object(prereq, "is_macos", return_value=True):
        results = prereq.run_checks(with_gpu=True, with_ollama=False)
    names = [r.name for r in results]
    assert "macOS standalone stack" in names
    assert "NVIDIA GPU checks" in names
    assert "nvidia-smi available" not in names
    assert "ctranslate2 CUDA" not in names


def test_rtl_fm_hint_macos():
    with mock.patch.object(prereq.shutil, "which", return_value=None):
        with mock.patch.object(prereq, "is_macos", return_value=True):
            r = prereq.check_rtl_fm_present()
    assert r.ok is False
    assert "brew install librtlsdr" in r.detail


def test_rtl_fm_hint_linux():
    with mock.patch.object(prereq.shutil, "which", return_value=None):
        with mock.patch.object(prereq, "is_macos", return_value=False):
            r = prereq.check_rtl_fm_present()
    assert r.ok is False
    assert "apt install rtl-sdr" in r.detail
