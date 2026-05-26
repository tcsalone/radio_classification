"""NVIDIA wheel auto-preloader."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from radio_classifier.gpu import _wheel_lib_dir, preload_nvidia_libs


def test_preload_returns_list_and_does_not_raise() -> None:
    """The preloader must be safe even when the [gpu] extra is uninstalled."""
    result = preload_nvidia_libs()
    assert isinstance(result, list)
    for p in result:
        assert isinstance(p, Path)


@pytest.mark.skipif(sys.platform != "linux", reason="preloader is a no-op off Linux")
def test_preload_finds_wheel_libs_when_installed() -> None:
    """If nvidia-cublas-cu12 is in the venv, the preloader picks up its .so files."""
    try:
        importlib.import_module("nvidia.cublas.lib")
    except Exception:
        pytest.skip("nvidia-cublas-cu12 not installed in this environment")
    loaded = preload_nvidia_libs()
    assert loaded, "expected at least one NVIDIA library to be dlopen()'d"
    so_names = {p.name for p in loaded}
    assert any(name.startswith("libcublas") for name in so_names)


def test_wheel_lib_dir_returns_none_for_missing_package() -> None:
    assert _wheel_lib_dir("definitely.not.a.real.nvidia.subpackage.xyz") is None
