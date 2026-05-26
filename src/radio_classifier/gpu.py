"""Auto-discover and pre-load NVIDIA CUDA shared libraries shipped via pip wheels.

CTranslate2 / faster-whisper need ``libcublas.so.12`` and ``libcudnn.so.*`` at
runtime. On WSL2 the Windows-side driver makes the device visible but the
Linux user-space libraries are not installed by default. Operators install the
``[gpu]`` extra which pulls in:

* ``nvidia-cublas-cu12``
* ``nvidia-cudnn-cu12``
* ``nvidia-cuda-nvrtc-cu12`` (transitively from cublas)

…but those wheels drop their ``.so`` files into
``<site-packages>/nvidia/<libname>/lib/`` which is NOT on the dynamic
linker's search path. Rather than ask operators to set ``LD_LIBRARY_PATH``
correctly (which is brittle: ``__file__`` on namespace sub-packages can be
``None``), this module ``dlopen()``s the libraries with ``RTLD_GLOBAL`` so
later imports of ``faster_whisper`` / ``ctranslate2`` resolve them.

Call :func:`preload_nvidia_libs` **before** the first ``WhisperModel(...)``
construction. It is a no-op when the wheels are not installed.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Order matters: cublas before cudnn so cudnn can resolve its dependency on
# cublas through the already-loaded global symbol table.
_PRELOAD_PACKAGES: tuple[str, ...] = (
    "nvidia.cuda_runtime.lib",
    "nvidia.cublas.lib",
    "nvidia.cuda_nvrtc.lib",
    "nvidia.cudnn.lib",
)


def _wheel_lib_dir(import_path: str) -> Path | None:
    """Locate the directory of an installed NVIDIA wheel ``lib`` sub-package."""
    try:
        mod = importlib.import_module(import_path)
    except Exception:  # noqa: BLE001 — module may not be installed
        return None
    paths = getattr(mod, "__path__", None)
    if not paths:
        return None
    p = Path(next(iter(paths)))
    return p if p.is_dir() else None


def preload_nvidia_libs() -> list[Path]:
    """``dlopen`` NVIDIA CUDA shared libraries from installed pip wheels.

    Returns the list of ``.so`` paths successfully loaded. Safe to call when
    the ``[gpu]`` extra is not installed (returns an empty list).
    """
    loaded: list[Path] = []
    if sys.platform != "linux":
        return loaded
    for pkg in _PRELOAD_PACKAGES:
        directory = _wheel_lib_dir(pkg)
        if directory is None:
            continue
        for so in sorted(directory.glob("lib*.so*")):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                logger.debug("preload failed for %s: %s", so, exc)
                continue
            loaded.append(so)
    return loaded
