"""Auto-discover and pre-load NVIDIA CUDA shared libraries shipped via pip wheels.

Two consumers in this project need CUDA user-space libraries at runtime:

* **CTranslate2 / faster-whisper** (Tier 3) needs ``libcublas.so.12``,
  ``libcudart.so.12`` and ``libcudnn.so.*``.
* **TensorFlow / YAMNet** (Tier 2) additionally requires ``libcufft.so.11``,
  ``libcusolver.so.11``, and ``libcusparse.so.12``.

On WSL2 the Windows-side driver makes the device visible but the Linux
user-space libraries are not installed by default. Operators install the
``[gpu]`` extra which pulls in all of the above via the matching
``nvidia-*-cu12`` wheels.

Those wheels drop their ``.so`` files into
``<site-packages>/nvidia/<libname>/lib/`` which is NOT on the dynamic
linker's search path. Rather than ask operators to set ``LD_LIBRARY_PATH``
correctly (which is brittle: ``__file__`` on namespace sub-packages can be
``None``), this module ``dlopen()``s the libraries with ``RTLD_GLOBAL`` so
later imports of ``faster_whisper`` / ``ctranslate2`` / ``tensorflow``
resolve them.

Call :func:`preload_nvidia_libs` **before** the first ``WhisperModel(...)``
construction *and* before the first ``tensorflow``/``tensorflow_hub`` import.
It is a no-op when the wheels are not installed.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Order matters: cudart and cublas come before cudnn / cufft / cusolver /
# cusparse so the latter can resolve their dependencies on cublas+cudart
# through the already-loaded global symbol table.
_PRELOAD_PACKAGES: tuple[str, ...] = (
    "nvidia.cuda_runtime.lib",
    "nvidia.cublas.lib",
    "nvidia.cuda_nvrtc.lib",
    "nvidia.cudnn.lib",
    "nvidia.cufft.lib",
    "nvidia.cusolver.lib",
    "nvidia.cusparse.lib",
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
