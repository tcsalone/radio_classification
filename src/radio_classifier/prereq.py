"""Pre-flight checks for the heavy runtime stack.

Surfaced via ``radio-classifier prereq-check``. Exits non-zero if any check
fails so that an operator can wire this into shell scripts.

Each check is intentionally independent: missing GPU or missing Ollama
does not crash the others — the script reports each result and the overall
verdict.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path

from radio_classifier.platform import is_macos


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def check_rtl_fm_present() -> CheckResult:
    if shutil.which("rtl_fm"):
        return CheckResult("rtl_fm on PATH", True, "")
    if is_macos():
        return CheckResult(
            "rtl_fm on PATH",
            False,
            "rtl_fm not found; brew install librtlsdr",
        )
    return CheckResult("rtl_fm on PATH", False, "rtl_fm not found; apt install rtl-sdr")


def check_audfprint_present() -> CheckResult:
    override = os.environ.get("RADIO_CLASSIFIER_AUDFPRINT_BIN")
    if override:
        argv = [
            os.path.expanduser(os.path.expandvars(t)) for t in shlex.split(override)
        ]
        try:
            proc = subprocess.run(
                [*argv, "--help"], capture_output=True, text=True
            )
        except FileNotFoundError as exc:
            return CheckResult(
                "audfprint external CLI",
                False,
                f"RADIO_CLASSIFIER_AUDFPRINT_BIN cannot be executed: {exc}",
            )
        if proc.returncode == 0:
            return CheckResult(
                "audfprint external CLI",
                True,
                f"RADIO_CLASSIFIER_AUDFPRINT_BIN={' '.join(argv)}",
            )
        return CheckResult(
            "audfprint external CLI",
            False,
            f"RADIO_CLASSIFIER_AUDFPRINT_BIN failed: {(proc.stderr or proc.stdout).strip()[:200]}",
        )
    if shutil.which("audfprint"):
        return CheckResult("audfprint on PATH", True, "")
    local_clone = Path("~/dev/audfprint/audfprint.py").expanduser()
    if local_clone.is_file():
        rc = subprocess.run(
            [sys.executable, str(local_clone), "--help"],
            capture_output=True,
            text=True,
        ).returncode
        if rc == 0:
            return CheckResult(
                "audfprint local clone",
                True,
                f"via {sys.executable} {local_clone}",
            )
    # Fall back to python -m audfprint
    rc = subprocess.run(
        [sys.executable, "-m", "audfprint", "--help"],
        capture_output=True,
        text=True,
    ).returncode
    if rc == 0:
        return CheckResult("audfprint module", True, "via python -m audfprint")
    return CheckResult(
        "audfprint module",
        False,
        "neither `audfprint` on PATH nor `python -m audfprint` importable",
    )


def check_nvidia_smi() -> CheckResult:
    if not shutil.which("nvidia-smi"):
        return CheckResult("nvidia-smi available", False, "nvidia-smi not on PATH inside WSL")
    proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if proc.returncode != 0:
        return CheckResult("nvidia-smi available", False, proc.stderr.strip()[:200])
    summary = ""
    for line in proc.stdout.splitlines():
        if "GeForce" in line or "Driver Version" in line:
            summary = line.strip()
            break
    return CheckResult("nvidia-smi available", True, summary[:200])


def check_ctranslate2_cuda() -> CheckResult:
    try:
        import ctranslate2  # type: ignore
    except ImportError as e:
        return CheckResult("ctranslate2 import", False, str(e))
    try:
        n = ctranslate2.get_cuda_device_count()
    except Exception as e:  # noqa: BLE001
        return CheckResult("ctranslate2 CUDA", False, str(e))
    return CheckResult("ctranslate2 CUDA", n >= 1, f"cuda_device_count={n}")


def check_whisper_tiny_cuda_smoke() -> CheckResult:
    """Run a 0.5-second sine through tiny / cuda / float16 to verify the path."""
    try:
        import numpy as np

        from radio_classifier.gpu import preload_nvidia_libs

        preloaded = preload_nvidia_libs()
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        return CheckResult("faster-whisper GPU smoke", False, str(e))

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "smoke.wav"
        rate = 16_000
        seconds = 0.5
        t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
        sine = (np.sin(2 * np.pi * 440.0 * t) * 0.2 * 32767.0).astype("<i2")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(sine.tobytes())
        try:
            model = WhisperModel("tiny", device="cuda", compute_type="float16")
            segments, _info = model.transcribe(str(wav_path), language="en")
            for _ in segments:
                pass
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "libcublas" in msg or "libcudnn" in msg:
                if not preloaded:
                    msg += " | Fix: pip install -e '.[gpu]' to add NVIDIA CUDA wheels."
                else:
                    msg += (
                        " | NVIDIA wheels preloaded "
                        f"({len(preloaded)} libs) but CTranslate2 still cannot resolve "
                        "them. Check that nvidia-cublas-cu12 + nvidia-cudnn-cu12 are the "
                        "versions expected by your CTranslate2 build."
                    )
            return CheckResult("faster-whisper GPU smoke", False, msg[:500])
    detail = f"preloaded {len(preloaded)} NVIDIA libs" if preloaded else ""
    return CheckResult("faster-whisper GPU smoke", True, detail)


def check_mlx_whisper_smoke() -> CheckResult:
    """Run a 0.5-second sine through mlx-whisper (Apple Metal) to verify the path.

    Uses ``whisper-tiny`` so first-run download is small. Confirms the mlx
    backend is importable and can load + run a model on this machine.
    """
    try:
        import numpy as np

        import mlx_whisper  # type: ignore
    except ImportError as e:
        return CheckResult(
            "mlx-whisper smoke",
            False,
            f"{e} | Fix: pip install -e '.[mlx]' (Apple Silicon only).",
        )

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "smoke.wav"
        rate = 16_000
        seconds = 0.5
        t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
        sine = (np.sin(2 * np.pi * 440.0 * t) * 0.2 * 32767.0).astype("<i2")
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(sine.tobytes())
        try:
            mlx_whisper.transcribe(
                str(wav_path),
                path_or_hf_repo="mlx-community/whisper-tiny",
                language="en",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult("mlx-whisper smoke", False, str(e)[:500])
    return CheckResult("mlx-whisper smoke", True, "mlx-community/whisper-tiny on Metal")


def check_ollama_tags(base_url: str | None = None) -> CheckResult:
    raw = base_url or os.environ.get(
        "RADIO_CLASSIFIER_OLLAMA_HOST", "http://127.0.0.1:11434"
    )
    url = raw.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return CheckResult("ollama /api/tags", False, str(e))
    models = data.get("models", []) if isinstance(data, dict) else []
    return CheckResult("ollama /api/tags", True, f"models={len(models)}")


def check_macos_stack() -> CheckResult:
    """Report macOS standalone expectations (Metal Ollama, no NVIDIA CUDA wheels)."""
    if not is_macos():
        return CheckResult("macOS standalone stack", True, "not applicable on this OS")
    detail = (
        "Apple Silicon: Ollama.app on :11434, pip install without [gpu], "
        "source macos/env.defaults before capture"
    )
    return CheckResult("macOS standalone stack", True, detail)


def run_checks(*, with_gpu: bool, with_ollama: bool) -> list[CheckResult]:
    checks: list[CheckResult] = [
        check_rtl_fm_present(),
        check_audfprint_present(),
    ]
    if is_macos():
        checks.append(check_macos_stack())
        # On Apple Silicon the STT path is mlx-whisper on Metal (not CUDA).
        # Smoke-test it when that backend is selected so validate.sh catches a
        # broken mlx install before a long capture.
        if os.environ.get("WHISPER_BACKEND") == "mlx":
            checks.append(check_mlx_whisper_smoke())
        if with_gpu:
            checks.append(
                CheckResult(
                    "NVIDIA GPU checks",
                    True,
                    "skipped on macOS (use Ollama Metal; Whisper/YAMNet on CPU)",
                )
            )
    elif with_gpu:
        checks.append(check_nvidia_smi())
        checks.append(check_ctranslate2_cuda())
        checks.append(check_whisper_tiny_cuda_smoke())
    if with_ollama:
        checks.append(check_ollama_tags())
    return checks
