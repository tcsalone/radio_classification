"""Behavioural tests for the macOS continuous capture/classify orchestration.

Covers the phased-24h hardening: bounded per-block wait (no infinite hang on a
missing sidecar) and immediate per-block WAV pruning after a successful classify.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_REL = "macos/scripts/continuous_capture_blocks.sh"


def _install_script(workdir: Path) -> None:
    (workdir / ".venv" / "bin").mkdir(parents=True)
    stub = _write_stub_python(workdir / "_stubs")
    (workdir / ".venv" / "bin" / "python").symlink_to(stub.resolve())
    (workdir / "macos" / "scripts").mkdir(parents=True)
    (workdir / "macos" / "lib").mkdir(parents=True)
    shutil.copy(REPO_ROOT / SCRIPT_REL, workdir / SCRIPT_REL)
    shutil.copy(REPO_ROOT / "macos" / "env.defaults", workdir / "macos" / "env.defaults")
    shutil.copy(REPO_ROOT / "macos" / "lib" / "file_size.sh", workdir / "macos" / "lib" / "file_size.sh")


def _script_env(workdir: Path, **extra: str) -> dict[str, str]:
    env = {
        **os.environ,
        "PATH": f"{workdir / '_stubs'}:{os.environ['PATH']}",
        "BLOCK_SECONDS": "1",
    }
    env.update(extra)
    return env


def _run_script(workdir: Path, *argv: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", SCRIPT_REL, *argv],
        cwd=workdir,
        env=env or _script_env(workdir),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _write_stub_python(stub_dir: Path) -> Path:
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python"
    real_python = sys.executable
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            # Let the script's JSON metadata helper run as real Python.
            if [[ "${{1:-}}" == "-" ]]; then
              exec "{real_python}" "$@"
            fi

            SUBCMD=""; CAPTURE_SUBCMD=""; OUT_DIR=""; RUN_ID=""; DB_PATH=""
            args=("$@")
            for ((i=0; i<$#; i++)); do
              if [[ "${{args[$i]}}" == "radio_classifier" ]]; then
                SUBCMD="${{args[$((i+1))]:-}}"; CAPTURE_SUBCMD="${{args[$((i+2))]:-}}"
              fi
              [[ "${{args[$i]}}" == "--out-dir" ]] && OUT_DIR="${{args[$((i+1))]}}"
              [[ "${{args[$i]}}" == "--run-id" ]] && RUN_ID="${{args[$((i+1))]}}"
              [[ "${{args[$i]}}" == "--db-path" ]] && DB_PATH="${{args[$((i+1))]}}"
            done

            if [[ "$SUBCMD" == "db" ]]; then
              mkdir -p "$(dirname "$DB_PATH")"; : > "$DB_PATH"; exit 0
            fi
            if [[ "$SUBCMD" == "runs" ]]; then
              [[ "${{args[*]}}" == *" start "* ]] && echo "123"
              exit 0
            fi
            if [[ "$SUBCMD" == "capture" && "$CAPTURE_SUBCMD" == "chunks" ]]; then
              mkdir -p "$OUT_DIR"
              for i in ${{STUB_CAPTURE_BLOCKS:-1 2}}; do
                block="$(printf '%s_block%04d' "$RUN_ID" "$i")"
                wav="$OUT_DIR/$block.wav"; json="$OUT_DIR/$block.json"
                printf 'STUBWAV' > "$wav"
                cat > "$json" <<JSON
            {{"block_index": $i, "capture_start_utc": "2020-01-01T00:00:0$((i-1)).000Z", "complete": true, "wav_path": "$wav"}}
            JSON
              done
              sleep "${{STUB_CAPTURE_SLEEP:-0}}"
              exit 0
            fi

            # classify, reports, songs discovered, runs end all succeed.
            echo "stub:$SUBCMD ${{args[*]}}"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_dir(workdir: Path) -> Path:
    caps = workdir / "data" / "captures"
    subdirs = [p for p in caps.iterdir() if p.is_dir()]
    assert len(subdirs) == 1, f"expected one run dir, got {subdirs}"
    return subdirs[0]


def test_happy_path_classifies_and_prunes(tmp_path: Path) -> None:
    _install_script(tmp_path)
    proc = _run_script(tmp_path, "2")

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert "blocks classified: 2/2" in proc.stdout
    assert "capture_run_id=123" in proc.stdout
    # Immediate pruning: classified WAVs are gone, JSON sidecars retained.
    run_dir = _run_dir(tmp_path)
    wavs = list(run_dir.glob("*.wav"))
    jsons = list(run_dir.glob("*.json"))
    assert wavs == [], f"expected WAVs pruned, found {wavs}"
    assert len(jsons) == 2, f"expected 2 sidecars retained, found {jsons}"
    assert "pruned classified wav" in proc.stdout


def test_missing_sidecar_times_out_without_hanging(tmp_path: Path) -> None:
    """A stalled capture (block 2 never written) must not hang the run."""
    _install_script(tmp_path)
    # Write only block 1, then keep the capture process alive (sleeping) so the
    # consumer hits the per-block DEADLINE rather than the process-ended path.
    env = _script_env(tmp_path, STUB_CAPTURE_BLOCKS="1", STUB_CAPTURE_SLEEP="6")
    proc = _run_script(tmp_path, "2", env=env)

    # Block 1 classified (COMPLETED>0 => exit 0); block 2 skipped via timeout.
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert "sidecar not ready within" in proc.stderr
    assert "blocks classified: 1/2" in proc.stdout
    assert "blocks with classify failure: 1" in proc.stdout
