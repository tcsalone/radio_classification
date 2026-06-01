"""Behavioural smoke tests for continuous capture/chunk classify orchestration."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "continuous_capture_blocks.sh"


def _install_script(workdir: Path) -> None:
    (workdir / ".venv" / "bin").mkdir(parents=True)
    stub = _write_stub_python(workdir / "_stubs")
    (workdir / ".venv" / "bin" / "python").symlink_to(stub.resolve())
    (workdir / "scripts").mkdir()
    shutil.copy(SCRIPT_PATH, workdir / "scripts" / "continuous_capture_blocks.sh")


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
        ["bash", "scripts/continuous_capture_blocks.sh", *argv],
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

            SUBCMD=""
            CAPTURE_SUBCMD=""
            OUT_DIR=""
            RUN_ID=""
            DB_PATH=""
            args=("$@")
            for ((i=0; i<$#; i++)); do
              if [[ "${{args[$i]}}" == "radio_classifier" ]]; then
                SUBCMD="${{args[$((i+1))]:-}}"
                CAPTURE_SUBCMD="${{args[$((i+2))]:-}}"
              fi
              if [[ "${{args[$i]}}" == "--out-dir" ]]; then
                OUT_DIR="${{args[$((i+1))]}}"
              fi
              if [[ "${{args[$i]}}" == "--run-id" ]]; then
                RUN_ID="${{args[$((i+1))]}}"
              fi
              if [[ "${{args[$i]}}" == "--db-path" ]]; then
                DB_PATH="${{args[$((i+1))]}}"
              fi
            done

            if [[ "$SUBCMD" == "db" ]]; then
              mkdir -p "$(dirname "$DB_PATH")"
              : > "$DB_PATH"
              exit 0
            fi

            if [[ "$SUBCMD" == "capture" && "$CAPTURE_SUBCMD" == "chunks" ]]; then
              mkdir -p "$OUT_DIR"
              for i in 1 2; do
                block="$(printf '%s_block%04d' "$RUN_ID" "$i")"
                wav="$OUT_DIR/$block.wav"
                json="$OUT_DIR/$block.json"
                printf 'STUBWAV' > "$wav"
                cat > "$json" <<JSON
            {{"block_index": $i, "capture_start_utc": "2020-01-01T00:00:0$((i-1)).000Z", "complete": true, "wav_path": "$wav"}}
            JSON
              done
              exit 0
            fi

            # classify, reports, and songs discovered all succeed.
            echo "stub:$SUBCMD ${{args[*]}}"
            exit 0
            """
        ),
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def test_continuous_capture_script_classifies_completed_chunks(tmp_path: Path) -> None:
    _install_script(tmp_path)
    proc = _run_script(tmp_path, "2")

    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert "continuous capture start" in proc.stdout
    assert "capture_start_utc=2020-01-01T00:00:00.000Z" in proc.stdout
    assert "capture_start_utc=2020-01-01T00:00:01.000Z" in proc.stdout
    assert "blocks classified: 2/2" in proc.stdout
    assert "combined reports" in proc.stdout


def test_continuous_capture_run_id_includes_start_time(tmp_path: Path) -> None:
    """Auto-generated run ids must encode the UTC start time, not just date.

    Two consecutive runs on the same UTC date would otherwise collide and
    silently mix into the prior run's directory and DB.
    """
    _install_script(tmp_path)
    proc = _run_script(tmp_path, "2")
    assert proc.returncode == 0, proc.stderr

    match = re.search(r"^run_id=(.+)$", proc.stdout, re.MULTILINE)
    assert match is not None, proc.stdout
    run_id = match.group(1)
    # Format: continuous_YYYYmmddTHHMMSSZ_<count>x<minutes>m
    assert re.fullmatch(
        r"continuous_\d{8}T\d{6}Z_2x0m", run_id
    ), f"unexpected run id: {run_id!r}"


def test_continuous_capture_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    """If RUN_ID already has artifacts on disk, the script must refuse to start.

    Prevents the foot-gun where re-running the same command in the same UTC
    minute (or via an explicit RUN_ID env var) would mix chunks/DB rows from
    two captures into one directory.
    """
    _install_script(tmp_path)
    env = _script_env(tmp_path, RUN_ID="continuous_collision_test")

    first = _run_script(tmp_path, "2", env=env)
    assert first.returncode == 0, first.stderr

    second = _run_script(tmp_path, "2", env=env)
    assert second.returncode == 2, (
        f"second run should refuse, got rc={second.returncode}\n"
        f"stdout:\n{second.stdout}\n\nstderr:\n{second.stderr}"
    )
    assert "already has artifacts on disk" in second.stderr
    assert "continuous_collision_test" in second.stderr
