"""Behavioural tests for `scripts/morning_run_blocks.sh`.

The script is a thin orchestrator, but its per-block resilience is the whole
point of the 2026-05-30 hardening: an ``rtl_fm`` failure in one block must NOT
take down later blocks. We exercise that by intercepting ``python`` on PATH
with a shell stub that fails block 1's ingest and succeeds for everything
else, then asserting the script:

  *  continues through all blocks
  *  writes a single line to the skip log
  *  reaches the report section
  *  exits 0 because at least one block completed

We deliberately stub the entire ``radio_classifier`` Python entrypoint rather
than running the real pipeline — the test is about the bash glue, not the
classifier.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "morning_run_blocks.sh"


def _write_stub_python(stub_dir: Path, *, fail_block_1_ingest: bool) -> Path:
    """Drop a fake ``python`` shim into ``stub_dir`` that mimics the script's
    Python invocations enough to drive it through ``BLOCK_COUNT`` iterations.

    The shim:
    - For ``radio_classifier ingest``: writes a tiny WAV file at
      ``--capture-wav`` and exits 0 (or exits 1 when ``fail_block_1_ingest``
      is set AND the WAV path ends in ``block1.wav``).
    - For everything else: prints args and exits 0.
    """
    stub_dir.mkdir(parents=True, exist_ok=True)
    stub = stub_dir / "python"
    stub.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            # Test stub — pretends to be the project's python entrypoint.
            FAIL_BLOCK_1_INGEST={"1" if fail_block_1_ingest else "0"}

            # Look for a `-m radio_classifier <subcommand>` invocation.
            SUBCMD=""
            CAPTURE_WAV=""
            i=0
            args=("$@")
            while [[ $i -lt $# ]]; do
              if [[ "${{args[$i]}}" == "radio_classifier" ]]; then
                SUBCMD="${{args[$((i+1))]}}"
              fi
              if [[ "${{args[$i]}}" == "--capture-wav" ]]; then
                CAPTURE_WAV="${{args[$((i+1))]}}"
              fi
              i=$((i+1))
            done

            if [[ "$SUBCMD" == "ingest" ]]; then
              if [[ "$FAIL_BLOCK_1_INGEST" == "1" && "$CAPTURE_WAV" == *block1.wav ]]; then
                echo "stub: simulating rtl_fm failure for block 1" >&2
                exit 1
              fi
              if [[ -n "$CAPTURE_WAV" ]]; then
                # Write a non-empty placeholder so the script's "is the WAV
                # actually populated" check passes.
                mkdir -p "$(dirname "$CAPTURE_WAV")"
                printf 'STUBWAV' > "$CAPTURE_WAV"
              fi
              exit 0
            fi

            # All other invocations (db init, classify, report, songs) are
            # treated as successes — they just need to not blow up.
            echo "stub:$SUBCMD ${{args[*]}}"
            exit 0
            """
        )
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stub


def _run_script(
    workdir: Path,
    *,
    block_count: int,
    block_seconds: int,
    fail_block_1_ingest: bool,
) -> subprocess.CompletedProcess:
    """Copy the script into a fresh worktree-ish ``workdir`` and run it with
    PATH pointed at a stub ``python``.

    We need ``.venv/bin/python`` to resolve to our stub, so we drop a symlink
    of the stub into ``workdir/.venv/bin/python``.
    """
    (workdir / ".venv" / "bin").mkdir(parents=True)
    stub = _write_stub_python(workdir / "_stubs", fail_block_1_ingest=fail_block_1_ingest)
    (workdir / ".venv" / "bin" / "python").symlink_to(stub.resolve())

    (workdir / "scripts").mkdir()
    shutil.copy(SCRIPT_PATH, workdir / "scripts" / "morning_run_blocks.sh")

    env = {
        **os.environ,
        "PATH": f"{workdir / '_stubs'}:{os.environ['PATH']}",
        "BLOCK_SECONDS": str(block_seconds),
    }
    return subprocess.run(
        ["bash", "scripts/morning_run_blocks.sh", str(block_count)],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_per_block_capture_failure_logs_and_continues(tmp_path: Path) -> None:
    """When block 1's capture fails, the script must log the skip, continue
    to block 2 (and beyond), and exit 0 because at least one block landed."""
    proc = _run_script(tmp_path, block_count=3, block_seconds=1, fail_block_1_ingest=True)

    # Script must succeed overall — at least block 2 and 3 ran.
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    # Skip log must exist and mention block 1.
    skip_logs = list((tmp_path / "data" / "logs").glob("*_skips.log"))
    assert len(skip_logs) == 1, f"expected exactly one skip log, found: {skip_logs}"
    skip_body = skip_logs[0].read_text()
    assert "block 1: capture FAILED" in skip_body

    # Run summary must show the partial completion.
    assert "blocks completed: 2/3" in proc.stdout
    assert "blocks with capture failure: 1" in proc.stdout
    assert "combined reports" in proc.stdout


def test_all_blocks_succeed_path_is_unchanged(tmp_path: Path) -> None:
    """Sanity: with the stub passing every call, all blocks complete and no
    skip log is written."""
    proc = _run_script(tmp_path, block_count=2, block_seconds=1, fail_block_1_ingest=False)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    assert "blocks completed: 2/2" in proc.stdout
    assert "blocks with capture failure: 0" in proc.stdout

    skip_logs = list((tmp_path / "data" / "logs").glob("*_skips.log"))
    # Either the file doesn't exist or it's empty.
    if skip_logs:
        assert skip_logs[0].read_text() == ""


def test_all_blocks_fail_exits_nonzero_and_skips_reports(tmp_path: Path) -> None:
    """If every block's capture fails, the script must exit non-zero AND not
    print the reports section (which would dump empty tables 5x)."""
    # Stub fails block 1's ingest specifically; we run with block_count=1 so
    # the only block is the failing one.
    proc = _run_script(tmp_path, block_count=1, block_seconds=1, fail_block_1_ingest=True)
    assert proc.returncode != 0
    assert "no blocks completed" in proc.stderr
    assert "combined reports" not in proc.stdout
