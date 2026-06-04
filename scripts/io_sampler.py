#!/usr/bin/env python3
"""Low-overhead WSL I/O + memory sampler for diagnosing host HDD saturation.

Appends a compact snapshot every ``--interval`` seconds covering:

* swap activity (pages swapped in/out per interval) and dirty/writeback pages,
* the backing block device's utilisation and read/write throughput, and
* the top processes by I/O delta over the interval.

It reads only ``/proc`` (in-memory kernel counters), so it adds negligible
disk load while a capture run is hammering HDD 0. Optionally exits when a
watched PID (the capture supervisor) goes away.

Usage:
    python scripts/io_sampler.py --interval 60 --device sdd \
        --stop-pid 849047 --out data/logs/io_sample.log
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

_SECTOR_BYTES = 512


def _read_vmstat() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, _, val = line.partition(" ")
        try:
            out[key] = int(val)
        except ValueError:
            continue
    return out


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0].rstrip(":")] = int(parts[1])  # kB
            except ValueError:
                continue
    return out


def _read_diskstats(device: str) -> dict[str, int] | None:
    for line in Path("/proc/diskstats").read_text().splitlines():
        t = line.split()
        if len(t) < 14 or t[2] != device:
            continue
        # Fields after name: 1..11 per Documentation/admin-guide/iostats.
        f = t[3:]
        return {
            "sectors_read": int(f[2]),
            "ms_reading": int(f[3]),
            "sectors_written": int(f[6]),
            "ms_writing": int(f[7]),
            "ms_doing_io": int(f[9]),
        }
    return None


def _read_proc_io() -> dict[int, tuple[str, int]]:
    """Return ``{pid: (comm, read_bytes + write_bytes)}`` for readable procs."""
    out: dict[int, tuple[str, int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            io_text = Path(f"/proc/{pid}/io").read_text()
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        total = 0
        for line in io_text.splitlines():
            if line.startswith(("read_bytes:", "write_bytes:")):
                try:
                    total += int(line.split()[1])
                except (IndexError, ValueError):
                    pass
        out[pid] = (comm, total)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--device", default="sdd", help="block device in /proc/diskstats (default: sdd)")
    ap.add_argument("--stop-pid", type=int, default=None, help="exit when this PID disappears")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/logs/io_sample.log"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(msg: str) -> None:
        with args.out.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        print(msg, flush=True)

    emit(f"io_sampler: start ts={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
         f"device={args.device} interval={args.interval}s stop_pid={args.stop_pid}")

    prev_vm = _read_vmstat()
    prev_disk = _read_diskstats(args.device)
    prev_io = _read_proc_io()
    prev_t = time.monotonic()

    while True:
        time.sleep(args.interval)
        now_t = time.monotonic()
        dt = now_t - prev_t
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        vm = _read_vmstat()
        mem = _read_meminfo()
        disk = _read_diskstats(args.device)
        io = _read_proc_io()

        swpin = vm.get("pswpin", 0) - prev_vm.get("pswpin", 0)
        swpout = vm.get("pswpout", 0) - prev_vm.get("pswpout", 0)

        lines = [f"=== {ts} (interval {dt:.0f}s) ==="]
        lines.append(
            f"mem: free={mem.get('MemFree', 0)//1024}MiB avail={mem.get('MemAvailable', 0)//1024}MiB "
            f"dirty={mem.get('Dirty', 0)//1024}MiB writeback={mem.get('Writeback', 0)//1024}MiB "
            f"swapfree={mem.get('SwapFree', 0)//1024}/{mem.get('SwapTotal', 0)//1024}MiB"
        )
        lines.append(f"swap: pages_in={swpin} pages_out={swpout} (>0 out = thrashing pressure)")

        if disk and prev_disk:
            util = (disk["ms_doing_io"] - prev_disk["ms_doing_io"]) / (dt * 1000.0) * 100.0
            rd = (disk["sectors_read"] - prev_disk["sectors_read"]) * _SECTOR_BYTES / dt / 1e6
            wr = (disk["sectors_written"] - prev_disk["sectors_written"]) * _SECTOR_BYTES / dt / 1e6
            lines.append(
                f"{args.device}: util={util:.0f}% read={rd:.2f}MB/s write={wr:.2f}MB/s"
            )

        deltas = []
        for pid, (comm, total) in io.items():
            if pid in prev_io:
                d = total - prev_io[pid][1]
                if d > 0:
                    deltas.append((d, pid, comm))
        deltas.sort(reverse=True)
        lines.append("top proc I/O this interval (MB):")
        for d, pid, comm in deltas[: args.top]:
            lines.append(f"  {comm:<20} pid={pid:<8} {d/1e6:8.2f}MB")
        if not deltas:
            lines.append("  (no measurable per-proc disk I/O)")

        emit("\n".join(lines))

        prev_vm, prev_disk, prev_io, prev_t = vm, disk, io, now_t

        if args.stop_pid is not None:
            try:
                os.kill(args.stop_pid, 0)
            except ProcessLookupError:
                emit(f"io_sampler: stop_pid {args.stop_pid} gone; exiting at {ts}")
                return 0
            except PermissionError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
