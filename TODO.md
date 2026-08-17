# radio-classifier — TODO

## HIGH PRIORITY

### [~] Capture supervisor watchdog for "stuck-but-alive" capture hangs
**Added:** 2026-06-09 · **Priority:** HIGH · **Status:** partially done (macOS)

**Update 2026-07-15 (macOS fork):** `macos/scripts/continuous_capture_blocks.sh`
now has both safety nets: the stall watchdog (kills a capture whose WAV stops
growing) *and* a bounded `wait_for_sidecar` with a per-block hard deadline
(`BLOCK_DEADLINE = block_seconds * 1.5`, proposal items 1 + 4) — a missing/partial
sidecar now skips the block and drains the run instead of hanging. Covered by
`tests/test_macos_continuous_capture.py::test_missing_sidecar_times_out_without_hanging`.
Still open: partial-WAV flush→classify (items 2/3) and porting the deadline to the
Linux/WSL `scripts/continuous_capture_blocks.sh`.

**Problem**
During the 2026-06-08 48h run (`capture_run_id=448`), the pipeline silently
**hung at block 24/96** for hours without crashing. The process tree was all
alive (`rtl_fm`, both `continuous_capture_blocks.sh` workers, the supervisor),
WSL was up, but **no progress was made**: block 24 had a partial WAV (~67 MB)
and **no sidecar JSON**, so `continuous_capture_blocks.sh`'s `wait_for_sidecar`
blocked indefinitely waiting for a chunk that would never complete.

**Root cause (diagnosed)**
A transient RTL-SDR / USB-IP capture hiccup left `rtl_fm` (or the
`capture chunks` writer) alive but no longer producing a *completable* chunk.
`wait_for_sidecar()` only breaks out if the capture PID dies — it has **no
timeout for a capture process that is alive but stalled**. So a half-written
block wedges the whole supervisor.

**Why current safety nets don't catch it**
- `MAX_ZERO_PROGRESS_ITERS` only triggers between *iterations*, not inside a
  block that never finishes.
- `wait_for_sidecar` loops on `sleep 2` forever while the PID is alive.
- No monitoring of WAV file growth (the real signal of liveness).

**Proposed fix (design to be planned)**
1. **Stall watchdog in `wait_for_sidecar`**: bound the wait. Track the WAV's
   size/mtime; if it hasn't grown in N seconds (e.g. > expected block_seconds +
   grace, or no growth for ~90s), declare the block stalled.
2. **On stall**: kill the capture child + `rtl_fm`, finalize/flush whatever
   partial WAV exists (write a `complete:false` sidecar so it can still be
   classified), and let the supervisor's iteration/backoff logic restart a
   fresh capture child.
3. **Optional**: a lightweight heartbeat/health line (last WAV mtime, bytes/s)
   in the capture log so a monitor (or cheaper LLM operator) can detect stalls.
4. Consider a hard per-block deadline = `block_seconds * 1.5`.

**Acceptance criteria**
- A simulated mid-block `rtl_fm` freeze (alive but not writing) is detected
  within ~2 min, the partial block is flushed + classified, and capture
  resumes automatically without operator intervention.
- No regression to the normal completed-block path.
- Unit/integration coverage for the stall-detection branch.

**Workaround until built:** monitor block progress; if a block's WAV stops
growing with no sidecar, stop the capture tree, flush+classify the partial
block, close the run, and relaunch for the remaining hours (the manual recovery
used on 2026-06-09).
