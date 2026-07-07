#!/usr/bin/env bash
set -euo pipefail

# macOS preflight: RTL dongle, Ollama, disk space, venv.
#
# Usage (from repo root):
#   source macos/env.defaults
#   bash macos/scripts/preflight.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
source "$ROOT_DIR/macos/env.defaults"

FAIL=0

echo "=== macOS preflight ==="
echo "repo=$ROOT_DIR"
echo "ollama_host=$RADIO_CLASSIFIER_OLLAMA_HOST"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "FAIL: .venv missing — run bash macos/install.sh first" >&2
  FAIL=1
else
  echo "ok  venv=$ROOT_DIR/.venv"
fi

if ! command -v rtl_test >/dev/null 2>&1; then
  echo "FAIL: rtl_test not on PATH — brew install librtlsdr" >&2
  FAIL=1
else
  echo "ok  rtl_test=$(command -v rtl_test)"
  if ! rtl_test -t 2>&1 | head -5; then
    echo "FAIL: rtl_test -t did not succeed — plug in the RTL dongle" >&2
    FAIL=1
  fi
fi

if ! curl -sf --max-time 5 "${RADIO_CLASSIFIER_OLLAMA_HOST%/}/api/tags" >/dev/null 2>&1; then
  echo "FAIL: Ollama unreachable at $RADIO_CLASSIFIER_OLLAMA_HOST" >&2
  echo "      Start Ollama.app or: ollama serve" >&2
  FAIL=1
else
  echo "ok  ollama reachable ($RADIO_CLASSIFIER_OLLAMA_HOST)"
fi

echo "disk:"
df -h . | tail -1

echo
echo "=== radio-classifier prereq-check ==="
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  "$ROOT_DIR/.venv/bin/python" -m radio_classifier prereq-check --ollama || FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi
echo "preflight: all checks passed"
