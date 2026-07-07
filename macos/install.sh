#!/usr/bin/env bash
set -euo pipefail

# macOS standalone install for radio-classifier (M4 / Apple Silicon).
#
# Installs Homebrew deps, Python venv, and pip extras WITHOUT the Linux CUDA
# [gpu] bundle. Does not modify WSL legacy scripts.
#
# Usage:
#   bash macos/install.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macos/install.sh: this script is for macOS only (uname=$(uname -s))" >&2
  exit 2
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required: https://brew.sh" >&2
  exit 2
fi

echo "=== Installing Homebrew packages ==="
brew install librtlsdr ffmpeg python@3.12 || true
# Ollama: brew formula or Ollama.app — either works on :11434
if ! command -v ollama >/dev/null 2>&1; then
  echo "Note: install Ollama from https://ollama.com or: brew install ollama"
fi

echo "=== Python venv ==="
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[acoustic,shazam,seeding,dev]"

echo "=== audfprint (external CLI) ==="
AUDFPRINT_DIR="${HOME}/dev/audfprint"
if [[ ! -f "${AUDFPRINT_DIR}/audfprint.py" ]]; then
  echo "Cloning audfprint to ${AUDFPRINT_DIR} ..."
  mkdir -p "$(dirname "$AUDFPRINT_DIR")"
  git clone https://github.com/dpwe/audfprint.git "$AUDFPRINT_DIR"
  pip install -r "${AUDFPRINT_DIR}/requirements.txt"
  chmod +x "${AUDFPRINT_DIR}/audfprint.py"
fi
echo "Add to your shell profile if needed:"
echo "  export RADIO_CLASSIFIER_AUDFPRINT_BIN=\"python ${AUDFPRINT_DIR}/audfprint.py\""

echo "=== Ollama model ==="
if command -v ollama >/dev/null 2>&1; then
  ollama pull llama3.2:latest || true
fi

echo
echo "=== Install complete ==="
echo "Next:"
echo "  source macos/env.defaults"
echo "  bash macos/scripts/preflight.sh"
