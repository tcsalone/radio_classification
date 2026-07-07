#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper: macOS operators call this; logic stays in scripts/post_run_cleanup.sh
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT_DIR/scripts/post_run_cleanup.sh" "$@"
