#!/usr/bin/env bash
# Portable file size in bytes (GNU stat vs BSD stat).
file_size_bytes() {
  local path="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    stat -f %z "$path" 2>/dev/null || echo 0
  else
    stat -c %s "$path" 2>/dev/null || echo 0
  fi
}
