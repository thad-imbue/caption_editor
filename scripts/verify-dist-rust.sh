#!/usr/bin/env bash
# Assert dist-rust/ has the Mach-O binaries electron-builder copies into the .app.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist-rust"

for name in transcribe-rs embed-rs ffmpeg; do
  path="$DIST/$name"
  if [[ ! -f "$path" ]]; then
    echo "error: missing $path — run: npm run build:rust" >&2
    exit 1
  fi
  if [[ ! -x "$path" ]]; then
    echo "error: not executable: $path — run: npm run build:rust" >&2
    exit 1
  fi
done

echo "OK: dist-rust bundle (transcribe-rs, embed-rs, ffmpeg)"
