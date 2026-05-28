#!/usr/bin/env bash
# Asserts that the `ASR_COMMIT_HASH` constant has the same value in
# electron/constants.ts and transcribe/constants.py.
#
# Why this exists: that hash is the git revision that production
# electron passes to `uvx …@${ASR_COMMIT_HASH}` to fetch the
# transcribe package, AND the literal that gets baked into
# `.captions_json5` schema blob URLs from both the Electron app and
# the Python CLI. The two files MUST agree — drift produces files
# whose blob URL doesn't match the code that wrote them, with no
# loud failure mode at runtime (the file just silently records the
# wrong commit). CLAUDE.md documents the two-commit release dance;
# this test enforces it in CI.
#
# `electron/constants.ts` carries APP_VERSION too, but APP_VERSION is
# only declared once in the repo (electron-builder reads it from the
# regex match in `scripts/package`), so there's nothing to compare it
# against and the check is unnecessary.
set -euo pipefail

# --- bazel runfiles bootstrap ---
f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null \
  || source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f 2- -d ' ')" 2>/dev/null \
  || source "$0.runfiles/$f" 2>/dev/null \
  || source "$0.runfiles/_main/$f" 2>/dev/null \
  || { echo>&2 "ERROR: cannot find bazel runfiles bootstrap"; exit 1; }

electron_ts="$(rlocation _main/electron/constants.ts)"
python_py="$(rlocation _main/transcribe/constants.py)"

[[ -f "$electron_ts" ]] || { echo "ERROR: electron/constants.ts not found at $electron_ts" >&2; exit 1; }
[[ -f "$python_py" ]]   || { echo "ERROR: transcribe/constants.py not found at $python_py" >&2; exit 1; }

# Pull the literal from each side. Patterns are anchored to the exact
# `export const ASR_COMMIT_HASH = '...'` (TS) and
# `ASR_COMMIT_HASH = "..."` (Python) shapes the files currently use.
# `head -n1` is defensive in case future code drops a comment with the
# same text.
ts_hash="$(grep -E "^export const ASR_COMMIT_HASH = '[^']+'" "$electron_ts" \
            | head -n1 | sed -E "s/^export const ASR_COMMIT_HASH = '([^']+)'.*/\1/")"
py_hash="$(grep -E '^ASR_COMMIT_HASH = "[^"]+"' "$python_py" \
            | head -n1 | sed -E 's/^ASR_COMMIT_HASH = "([^"]+)".*/\1/')"

if [[ -z "$ts_hash" ]]; then
    echo "ERROR: could not find ASR_COMMIT_HASH declaration in $electron_ts" >&2
    exit 1
fi
if [[ -z "$py_hash" ]]; then
    echo "ERROR: could not find ASR_COMMIT_HASH declaration in $python_py" >&2
    exit 1
fi

if [[ "$ts_hash" != "$py_hash" ]]; then
    cat >&2 <<EOF
ASR_COMMIT_HASH mismatch — these MUST stay in lockstep (the value is
both the uvx pin and the .captions_json5 schema blob URL revision).

  electron/constants.ts   ASR_COMMIT_HASH = '$ts_hash'
  transcribe/constants.py ASR_COMMIT_HASH = "$py_hash"

See CLAUDE.md → "Version Management" for the two-commit release
dance that bumps these together.
EOF
    exit 1
fi

echo "OK: both files declare ASR_COMMIT_HASH = $ts_hash"
