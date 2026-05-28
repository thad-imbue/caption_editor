#!/usr/bin/env bash
# Generic wrapper: cd into the source workspace and exec "$@".
#
# Usable from both `bazel run` (uses $BUILD_WORKSPACE_DIRECTORY) and
# `bazel test` (resolves the source tree via the package.json runfile
# sentinel, then exec's into the source dir — these wrappers are
# intentionally non-hermetic so they can reuse node_modules/ and
# transcribe/.venv/ that the developer has already populated).
#
# This script is a sh_binary; the per-suite sh_test targets in
# //:BUILD.bazel reference it as `srcs` and pass the command + args
# via the rule's `args` attribute.
set -eo pipefail

# --- bazel runfiles bootstrap (standard snippet) ---
f=bazel_tools/tools/bash/runfiles/runfiles.bash
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null \
  || source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f 2- -d ' ')" 2>/dev/null \
  || source "$0.runfiles/$f" 2>/dev/null \
  || source "$0.runfiles/_main/$f" 2>/dev/null \
  || { echo>&2 "ERROR: cannot find bazel runfiles bootstrap"; exit 1; }
set -u

WORKSPACE="${BUILD_WORKSPACE_DIRECTORY:-}"
if [[ -z "$WORKSPACE" ]]; then
  # In a sh_test the script runs in the runfiles tree, where
  # package.json is a symlink back to the source. Resolve through it
  # to get the real source dir.
  SENTINEL_RUNFILE="$(rlocation _main/package.json)" || SENTINEL_RUNFILE=""
  if [[ -z "$SENTINEL_RUNFILE" || ! -e "$SENTINEL_RUNFILE" ]]; then
    echo "ERROR: cannot locate package.json in runfiles" >&2
    exit 1
  fi
  SENTINEL_REAL="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$SENTINEL_RUNFILE")"
  WORKSPACE="$(dirname "$SENTINEL_REAL")"
fi
[[ -d "$WORKSPACE" ]] || { echo "ERROR: workspace not found: $WORKSPACE" >&2; exit 1; }

cd "$WORKSPACE"
echo "[bazel] cwd=$WORKSPACE exec: $*" >&2
exec "$@"
