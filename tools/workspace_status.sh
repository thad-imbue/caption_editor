#!/usr/bin/env bash
# Bazel --workspace_status_command for caption-editor.
#
# Runs once per build (when --stamp is in effect, which .bazelrc turns
# on by default) and writes STABLE_* keys to bazel-out/stable-status.txt.
# Anything prefixed STABLE_ participates in the cache key for stamped
# rules; non-stable keys are runtime-only.
#
# Both keys fall back to "unknown" outside a git checkout (release
# tarballs, etc.) so consumers always see a non-empty value.
set -euo pipefail

if hash=$(git rev-parse HEAD 2>/dev/null); then
    echo "STABLE_GIT_HASH ${hash}"
else
    echo "STABLE_GIT_HASH unknown"
fi

# `git describe --tags --always --dirty` resolves to one of:
#   v0.1.2              — at a tag
#   v0.1.2-3-gabc123d   — N commits past v0.1.2
#   abc123d             — no reachable tag (--always)
#   <any>-dirty         — working tree has uncommitted changes
if describe=$(git describe --tags --always --dirty 2>/dev/null); then
    echo "STABLE_GIT_DESCRIBE ${describe}"
else
    echo "STABLE_GIT_DESCRIBE unknown"
fi
