#!/usr/bin/env bash
# Refresh a JS coverage report against the source tree.
#
# Invoked via `bazel run //:ts_vitest_coverage` or
# `bazel run //:ts_full_coverage` (which delegate to
# `npm run test:unit:coverage` and `npm run test:coverage` respectively
# — see the targets in //:BUILD.bazel). Both produce HTML output under
# `coverage/` at the repo root, plus the lcov files that
# `coverage-v8` / `nyc` emit alongside.
#
# Not a sh_test on purpose: there's no pass/fail signal beyond what
# the underlying vitest / nyc invocations already provide, and we
# don't want `bazel test //...` to pay the coverage-instrumentation
# cost on every run. Use `bazel run` when you want a fresh report.
#
# First positional arg = the npm script name to invoke. Subsequent
# args are forwarded to `npm run`.
set -euo pipefail

SCRIPT="${1:?missing npm script name}"
shift

WORKSPACE="${BUILD_WORKSPACE_DIRECTORY:?must be invoked via 'bazel run'}"
cd "$WORKSPACE"

if [[ ! -d node_modules ]]; then
  echo "ERROR: source-tree node_modules missing — run 'npm install' first." >&2
  exit 1
fi

# Headless so the Electron window doesn't pop to the foreground during
# the e2e coverage pass (same default as tools/bazel/run_playwright.sh).
export HEADLESS="${HEADLESS:-true}"

echo "[bazel] cwd=$WORKSPACE exec: npm run $SCRIPT -- $*" >&2
exec npm run "$SCRIPT" -- "$@"
