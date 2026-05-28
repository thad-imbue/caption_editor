#!/usr/bin/env bash
# Run the Playwright Electron e2e suite by delegating to the existing
# `npm run test:e2e` script (which does vue-tsc → vite build →
# vite-electron build → playwright test) against the source tree.
#
# Why this stays non-hermetic (unlike //:ts_typecheck, //:ts_vitest_unit,
# //:ts_eslint, which now run against the bazel-linked `:node_modules`):
#
#   * `npm run test:e2e` chains several tools (`vue-tsc`, `vite`,
#     `electron-builder`, `playwright`) via npm scripts. npm scripts
#     find binaries by prepending `node_modules/.bin/` to PATH. But
#     aspect_rules_js's linked `node_modules/` tree does NOT
#     materialize pnpm-style `.bin/` shims — packages sit at their
#     canonical `node_modules/<scope>/<name>/` paths and downstream
#     callers are expected to invoke their CLIs via `node` directly.
#
#   * Doing that here would mean re-implementing every npm script as
#     a bazel rule (one js_run_binary per tool, then a sh_test that
#     consumes the dist outputs). That is the right destination but
#     a separate, larger piece of work — see the comment block on
#     `//:e2e_playwright` in BUILD.bazel.
#
# In the meantime: the e2e wrapper just cd's to the workspace (under
# `bazel run` via BUILD_WORKSPACE_DIRECTORY; under `bazel test` via a
# package.json runfile sentinel) and shells out to `npm`. Source-tree
# `node_modules/` must exist (`npm install` once). Tagged `no-sandbox`
# + `requires-network` in the sh_test rule.
set -eo pipefail

# --- bazel runfiles bootstrap ---
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
  # `bazel test` mode: resolve the source tree via the package.json
  # runfile sentinel (same trick run_in_workspace.sh uses).
  PKG_RUNFILE="$(rlocation _main/package.json)" || PKG_RUNFILE=""
  if [[ -z "$PKG_RUNFILE" || ! -e "$PKG_RUNFILE" ]]; then
    echo "ERROR: cannot locate package.json in runfiles" >&2
    exit 1
  fi
  WORKSPACE="$(dirname "$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$PKG_RUNFILE")")"
fi
[[ -d "$WORKSPACE" ]] || { echo "ERROR: workspace not found: $WORKSPACE" >&2; exit 1; }
cd "$WORKSPACE"

if [[ ! -d node_modules ]]; then
  echo "ERROR: source-tree node_modules missing — run 'npm install' first." >&2
  exit 1
fi

# Force headless Electron unless the caller explicitly set HEADLESS=
# (e.g. `HEADLESS=false bazel run //:e2e_playwright_bin` to debug a
# flake visually). `electron/main.ts` reads this and passes `show:
# false` to BrowserWindow; `playwright.config.ts` reads it too for
# its `use.headless` flag. Without this the Electron window pops to
# the foreground on every test run and steals focus.
export HEADLESS="${HEADLESS:-true}"

echo "[bazel] cwd=$WORKSPACE HEADLESS=$HEADLESS exec: npm run test:e2e $*" >&2
exec npm run test:e2e -- "$@"
