#!/usr/bin/env bash
# Asserts that APP_VERSION matches everywhere it's declared:
#
#   electron/constants.ts       APP_VERSION = '1.6.1'
#   transcribe_rs/version.bzl   APP_VERSION = "1.6.1"
#   transcribe_rs/Cargo.toml    version = "1.6.1"   (workspace.package)
#
# Drift between these is silent at runtime — the symptom is that the
# Electron app downloads release artifacts for one version while the
# Rust binaries actually report themselves as a different version,
# and `.captions_json5` blob URLs point at the wrong git tag.
#
# Bumping APP_VERSION: see CLAUDE.md → "Version Management". Touch all
# three files in one commit; this test will fail loudly if you miss one.

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
version_bzl="$(rlocation _main/transcribe_rs/version.bzl)"
cargo_toml="$(rlocation _main/transcribe_rs/Cargo.toml)"

for f in "$electron_ts" "$version_bzl" "$cargo_toml"; do
  [[ -f "$f" ]] || { echo "ERROR: file not found at $f" >&2; exit 1; }
done

# electron/constants.ts: `export const APP_VERSION = '1.6.1'`
ts_v="$(grep -E "^export const APP_VERSION = '[^']+'" "$electron_ts" \
        | head -n1 | sed -E "s/^export const APP_VERSION = '([^']+)'.*/\1/")"

# transcribe_rs/version.bzl: `APP_VERSION = "1.6.1"`
bzl_v="$(grep -E '^APP_VERSION = "[^"]+"' "$version_bzl" \
         | head -n1 | sed -E 's/^APP_VERSION = "([^"]+)".*/\1/')"

# transcribe_rs/Cargo.toml: pick the `version = "1.6.1"` line that sits
# inside the [workspace.package] table. We assume that table appears
# before any other `version = ...` line in the file (which is true today
# and a fine assumption for a workspace-root Cargo.toml).
cargo_v="$(awk '
  /^\[workspace\.package\]/ { in_wsp = 1; next }
  /^\[/                     { in_wsp = 0 }
  in_wsp && /^version *=/ {
    match($0, /"[^"]+"/);
    print substr($0, RSTART + 1, RLENGTH - 2);
    exit
  }
' "$cargo_toml")"

if [[ -z "$ts_v" || -z "$bzl_v" || -z "$cargo_v" ]]; then
  echo "ERROR: could not extract APP_VERSION from one of:" >&2
  echo "  electron/constants.ts → '$ts_v'" >&2
  echo "  transcribe_rs/version.bzl → '$bzl_v'" >&2
  echo "  transcribe_rs/Cargo.toml → '$cargo_v'" >&2
  exit 1
fi

if [[ "$ts_v" != "$bzl_v" || "$ts_v" != "$cargo_v" ]]; then
  cat >&2 <<EOF
APP_VERSION drift — bump these together (see CLAUDE.md → Version Management):

  electron/constants.ts        APP_VERSION = '$ts_v'
  transcribe_rs/version.bzl    APP_VERSION = "$bzl_v"
  transcribe_rs/Cargo.toml     version = "$cargo_v"
EOF
  exit 1
fi

echo "OK: APP_VERSION = $ts_v is consistent across electron, version.bzl, Cargo.toml."
