"""Integration test: download the Rust ASR binaries for the current
APP_VERSION from the GitHub Release and verify they run end-to-end.

Pair with //transcribe:transcribe_rs_parity_test which exercises the
locally-Bazel-built binaries. This test exercises the actual artifacts
ship to users — same binary, same code path, but pulled over the wire
from `releases/download/v${APP_VERSION}/...`. If a release shipped a
broken binary (mis-stamped version, missing notarization, wrong arch)
this test catches it.

Tagged `manual + expensive + requires-network` — won't run on default
`bazel test //...`. Trigger explicitly:

    bazel test //transcribe:released_rust_asr_test \\
        --test_tag_filters=expensive --test_output=streamed

The release is pinned to the APP_VERSION currently in the working tree,
so a checkout at v1.6.1 exercises the v1.6.1 release's artifact. Releases
are immutable on GitHub — re-running the test against the same checkout
always hits the same bytes.
"""

from __future__ import annotations

import re
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from captions_json5_lib import parse_captions_json5_file


# Repo coordinates (lifted from electron/constants.ts at test time so
# there's no second source of truth to drift).
REPO_HTTP = "https://github.com/thadd3us/caption_editor"
ASSET_SUFFIX = "darwin-arm64"  # only platform we currently publish.


def _read_app_version(repo_root: Path) -> str:
    """Read APP_VERSION from electron/constants.ts (the canonical declaration).

    Matches `//tools/bazel:version_consistency_test`'s regex, which guarantees
    this same value is present in transcribe_rs/version.bzl + Cargo.toml.
    """
    txt = (repo_root / "electron" / "constants.ts").read_text()
    m = re.search(r"^export const APP_VERSION = '([^']+)'", txt, re.MULTILINE)
    assert m, "couldn't extract APP_VERSION from electron/constants.ts"
    return m.group(1)


def _platform_ok() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`. Skips with a clear reason on 404 (release
    not yet published for this version), so the test reports "not yet"
    rather than failing with HTTP-error noise."""
    req = urllib.request.Request(url, headers={"User-Agent": "caption-editor-test"})
    try:
        with urllib.request.urlopen(req) as resp:
            dest.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            pytest.skip(
                f"Release artifact not found at {url}. Has the release been "
                f"published with the Rust binaries attached? (HTTP 404)"
            )
        raise
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def released_binaries(repo_root: Path) -> dict[str, Path]:
    """Download both binaries on first request. The cache dir is keyed off
    the release tag (so a re-run of the same test re-uses the bytes), so
    function-scoping the fixture is cheap — at most a couple of stat()
    calls per test after the first download."""
    if not _platform_ok():
        pytest.skip("Released Rust binaries are only published for darwin-arm64.")

    app_version = _read_app_version(repo_root)
    tag = f"v{app_version}"
    cache = Path(tempfile.gettempdir()) / f"caption_editor_released_rs_{tag}"
    cache.mkdir(exist_ok=True)

    out: dict[str, Path] = {}
    for name in ("transcribe-rs", "embed-rs"):
        local = cache / name
        if not local.exists():
            url = f"{REPO_HTTP}/releases/download/{tag}/{name}-{tag}-{ASSET_SUFFIX}"
            _download(url, local)

        # Self-check: the binary's --version output should match APP_VERSION.
        ver_out = subprocess.run(
            [str(local), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        # clap formats this as "<binary> <version>"
        reported = ver_out.split()[-1] if ver_out else ""
        assert reported == app_version, (
            f"{name} downloaded from {tag} reports --version={reported!r}, "
            f"expected {app_version!r}. Release artifact may have been built "
            f"from a different commit than the tag claims."
        )

        out[name] = local
    return out


@pytest.mark.expensive
def test_released_transcribe_rs_runs(
    repo_root: Path, tmp_path: Path, released_binaries: dict[str, Path]
) -> None:
    """End-to-end on the released binary: real audio → real ASR → parseable doc."""
    audio = repo_root / "test_data" / "OSR_us_000_0010_8k.wav"
    output = tmp_path / "out.captions_json5"

    result = subprocess.run(
        [
            str(released_binaries["transcribe-rs"]),
            str(audio),
            "--output",
            str(output),
            "--chunk-size",
            "10",
            "--overlap",
            "5",
            "--deterministic-ids",
            "--no-embed",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"released transcribe-rs failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output.exists()

    doc = parse_captions_json5_file(output)
    assert doc.metadata.id == "doc_id"
    assert len(doc.segments) > 0
    assert all(seg.id.startswith("id_") for seg in doc.segments)
    assert doc.raw_asr_output is not None


@pytest.mark.expensive
def test_released_embed_rs_runs(
    repo_root: Path, tmp_path: Path, released_binaries: dict[str, Path]
) -> None:
    """End-to-end on the released embed-rs: transcribe → embed → embeddings present."""
    audio_src = repo_root / "test_data" / "OSR_us_000_0010_8k.wav"
    audio = (
        tmp_path / "OSR_us_000_0010_8k.wav"
    )  # local copy so mediaFilePath is portable
    audio.write_bytes(audio_src.read_bytes())
    output = tmp_path / "out.captions_json5"

    subprocess.run(
        [
            str(released_binaries["transcribe-rs"]),
            str(audio),
            "--output",
            str(output),
            "--chunk-size",
            "10",
            "--overlap",
            "5",
            "--deterministic-ids",
            "--no-embed",
        ],
        check=True,
    )
    subprocess.run(
        [str(released_binaries["embed-rs"]), str(output)],
        check=True,
    )

    doc = parse_captions_json5_file(output)
    assert doc.embeddings is not None
    assert len(doc.embeddings) > 0

    from schema import decode_embedding

    for e in doc.embeddings:
        vec = decode_embedding(e.speaker_embedding)
        assert len(vec) > 0
