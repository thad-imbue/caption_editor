#!/usr/bin/env python3
"""Copy a platform ffmpeg binary into dist-rust/ffmpeg for app bundling.

Used by `npm run build:ffmpeg` before electron-builder packs
Contents/Resources/bin/. The Rust CLIs never use the user's PATH at runtime —
only this staged copy (or CAPTION_EDITOR_FFMPEG).

Staging source (build time only): imageio-ffmpeg wheel, else `ffmpeg` on PATH
(e.g. `brew install ffmpeg` on release CI).
"""

from __future__ import annotations

import os
import shutil
import stat
import sys


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find_source() -> str:
    try:
        import imageio_ffmpeg  # type: ignore[import-untyped]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    path = shutil.which("ffmpeg")
    if path:
        return path

    print(
        "Could not locate ffmpeg for bundling.\n"
        "  - Install imageio-ffmpeg:  cd transcribe && uv sync\n"
        "  - Or install ffmpeg on PATH:  brew install ffmpeg",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    src = _find_source()
    dst = os.path.join(_repo_root(), "dist-rust", "ffmpeg")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    mode = os.stat(dst).st_mode
    os.chmod(dst, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Staged ffmpeg for bundling: {dst} ({os.path.getsize(dst)} bytes, from {src})")


if __name__ == "__main__":
    main()
