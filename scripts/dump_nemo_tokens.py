#!/usr/bin/env python3
"""Dump char/word/segment-level NeMo TDT timestamp output to JSON for
side-by-side comparison with `transcribe-rs --dump-tokens`.

Why: parakeet-rs and NeMo both run the same parakeet-tdt-0.6b-v3 weights but
disagree on a few percent of segment boundaries — most visibly punctuation
that NeMo emits and parakeet-rs's Rust inference apparently doesn't. We need
to know whether the *raw tokens* differ (-> file an issue against parakeet-rs)
or just the *grouping* differs (-> fix our grouping locally).

Usage:
    cd transcribe && uv run python ../scripts/dump_nemo_tokens.py \\
        <audio.wav> --chunk-size 60 --overlap 5 \\
        --output /tmp/nemo_tokens.json

Output schema (one entry per chunk, mirrors transcribe-rs --dump-tokens):
    [
      {
        "chunk_index": int,
        "chunk_start_s": float,
        "chunk_end_s": float,
        "tokens": [ { "text": str, "start": float, "end": float }, ... ],
        "segments": [ { "segment": str, "start": float, "end": float }, ... ]
      },
      ...
    ]
The "tokens" list is NeMo's char-level timestamp output (closest analog to
parakeet-rs's `TimestampMode::Tokens`). The "segments" list is NeMo's own
sentence segmentation (no parakeet-rs equivalent — parakeet-rs's
`group_by_sentences` is our hand-rolled approximation).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import soundfile as sf


def chunk_bounds(duration: float, chunk_size: float, overlap: float):
    """Mirror transcribe-rs / transcribe_cli's chunked driver."""
    stride = chunk_size - overlap
    n = max(math.ceil((duration - overlap) / stride), 1)
    for i in range(n):
        start = i * stride
        end = min(start + chunk_size, duration)
        if end <= start:
            return
        yield i, start, end


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path, help="Input WAV (16kHz mono recommended; ffmpeg conversion not done here).")
    p.add_argument("--chunk-size", type=float, default=60.0)
    p.add_argument("--overlap", type=float, default=5.0)
    p.add_argument("--model-id", default="nvidia/parakeet-tdt-0.6b-v3")
    p.add_argument("--output", type=Path, default=Path("/tmp/nemo_tokens.json"))
    args = p.parse_args()

    import nemo.collections.asr as nemo_asr  # noqa: PLC0415 — heavy
    import numpy as np  # noqa: PLC0415

    print(f"Loading {args.model_id} via NeMo...", file=sys.stderr)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_id)

    info = sf.info(str(args.audio))
    audio, sr = sf.read(str(args.audio))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"audio must be 16kHz mono; got {sr} Hz (run ffmpeg first)")
    print(f"Audio: {info.duration:.1f}s, {sr} Hz", file=sys.stderr)

    out = []
    for i, c_start, c_end in chunk_bounds(info.duration, args.chunk_size, args.overlap):
        start_idx = int(round(c_start * sr))
        end_idx = int(round(c_end * sr))
        slice_ = audio[start_idx:end_idx]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, slice_, sr)
            res = model.transcribe([tmp.name], timestamps=True, verbose=False)
        hyp = res[0]
        ts = getattr(hyp, "timestamp", None) or {}

        # NeMo timestamp keys: 'char', 'word', 'segment' (varies by model).
        # We keep all three; transcribe-rs only emits the equivalent of 'char'.
        def shape_offsets(items, text_key):
            shaped = []
            for it in items or []:
                shaped.append({
                    "text": it.get(text_key, ""),
                    # NeMo gives both frame offsets and time-in-seconds — prefer time.
                    "start": it.get("start", it.get("start_offset")),
                    "end": it.get("end", it.get("end_offset")),
                })
            return shaped

        # Token-level analog (NeMo char-level).
        tokens = shape_offsets(ts.get("char"), "char")
        # NeMo's segment-level (the thing we're trying to reproduce).
        segments = shape_offsets(ts.get("segment"), "segment")
        words = shape_offsets(ts.get("word"), "word")

        out.append({
            "chunk_index": i,
            "chunk_start_s": c_start,
            "chunk_end_s": c_end,
            "tokens": tokens,
            "words": words,
            "segments": segments,
        })
        print(f"  chunk {i}: {len(tokens)} char-tokens, {len(words)} words, {len(segments)} segments", file=sys.stderr)

    args.output.write_text(json.dumps(out, indent=2))
    print(f"\nWrote → {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
