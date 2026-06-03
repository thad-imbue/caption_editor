# wespeaker-voxceleb-resnet34-LM ONNX export — provenance

Built by `scripts/export_wespeaker_onnx.py` in the caption_editor repo.

## What this is

ONNX export of the speaker-embedding ResNet inside pyannote's
`pyannote/wespeaker-voxceleb-resnet34-LM`, consumed by `//transcribe_rs/embed-rs/`.

## Why our own export

The upstream `pyannote/wespeaker-voxceleb-resnet34-LM` HF repo only ships
`pytorch_model.bin` — no ONNX. embed-rs needs ONNX to run via
`ort`. Doing our own export keeps us numerically aligned with
`transcribe/embed_cli.py` (same weights, same preprocessing
conventions) and pins supply chain.

## Input / output contract

- Input: `feats` — float32 tensor `(batch, frames, 80)` log-mel
  fbank. Compute with kaldi-fbank settings: 16 kHz, 80 mel
  bins, 25 ms frame, 10 ms shift, hamming window, no dither.
  Apply per-utterance global mean subtraction before feeding.
  embed-rs uses `knf-rs::compute_fbank` which does all of this.
- Output: `embs` — float32 tensor `(batch, 256)` speaker
  embedding.

## When

2026-06-03T00:30:53+00:00

## Numerical drift vs PyTorch

See stderr output from the build run (target: < 1e-3 max abs
error on a zero-input).
