#!/usr/bin/env python3
"""Build our own ONNX export of pyannote/wespeaker-voxceleb-resnet34-LM
for consumption by //transcribe_rs/embed-rs/.

Why we do this ourselves:
  The pyannote HF repo (`pyannote/wespeaker-voxceleb-resnet34-LM`) only
  publishes a PyTorch `pytorch_model.bin` — there is no ONNX file there
  for embed-rs's `ort` + `knf-rs` runtime to load. The community ONNX
  exports (sherpa-onnx etc.) are different model variants with slightly
  different fbank conventions. Hosting our own export keeps embed-rs
  numerically aligned with `transcribe/embed_cli.py` (same weights, same
  preprocessing) and pins the supply chain.

Numerical parity with Python:
  pyannote's `BaseWeSpeakerResNet` does fbank inline with
  `torchaudio.compliance.kaldi.fbank` then subtracts the per-utterance
  mean (`features - features.mean(dim=1, keepdim=True)`). embed-rs uses
  `knf-rs::compute_fbank` which is also kaldi-fbank and *also* applies
  per-utterance mean centering. Pyannote scales waveforms by 32768
  before fbank; knf-rs scales by 1/32768. The 32768^2 energy ratio is a
  constant log-mel offset that cancels under mean centering — so the
  fbank feeding the ResNet matches up to FP noise. We export *just the
  ResNet* (fbank → embedding) and keep preprocessing in Rust.

Usage:
    cd transcribe && uv run python ../scripts/export_wespeaker_onnx.py
    # → out/wespeaker-voxceleb-resnet34-LM-onnx/model.onnx
    # Then optionally push to HF:
    #   uv run python ../scripts/export_wespeaker_onnx.py --upload <repo-id>

Output layout (matches embed-rs's expected `model.onnx` + `embs` output):
    out/wespeaker-voxceleb-resnet34-LM-onnx/
        model.onnx
        config.json                 (input/output shapes, fbank params)
        SOURCE_PROVENANCE.md        (human-readable build record)
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"
DEFAULT_OUT_DIR = Path("out") / "wespeaker-voxceleb-resnet34-LM-onnx"


def export(model_id: str, out_dir: Path) -> None:
    # Heavy imports kept inside the function so the script's --help works
    # without paying the torch / pyannote import cost.
    import torch
    import torch.nn as nn
    from pyannote.audio import Model

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_id} from HuggingFace via pyannote...", file=sys.stderr)
    model = Model.from_pretrained(model_id)
    model.eval()

    # Pyannote's WeSpeakerResNet34.forward takes raw waveforms and does
    # fbank inline. We want the ONNX to consume fbank features directly
    # so embed-rs can keep preprocessing in Rust (knf-rs). Wrap the
    # underlying resnet to expose a single-tensor in/out signature.
    resnet = model.resnet  # the bare ResNet34, takes (B, T, F=80) fbank

    class ResNetExport(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            # resnet.forward(fbank) → (embed_a_or_zero, embed). With
            # two_emb_layer=False (the WeSpeakerResNet34 default) the
            # first element is a scalar `torch.tensor(0.0)` placeholder.
            _, embed = self.resnet(feats)
            return embed

    wrapper = ResNetExport(resnet).eval()

    # Dummy input: 100 frames of 80-dim log-mel fbank. Shape is the
    # contract embed-rs depends on: (batch, frames, 80).
    dummy = torch.zeros(1, 100, 80, dtype=torch.float32)

    out_path = out_dir / "model.onnx"
    print(f"Exporting to {out_path}...", file=sys.stderr)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out_path),
        input_names=["feats"],
        output_names=["embs"],
        dynamic_axes={
            "feats": {0: "batch", 1: "frames"},
            "embs": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Sanity check: load with ort and run the dummy input through. Catches
    # most export bugs (missing ops, wrong shapes) before we ship.
    print("Validating ONNX with onnxruntime...", file=sys.stderr)
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"feats": dummy.numpy()})[0]
    assert out.shape == (1, 256), f"expected (1, 256) embedding, got {out.shape}"
    print(f"  → output shape OK: {out.shape}, dtype {out.dtype}", file=sys.stderr)

    # PyTorch vs ONNX numerical drift check.
    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()
    max_abs_err = float(np.max(np.abs(torch_out - out)))
    print(f"  → max abs error vs PyTorch: {max_abs_err:.2e}", file=sys.stderr)
    if max_abs_err > 1e-3:
        print(
            f"  ! WARNING: drift > 1e-3, ONNX may not be a faithful export",
            file=sys.stderr,
        )

    # Config sidecar — small but useful for anyone consuming the model
    # without reading this script.
    config_path = out_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "model_type": "wespeaker-resnet34",
                "input_name": "feats",
                "input_shape": "(batch, frames, 80)",
                "output_name": "embs",
                "output_shape": "(batch, 256)",
                "fbank": {
                    "sample_rate": 16000,
                    "num_mel_bins": 80,
                    "frame_length_ms": 25,
                    "frame_shift_ms": 10,
                    "dither": 0.0,
                    "window_type": "hamming",
                    "centering": "global_mean",
                    "comment": (
                        "Preprocessing matches kaldi.fbank with these params, "
                        "applied to int16-scaled audio, then per-utterance mean "
                        "subtraction. knf-rs (kaldi-native-fbank) produces "
                        "equivalent features up to a constant log-mel offset "
                        "that the mean-subtraction cancels."
                    ),
                },
            },
            f,
            indent=2,
        )
        f.write("\n")

    provenance = out_dir / "SOURCE_PROVENANCE.md"
    provenance.write_text(
        textwrap.dedent(
            f"""\
            # wespeaker-voxceleb-resnet34-LM ONNX export — provenance

            Built by `scripts/export_wespeaker_onnx.py` in the caption_editor repo.

            ## What this is

            ONNX export of the speaker-embedding ResNet inside pyannote's
            `{model_id}`, consumed by `//transcribe_rs/embed-rs/`.

            ## Why our own export

            The upstream `{model_id}` HF repo only ships
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

            {datetime.now(tz=timezone.utc).isoformat(timespec="seconds")}

            ## Numerical drift vs PyTorch

            See stderr output from the build run (target: < 1e-3 max abs
            error on a zero-input).
            """
        )
    )

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    n_files = len(list(out_dir.rglob("*")))
    print(
        f"\nDone. Wrote {n_files} files, {total / 1e6:.1f} MB to {out_dir}",
        file=sys.stderr,
    )


def upload(repo_id: str, out_dir: Path) -> None:
    """Push the export to a HuggingFace repo. Requires `huggingface-cli login`
    or HF_TOKEN env var. Creates the repo if it doesn't exist."""
    from huggingface_hub import HfApi, create_repo  # noqa: PLC0415

    api = HfApi()
    print(f"Ensuring HF repo exists: {repo_id}", file=sys.stderr)
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"Uploading {out_dir} → {repo_id}...", file=sys.stderr)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(out_dir),
        repo_type="model",
        commit_message=(
            "Initial ONNX export of pyannote/wespeaker-voxceleb-resnet34-LM "
            "(resnet only, takes fbank input)."
        ),
    )
    print(f"\nUploaded. https://huggingface.co/{repo_id}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model id to export (default: {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--upload",
        metavar="REPO_ID",
        help=(
            "After exporting, push the result to this HuggingFace repo "
            "(e.g. thadd3us/wespeaker-voxceleb-resnet34-LM-onnx). Requires "
            "`huggingface-cli login` or HF_TOKEN env var."
        ),
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip the export step (use an existing --out-dir). Useful with --upload.",
    )
    args = p.parse_args()

    if not args.skip_export:
        export(args.model_id, args.out_dir)
    if args.upload:
        upload(args.upload, args.out_dir)


if __name__ == "__main__":
    main()
