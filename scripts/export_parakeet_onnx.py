#!/usr/bin/env python3
"""Build our own ONNX export of nvidia/parakeet-tdt-0.6b-v3 from the official
NeMo checkpoint, suitable for consumption by //transcribe_rs/transcribe-rs/.

Why we do this ourselves instead of using istupakov's pre-built export
(https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx):

  istupakov's export script (verified at
  https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx/discussions/1)
  modifies the model's attention before exporting:

      model.change_attention_model('rel_pos_local_attn', [128, 128])
      model.change_subsampling_conv_chunking_factor(...)

  This swaps full attention for local-attention windows for ONNX long-audio
  stability. The tradeoff is weaker long-range context — most visibly,
  punctuation prediction (which often relies on sentence-rhythm cues several
  seconds away) is degraded. We chunk audio at 60s already, so we don't need
  the local-attn workaround and we'd rather keep the punctuation accuracy.

  Hosting our own export also closes the supply-chain question: weights come
  from NVIDIA's official `nvidia/parakeet-tdt-0.6b-v3.nemo`, the export goes
  through NeMo's blessed `.export()` API, and we control the on-disk SHA.

Usage:
    cd transcribe && uv run python ../scripts/export_parakeet_onnx.py
    # Outputs go to ./out/parakeet-tdt-0.6b-v3-onnx/
    # Then optionally push to HF:
    #   uv run python ../scripts/export_parakeet_onnx.py --upload <repo-id>
    # which requires `huggingface-cli login` first.

Output layout (matches parakeet-rs's `ParakeetTDT::from_pretrained` contract):
    out/parakeet-tdt-0.6b-v3-onnx/
        encoder-model.onnx
        encoder-model.onnx.data        (external weights >2GB)
        decoder_joint-model.onnx
        vocab.txt                       (one token per line: "<token> <id>")
        config.json                     (minimal {model_type, features_size, ...})
        SOURCE_PROVENANCE.md            (human-readable record of how this was built)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_OUT_DIR = Path("out") / "parakeet-tdt-0.6b-v3-onnx"


def export(model_id: str, out_dir: Path) -> None:
    # Heavy imports kept inside the function so the script's --help works
    # without paying the NeMo / torch import cost.
    import nemo.collections.asr as nemo_asr
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    out_dir = out_dir.resolve()
    if out_dir.exists():
        print(f"Removing existing {out_dir}", file=sys.stderr)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir()

    print(f"Loading {model_id} from HuggingFace via NeMo...", file=sys.stderr)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)

    # IMPORTANT: do NOT call model.change_attention_model('rel_pos_local_attn', ...).
    # We export with the *original* full-attention configuration so the resulting
    # ONNX produces the same logits as NeMo's PyTorch runtime — that's the point
    # of doing our own export.

    print("Calling model.export() — produces encoder + decoder_joint ONNX...", file=sys.stderr)
    model.export(str(tmp_dir / "model.onnx"))

    # NeMo emits encoder-model.onnx with weights stored inline. For the v3 (~600M
    # param) encoder this is ~2.5GB, which exceeds ONNX's 2GB protobuf limit on
    # some loaders. Rewrite with external data — same convention istupakov uses
    # and parakeet-rs / ORT both handle out of the box.
    encoder_in = tmp_dir / "encoder-model.onnx"
    encoder_out = out_dir / "encoder-model.onnx"
    encoder_data = encoder_out.name + ".data"
    print(f"Rewriting {encoder_in.name} with external data → {encoder_data}", file=sys.stderr)
    onnx_model = onnx.load(str(encoder_in))
    convert_model_to_external_data(
        onnx_model,
        all_tensors_to_one_file=True,
        location=encoder_data,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(
        onnx_model,
        str(encoder_out),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=encoder_data,
        size_threshold=0,
    )

    # decoder_joint is small (~72MB) — single-file is fine.
    dec_joint_in = tmp_dir / "decoder_joint-model.onnx"
    dec_joint_out = out_dir / "decoder_joint-model.onnx"
    print(f"Copying {dec_joint_in.name} → {dec_joint_out.name}", file=sys.stderr)
    shutil.move(str(dec_joint_in), str(dec_joint_out))

    # Vocab file. parakeet-rs reads this as one "<token> <id>" per line; we
    # use the same format istupakov uses so the on-disk schema is compatible.
    vocab_path = out_dir / "vocab.txt"
    print(f"Writing {vocab_path.name}", file=sys.stderr)
    with vocab_path.open("w") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")

    # Minimal config — parakeet-rs reads features_size + subsampling_factor.
    # We deliberately omit istupakov's `enable_local_attn` / `conv_chunking_factor`
    # keys since we didn't apply those transforms; downstream readers should
    # treat their absence as "default / full attention".
    config_path = out_dir / "config.json"
    print(f"Writing {config_path.name}", file=sys.stderr)
    with config_path.open("w") as f:
        json.dump(
            {
                "model_type": "nemo-conformer-tdt",
                "features_size": 128,
                "subsampling_factor": 8,
            },
            f,
            indent=2,
        )
        f.write("\n")

    # Provenance trail. Human-readable, lives next to the weights so anyone
    # downloading them later can see exactly how they were produced.
    provenance = out_dir / "SOURCE_PROVENANCE.md"
    provenance.write_text(
        textwrap.dedent(
            f"""\
            # parakeet-tdt-0.6b-v3 ONNX export — provenance

            Built by `scripts/export_parakeet_onnx.py` in the caption_editor repo.

            ## What this is

            ONNX export of NVIDIA's `{model_id}` ASR model, compatible with
            parakeet-rs's `ParakeetTDT::from_pretrained` contract.

            ## How it was built

            - Source weights: `{model_id}` (HuggingFace, NeMo `.nemo` checkpoint).
            - Tooling: NeMo `model.export()` (stock NVIDIA export path).
            - Attention model: **full attention** (unchanged from the upstream
              checkpoint). We deliberately did NOT apply
              `model.change_attention_model('rel_pos_local_attn', [128, 128])`
              the way `istupakov/parakeet-tdt-0.6b-v3-onnx` does — local attention
              is fine for long-audio robustness but it degrades punctuation
              prediction, and we chunk audio at 60s already so we don't need it.
            - Export script (with this exact behavior) is committed at
              `scripts/export_parakeet_onnx.py` in the caption_editor repo.

            ## When

            {datetime.now(tz=timezone.utc).isoformat(timespec="seconds")}

            ## File layout (matches istupakov's repo so parakeet-rs is a drop-in)

            - `encoder-model.onnx` + `encoder-model.onnx.data` (~2.5 GB external)
            - `decoder_joint-model.onnx` (~70 MB)
            - `vocab.txt` (token<sp>id, one per line; last token is `<blk>`)
            - `config.json` (`model_type`, `features_size`, `subsampling_factor`)
            """
        )
    )

    shutil.rmtree(tmp_dir)

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\nDone. Wrote {len(list(out_dir.rglob('*')))} files, {total / 1e9:.2f} GB to {out_dir}", file=sys.stderr)


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
        commit_message="Initial export from nvidia/parakeet-tdt-0.6b-v3 (full attention, no local-attn modification).",
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
        help="After exporting, push the result to this HuggingFace repo "
        "(e.g. thadd3us/parakeet-tdt-0.6b-v3-onnx). Requires "
        "`huggingface-cli login` or HF_TOKEN env var.",
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
