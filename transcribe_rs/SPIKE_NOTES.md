# Rust port spike — findings

Scoped to: can we replace `transcribe/` Python with Rust binaries, dropping the `uvx` distribution path?

## TL;DR

**Yes, with low residual risk.** The two scary models both have working Rust paths today:

| Concern | Status | Path |
|---|---|---|
| Parakeet-TDT-0.6b-v3 inference | ✅ solved | `parakeet-rs` (crates.io, v0.3) — TDT decoding, word timestamps, CoreML on Apple Silicon |
| ONNX export of Parakeet | ✅ already done | `istupakov/parakeet-tdt-0.6b-v3-onnx` on HF |
| wespeaker-voxceleb embedding | ✅ solved | Direct: `ort` 2.0 + `knf-rs` (kaldi-native-fbank), ~50 lines |
| HF model download + caching | ✅ solved | `hf-hub` crate (native, ureq+rustls) |
| Post-processing pipeline | ✅ trivial | pure logic, snapshot-tested already |
| Chunked driver (60s + 5s overlap) | ⚠ to do | Python logic in `transcribe_cli.transcribe_audio_file` ports straight over |

`cargo check` passes for both stub binaries (transcribe-rs, embed-rs) on this machine.

## What the spike contains

```
hack/rust_port_spike/
├── caption-types/        # serde mirrors of ASRSegment / WordTimestamp — wire-compatible with Python JSON
├── transcribe-rs/        # CLI stub using parakeet-rs (ort rc.12)
└── embed-rs/             # CLI stub using ort rc.12 + knf-rs directly
```

Two independent Cargo packages, not a workspace. Reason below.

### Reproduce

```bash
cd hack/rust_port_spike/transcribe-rs && cargo check
cd hack/rust_port_spike/embed-rs      && cargo check
```

First check downloads ~200 crates + builds the kaldi-native-fbank C++ via cmake/bindgen (one-time). Subsequent builds are fast.

## Friction we hit (and resolved)

### 1. `ort` version skew between candidate crates

`parakeet-rs 0.3.5` pins `ort = "2.0.0-rc.12"`. `pyannote-rs 0.3.4` still pins `ort = "2.0.0-rc.10"`. Cargo unified them to rc.12, then pyannote-rs failed to compile — rc.12 changed error-type Send/Sync bounds (`Error<SessionBuilder>` no longer auto-converts to `eyre::Report` via `?`).

**Fix taken:** dropped pyannote-rs entirely. It was only doing ~50 lines of glue around `ort` + `knf-rs`; we inlined that in `embed-rs/src/main.rs` and mapped the parameterized ort errors to plain strings. We get the same wespeaker model, with our own version control over ort.

**Future:** if we want pyannote-rs's segmentation pipeline, we'd vendor it (~291 lines src) and re-patch session.rs the same way, or file a PR upstream. We probably don't need segmentation — ASR already gives us segments.

### 2. ndarray version skew

`knf-rs 0.3` is on `ndarray 0.16`; `ort 2.0-rc.12` is on `ndarray 0.17`. Cargo allows both side-by-side but the types don't interop. Bridge: copy fbank output through raw shape + Vec (negligible tensor size).

### 3. parakeet-rs TDT 5-minute chunk limit

Same as our existing Python flow: we already chunk at 60s with 5s overlap (`transcribe/asr_results_to_captions.py` post-processing handles overlap merge). The chunked driver in `transcribe_cli.transcribe_audio_file` ports directly.

## Test-bridging strategy (what to do next)

The user's proposal — point Python tests at Rust binaries as a second source of truth — works cleanly:

1. **`AsrSegment` is wire-compatible.** `caption-types/src/lib.rs` matches the Python dataclass field-for-field; the snapshot fixtures under `transcribe/test_fixtures/*.json` parse straight into our Rust struct.
2. **Bridge at the CLI layer.** Add `--rust-binary` flag (or env var) to `bulk_cli.py` and the test helpers. `transcribe-rs` would emit identical JSON to `--dump-raw-asr`.
3. **Bridge at the post-processing layer.** `asr_results_to_captions_post_processing_pipeline_test.py` already feeds captured chunked ASR through `post_process_raw_asr_segments`. We can add a parametrized variant that pipes through `transcribe-rs --post-process` instead. **Or** — and this is what I'd actually do first — port the post-processing functions to Rust and parametrize the *Rust* tests against the same fixtures. That's the cheapest cross-check: pure logic, no model needed.

## Distribution payoff

Drops:
- `uvx` bootstrap, `package-for-uvx` flows (already partly dismantled — see commit `0d1b733`).
- The whole `nemo` import + lhotse/torch monkeypatch dance in `transcribe_cli.py:57-90`.
- The "warnings firehose" — NeMo's logging filter hack at line 65 stops being needed because we're only pulling in what we use.

Adds:
- Single static-ish Rust binary per arch. ORT native dylib (~30MB on Apple Silicon) is the only runtime payload; downloadable with the model.
- One-time C++ toolchain requirement to build (cmake/bindgen), but consumers just download a binary.

## Open questions (for the real port, not the spike)

1. Does parakeet-rs's output match Python NeMo numerically on our fixtures? (Need to run on `transcribe/test_fixtures/*.wav` and diff.) The TDT decoder reimplementation is the only place we'd expect drift.
2. CoreML acceleration on Apple Silicon — works in theory via `ort/coreml`, hasn't been exercised here.
3. Modal/VibeVoice remote ASR (`VibeVoiceModalRecognizer`) — this stays Python or becomes a tiny HTTP client in Rust. Not on the critical path.
4. `embed_cli.py` writes back into `.captions_json5`. We need a Rust JSON5 reader/writer (the `json5` crate exists, format is permissive — should be fine).

## Recommendation

Greenlight the full port. Next concrete steps in priority order:

1. **Port post-processing** (`asr_results_to_captions.py` → `caption-types` or sibling crate). Pure logic, snapshot-tested, no model dependency. Proves the test-bridging story end-to-end before we touch ML.
2. **Numerical parity check on parakeet-rs.** One day of work: download `istupakov/parakeet-tdt-0.6b-v3-onnx`, run our fixtures through it, diff against `transcribe/test_fixtures/*_raw_asr.json`.
3. **Wire chunked driver** mirroring `transcribe_audio_file`.
4. **Validate wespeaker numerical parity** the same way.
5. **Hook into Electron app** as a bundled binary (replaces `uvx` fork in `electron/`).

Steps 1–4 can mostly happen in this `hack/rust_port_spike/` directory. Step 5 is where it graduates out.
