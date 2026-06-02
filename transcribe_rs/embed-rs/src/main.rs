//! `embed-rs` — Rust replacement for `transcribe/embed_cli.py`.
//!
//! Workflow: parse a `.captions_json5` → resolve its `mediaFilePath` →
//! (re-encode to WAV via `ffmpeg` if not already WAV) → for each segment,
//! load the audio slice and compute a wespeaker speaker embedding via
//! ONNX → write the embeddings back into the document.
//!
//! Model source: `pyannote/wespeaker-voxceleb-resnet34-LM` (default).
//! Resolved via the `hf-hub` crate using the same cache layout HF Python
//! uses (`~/.cache/huggingface/hub/models--owner--name/...`), so a model
//! already pulled by the Python CLI is reused without re-downloading.
//!
//! Feature gap vs Python: UMAP reductions (the `--umap-dimensions` flag
//! in the Python CLI) are not yet computed here — see TODO at the bottom.
//! The doc field stays `umapEmbeddings: None` until a Rust UMAP impl is
//! wired in.

use caption_schema::{
    decode_embedding, encode_embedding, parse_captions_json5, serialize_captions_json5,
    CaptionsDocument, SegmentSpeakerEmbedding,
};

mod umap;
use clap::Parser;
use eyre::{eyre, Context, ContextCompat, Result};
use hf_hub::api::sync::ApiBuilder;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MODEL: &str = "pyannote/wespeaker-voxceleb-resnet34-LM";
const DEFAULT_MIN_SEGMENT_DURATION_SECS: f64 = 0.3;
/// Wespeaker reads 16 kHz mono; `ffmpeg` resamples to match.
const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Mirror of the `ASR_COMMIT_HASH` in `transcribe/constants.py` /
/// `electron/constants.ts`. Used for the `{HASH}`-substituted schema URLs
/// in the .captions_json5 header. Kept in sync via the version-bump dance
/// in CLAUDE.md.
const ASR_COMMIT_HASH: &str = "2986a2e3330c839ec45cb12a5c00f0dc24476ac5";

#[derive(Parser)]
#[command(
    about = "Compute speaker embeddings for a .captions_json5 file.",
    long_about = "Rust port of transcribe/embed_cli.py. Same wespeaker model, same on-disk \
                  HF cache, same .captions_json5 wire format.",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Args {
    /// Path to the .captions_json5 file. Edited in place.
    captions_path: PathBuf,
    /// HuggingFace model id for the embedding model.
    #[clap(long, short = 'm', default_value = DEFAULT_MODEL)]
    model: String,
    /// Skip segments shorter than this many seconds.
    #[clap(long, default_value_t = DEFAULT_MIN_SEGMENT_DURATION_SECS)]
    min_segment_duration: f64,
    /// UMAP target dimensionalities. Pass each one separately, e.g.
    /// `--umap-dimensions 1 --umap-dimensions 2`. Default `[1, 2]`
    /// matches the Python CLI; pass an empty list (`--no-umap`) to skip.
    #[clap(long = "umap-dimensions", value_name = "DIM", num_args = 0..)]
    umap_dimensions: Option<Vec<usize>>,
    /// Disable UMAP entirely.
    #[clap(long, conflicts_with = "umap_dimensions")]
    no_umap: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Parsing captions JSON: {}", args.captions_path.display());
    let content = std::fs::read_to_string(&args.captions_path)
        .with_context(|| format!("read {}", args.captions_path.display()))?;
    let mut document: CaptionsDocument =
        parse_captions_json5(&content).map_err(|e| eyre!("parse captions json5: {e}"))?;

    let media_path = resolve_media_path(&args.captions_path, &document)?;
    eprintln!("Media file: {}", media_path.display());
    eprintln!("Found {} segments", document.segments.len());

    let temp_dir = tempfile::tempdir()?;
    let wav_path = ensure_wav(&media_path, &temp_dir)?;

    eprintln!("Loading embedding model: {}", args.model);
    let model_onnx = download_wespeaker_onnx(&args.model)?;
    let mut session = create_session(&model_onnx)?;

    eprintln!("Computing embeddings...");
    let (samples, sample_rate) = read_wav_mono_i16(&wav_path)?;
    if sample_rate != TARGET_SAMPLE_RATE {
        return Err(eyre!(
            "expected {TARGET_SAMPLE_RATE} Hz after ffmpeg resample, got {sample_rate}"
        ));
    }

    let mut embeddings_out: Vec<SegmentSpeakerEmbedding> = Vec::new();
    let mut skipped = 0usize;
    for seg in &document.segments {
        let dur = seg.end_time - seg.start_time;
        if dur < args.min_segment_duration {
            skipped += 1;
            continue;
        }
        let slice = slice_samples(&samples, sample_rate, seg.start_time, seg.end_time);
        if slice.is_empty() {
            continue;
        }
        let emb = compute_embedding(&mut session, &slice)?;
        embeddings_out.push(SegmentSpeakerEmbedding {
            segment_id: seg.id.clone(),
            speaker_embedding: encode_embedding(&emb),
            umap_embeddings: None,
        });
    }
    eprintln!(
        "Embedded {} segments, skipped {} (too short)",
        embeddings_out.len(),
        skipped
    );

    // UMAP reductions. Defaults to [1, 2] matching the Python CLI. We compute
    // a separate fit per requested dimensionality (same shape Python emits:
    // `umap_embeddings[i].umap_embeddings = {"1": [...], "2": [...]}`).
    let umap_dims: Vec<usize> = if args.no_umap {
        Vec::new()
    } else {
        args.umap_dimensions.clone().unwrap_or_else(|| vec![1, 2])
    };
    if !umap_dims.is_empty() && embeddings_out.len() > 1 {
        let raw: Vec<Vec<f32>> = embeddings_out
            .iter()
            .map(|e| decode_embedding(&e.speaker_embedding))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| eyre!("decode embedding for UMAP input: {e}"))?;

        let mut per_segment: Vec<std::collections::BTreeMap<String, Vec<f64>>> =
            vec![Default::default(); embeddings_out.len()];
        for dim in umap_dims {
            eprintln!("UMAP n_components={dim} (n_neighbors=15, init=random) — fitting...");
            match umap::compute_umap(&raw, dim) {
                Ok(Some(reduced)) => {
                    for (i, row) in reduced.iter().enumerate() {
                        per_segment[i].insert(
                            dim.to_string(),
                            row.iter().map(|x| *x as f64).collect(),
                        );
                    }
                    eprintln!("UMAP n_components={dim} finished.");
                }
                Ok(None) => {
                    eprintln!("UMAP n_components={dim} skipped (too few embeddings).");
                }
                Err(e) => {
                    // Python catches all UMAP errors and just skips the dim;
                    // do the same so a flaky run doesn't lose the embeddings.
                    eprintln!("UMAP computation failed for n_components={dim}: {e}");
                }
            }
        }
        for (slot, mut entries) in embeddings_out.iter_mut().zip(per_segment.into_iter()) {
            if !entries.is_empty() {
                slot.umap_embeddings = Some(std::mem::take(&mut entries));
            }
        }
    }

    document.embeddings = Some(embeddings_out);
    document.embedding_model = Some(args.model.clone());

    eprintln!(
        "Writing embeddings to captions JSON: {}",
        args.captions_path.display()
    );
    let serialized = serialize_captions_json5(&document, ASR_COMMIT_HASH);
    std::fs::write(&args.captions_path, serialized)
        .with_context(|| format!("write {}", args.captions_path.display()))?;
    let n = document.embeddings.as_ref().map_or(0, Vec::len);
    eprintln!("Done! Wrote {n} embeddings to captions JSON");

    Ok(())
}

// ---------------------------------------------------------------------------
// Path / IO helpers
// ---------------------------------------------------------------------------

/// Resolve `metadata.media_file_path` relative to the captions file's
/// directory (matches Python's behavior in `embed_captions_path`).
fn resolve_media_path(captions_path: &Path, doc: &CaptionsDocument) -> Result<PathBuf> {
    let media = doc
        .metadata
        .media_file_path
        .as_deref()
        .context("metadata.mediaFilePath is required")?;
    let dir = captions_path.parent().unwrap_or_else(|| Path::new("."));
    let abs = dir.join(media);
    let normalized = normalize_path(&abs);
    if !normalized.exists() {
        return Err(eyre!("media file not found: {}", normalized.display()));
    }
    Ok(normalized)
}

/// Resolve `.` and `..` components without touching the filesystem. Mirrors
/// Python's `os.path.normpath` behavior for the joined captions/media path.
fn normalize_path(p: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// If `media` is already a WAV, return it unchanged. Otherwise run `ffmpeg`
/// to re-encode to a 16-kHz mono WAV under `temp_dir`. Matches Python's
/// `audio_utils.extract_audio_to_wav`.
fn ensure_wav(media: &Path, temp_dir: &tempfile::TempDir) -> Result<PathBuf> {
    let lower = media
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    if matches!(lower.as_deref(), Some("wav") | Some("wave")) {
        return Ok(media.to_path_buf());
    }

    eprintln!("Converting {} to WAV format...", media.display());
    let out_path = temp_dir.path().join("audio.wav");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-i",
            &media.to_string_lossy(),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            &out_path.to_string_lossy(),
        ])
        .status()
        .context("ffmpeg not on PATH — install ffmpeg or convert input to WAV manually")?;
    if !status.success() {
        return Err(eyre!("ffmpeg exited with {status}"));
    }
    Ok(out_path)
}

fn read_wav_mono_i16(path: &Path) -> Result<(Vec<i16>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int if spec.bits_per_sample == 16 => {
            reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map(|f| (f.clamp(-1.0, 1.0) * 32767.0) as i16))
            .collect::<Result<Vec<_>, _>>()?,
        other => return Err(eyre!("unsupported WAV format: {other:?}")),
    };
    // Mix-down if multi-channel. ffmpeg upstream produced mono so this is
    // usually a no-op, but plain WAV inputs can be stereo.
    let mono = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks(ch)
            .map(|frame| (frame.iter().map(|s| *s as i32).sum::<i32>() / ch as i32) as i16)
            .collect()
    } else {
        samples
    };
    Ok((mono, spec.sample_rate))
}

fn slice_samples(samples: &[i16], sample_rate: u32, start_s: f64, end_s: f64) -> Vec<i16> {
    let start_idx = (start_s * sample_rate as f64).round().max(0.0) as usize;
    let end_idx = (end_s * sample_rate as f64).round().max(0.0) as usize;
    let end_idx = end_idx.min(samples.len());
    if start_idx >= end_idx {
        return Vec::new();
    }
    samples[start_idx..end_idx].to_vec()
}

// ---------------------------------------------------------------------------
// Model download (matches HF Python cache layout)
// ---------------------------------------------------------------------------

/// Download the wespeaker ONNX model to the same HF cache directory the
/// Python CLI uses (`~/.cache/huggingface/hub/models--<owner>--<name>/...`).
/// We pull a single `*.onnx` file and look it up in the snapshot.
fn download_wespeaker_onnx(model_id: &str) -> Result<PathBuf> {
    let token = std::env::var("HF_TOKEN").ok();
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }
    let api = builder
        .build()
        .map_err(|e| eyre!("hf-hub init: {e}"))?
        .model(model_id.to_string());

    // pyannote/wespeaker-voxceleb-resnet34-LM publishes the ONNX as
    // `pytorch_model.bin`-style files; the actual ONNX export commonly
    // lives under `*.onnx`. Try a couple of common names — falling
    // back keeps this robust to repo-side renames.
    for candidate in ["pytorch_model.onnx", "model.onnx", "wespeaker.onnx"] {
        if let Ok(p) = api.get(candidate) {
            return Ok(p);
        }
    }
    Err(eyre!(
        "no ONNX file found in {model_id} (looked for pytorch_model.onnx / model.onnx / wespeaker.onnx). \
         Pre-download the model with the Python embed_cli once, or provide the path directly."
    ))
}

// ---------------------------------------------------------------------------
// ORT session + fbank → embedding (was in spike/main.rs; kept inline)
// ---------------------------------------------------------------------------

fn create_session(path: &Path) -> Result<Session> {
    // Map ort's parameterized SessionBuilder errors to plain eyre — rc.12's
    // `Error<SessionBuilder>` isn't Send+Sync because of operator boxes,
    // so `?` -> eyre auto-conversion fails. This is the same workaround
    // the spike used; see SPIKE_NOTES.md.
    Ok(Session::builder()
        .map_err(|e| eyre!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| eyre!("opt level: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| eyre!("intra threads: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| eyre!("inter threads: {e}"))?
        .commit_from_file(path)
        .map_err(|e| eyre!("commit_from_file: {e}"))?)
}

fn compute_embedding(session: &mut Session, samples_i16: &[i16]) -> Result<Vec<f32>> {
    let mut samples_f32 = vec![0.0f32; samples_i16.len()];
    knf_rs::convert_integer_to_float_audio(samples_i16, &mut samples_f32);

    // knf-rs pins ndarray 0.16, ort 0.17 — copy through raw shape to bridge.
    let feats_v016 = knf_rs::compute_fbank(&samples_f32)?;
    let (n_frames, n_bins) = feats_v016.dim();
    let (flat, _offset) = feats_v016.into_raw_vec_and_offset();
    let feats: Array2<f32> = Array2::from_shape_vec((n_frames, n_bins), flat)?;
    let feats = feats.insert_axis(ndarray::Axis(0)); // batch dim

    let inputs = ort::inputs!["feats" => Tensor::from_array(feats)?];
    let outs = session.run(inputs)?;
    let out = outs
        .get("embs")
        .context("output 'embs' missing from wespeaker session")?
        .try_extract_tensor::<f32>()
        .context("extract embs tensor")?;
    Ok(out.1.iter().copied().collect())
}
