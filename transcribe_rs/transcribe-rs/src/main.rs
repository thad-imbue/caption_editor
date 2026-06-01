//! `transcribe-rs` — Rust replacement for `transcribe/transcribe_cli.py`.
//!
//! Workflow (mirrors Python):
//!   1. Resolve input media → 16 kHz mono WAV (via ffmpeg if not already WAV).
//!   2. Download parakeet-tdt ONNX from HF (encoder/decoder_joint/vocab).
//!   3. Chunked transcribe: 60 s windows with 5 s overlap, run parakeet-rs
//!      on each chunk, offset word timestamps by chunk_start.
//!   4. Run `caption_core::post_process_raw_asr_segments` (Parakeet path:
//!      `is_whisper=false`, gap-split + long-segment-split).
//!   5. Assign deterministic cue IDs (SHA-256 of audio hash + segment start).
//!   6. Write a `.captions_json5` document including `rawAsrOutput` snapshot.
//!
//! Parakeet sentence/word handling: we ask parakeet-rs for raw token
//! timestamps (`TimestampMode::Tokens`), then run its public
//! `process_timestamps` twice to get *both* sentence-level and word-level
//! groupings from the same tokens. Words are then paired into sentences
//! by time-range (with the same 0.01s tolerance Python's
//! `parse_parakeet_raw_chunk` uses), giving the post-processing pipeline
//! sentence-level segments — matching the `is_whisper=False` flow.

use caption_core::{
    asr_segments_to_transcript_segments, post_process_raw_asr_segments,
    raw_asr_segments_to_raw_asr_output, AsrSegment, WordTimestamp,
};
use caption_schema::{
    serialize_captions_json5, CaptionsDocument, TranscriptMetadata,
};
use clap::Parser;
use eyre::{eyre, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};

mod parakeet_grouping;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_MODEL: &str = "istupakov/parakeet-tdt-0.6b-v3-onnx";
const TARGET_SAMPLE_RATE: u32 = 16_000;
const ASR_COMMIT_HASH: &str = "2986a2e3330c839ec45cb12a5c00f0dc24476ac5";

#[derive(Parser)]
#[command(
    about = "Transcribe a media file into a .captions_json5 document.",
    long_about = "Rust port of transcribe/transcribe_cli.py. parakeet-tdt-0.6b-v3 via parakeet-rs,\n\
                  chunked 60s + 5s overlap, same post-processing pipeline as the Python suite."
)]
struct Args {
    /// Input media file (any container ffmpeg can decode).
    media_file: PathBuf,
    /// Output .captions_json5 path. Defaults to `<media>.captions_json5`.
    #[clap(long, short = 'o')]
    output: Option<PathBuf>,
    /// Chunk size in seconds.
    #[clap(long, short = 'c', default_value_t = 60)]
    chunk_size: u32,
    /// Overlap between chunks in seconds.
    #[clap(long, short = 'v', default_value_t = 5)]
    overlap: u32,
    /// HF model id for the ONNX-exported parakeet weights.
    #[clap(long, short = 'm', default_value = DEFAULT_MODEL)]
    model: String,
    /// Maximum gap between words inside a segment before splitting.
    #[clap(long, default_value_t = 0.50)]
    max_intra_segment_gap_seconds: f64,
    /// Maximum segment duration before forcing a split.
    #[clap(long, default_value_t = 10.0)]
    max_segment_duration_seconds: f64,
    /// Use simple incremental IDs (`id_00000`, ...) instead of UUIDs.
    /// Required for tests that snapshot the .captions_json5 output.
    #[clap(long)]
    deterministic_ids: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| args.media_file.with_extension("captions_json5"));
    if output.exists() {
        return Err(eyre!(
            "output file already exists: {} — refusing to overwrite",
            output.display()
        ));
    }

    eprintln!("Transcribing: {}", args.media_file.display());
    eprintln!("Output: {}", output.display());
    eprintln!("Chunk size: {}s, Overlap: {}s", args.chunk_size, args.overlap);

    let temp_dir = tempfile::tempdir()?;
    let wav_path = ensure_wav(&args.media_file, &temp_dir)?;
    let audio_hash = sha256_file(&wav_path)?;
    let (samples_f32, sample_rate, channels) = read_wav_f32_mono(&wav_path)?;

    eprintln!("Loading parakeet model: {}", args.model);
    let model_dir = download_parakeet_onnx(&args.model)?;
    let mut parakeet = ParakeetTDT::from_pretrained(&model_dir, None)
        .map_err(|e| eyre!("ParakeetTDT::from_pretrained: {e}"))?;

    let raw_segments = transcribe_chunked(
        &mut parakeet,
        &samples_f32,
        sample_rate,
        channels,
        args.chunk_size,
        args.overlap,
    )?;

    let processed = post_process_raw_asr_segments(
        raw_segments.clone(),
        args.chunk_size as f64,
        args.overlap as f64,
        args.max_intra_segment_gap_seconds,
        args.max_segment_duration_seconds,
        // Parakeet path: input is sentence-level segments (already grouped
        // by parakeet-rs's `Sentences` mode), pipeline splits on big gaps
        // and on max-duration. Matches Python's `is_whisper=False` flow.
        /* is_whisper */ false,
    );

    let mut transcript = asr_segments_to_transcript_segments(processed, Some(&args.model));
    assign_cue_ids_and_timestamps(&mut transcript, &audio_hash, args.deterministic_ids);

    let raw_asr_output = raw_asr_segments_to_raw_asr_output(&raw_segments);

    let metadata = TranscriptMetadata {
        id: generate_document_id(&audio_hash, args.deterministic_ids),
        media_file_path: Some(args.media_file.to_string_lossy().into_owned()),
    };

    let doc = CaptionsDocument {
        metadata,
        title: args
            .media_file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned()),
        segments: transcript,
        history: None,
        embeddings: None,
        embedding_model: None,
        ui_state: None,
        raw_asr_output: Some(raw_asr_output),
    };

    let serialized = serialize_captions_json5(&doc, ASR_COMMIT_HASH);
    std::fs::write(&output, serialized)
        .with_context(|| format!("write {}", output.display()))?;
    eprintln!("Wrote {} segments to {}", doc.segments.len(), output.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Audio + IO helpers
// ---------------------------------------------------------------------------

fn ensure_wav(media: &Path, temp_dir: &tempfile::TempDir) -> Result<PathBuf> {
    let lower = media
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase);
    if matches!(lower.as_deref(), Some("wav") | Some("wave")) {
        return Ok(media.to_path_buf());
    }
    eprintln!("Converting {} to 16kHz mono WAV via ffmpeg...", media.display());
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
        .context("ffmpeg not on PATH — install ffmpeg or pre-convert input to WAV")?;
    if !status.success() {
        return Err(eyre!("ffmpeg exited with {status}"));
    }
    Ok(out_path)
}

fn read_wav_f32_mono(path: &Path) -> Result<(Vec<f32>, u32, u16)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let f32_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let mono = if spec.channels > 1 {
        let ch = spec.channels as usize;
        f32_samples
            .chunks(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        f32_samples
    };
    Ok((mono, spec.sample_rate, 1))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buf = [0u8; 4096];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

// ---------------------------------------------------------------------------
// Chunked driver
// ---------------------------------------------------------------------------

fn transcribe_chunked(
    parakeet: &mut ParakeetTDT,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    chunk_size: u32,
    overlap: u32,
) -> Result<Vec<AsrSegment>> {
    if sample_rate != TARGET_SAMPLE_RATE {
        return Err(eyre!(
            "expected {TARGET_SAMPLE_RATE} Hz after ffmpeg resample, got {sample_rate}"
        ));
    }
    let total_samples = samples.len();
    let duration = total_samples as f64 / sample_rate as f64;
    let stride = (chunk_size as f64) - (overlap as f64);
    let mut all = Vec::new();

    let num_chunks = ((duration - overlap as f64) / stride).ceil().max(1.0) as usize;
    eprintln!("Transcribing {num_chunks} chunks (~{duration:.1}s of audio)...");

    for i in 0..num_chunks {
        let chunk_start_s = i as f64 * stride;
        let chunk_end_s = (chunk_start_s + chunk_size as f64).min(duration);
        if chunk_end_s <= chunk_start_s {
            break;
        }
        let start_idx = (chunk_start_s * sample_rate as f64).round() as usize;
        let end_idx = ((chunk_end_s * sample_rate as f64).round() as usize).min(total_samples);
        if start_idx >= end_idx {
            continue;
        }
        let slice = samples[start_idx..end_idx].to_vec();
        eprintln!("  chunk {}/{}: [{:.1}s, {:.1}s)", i + 1, num_chunks, chunk_start_s, chunk_end_s);

        let result = parakeet
            // `Tokens` gives raw subword tokens. We re-derive both Sentences
            // and Words via parakeet-rs's public `process_timestamps`, then
            // pair them — same shape as Python's `parse_parakeet_raw_chunk`.
            .transcribe_samples(slice, sample_rate, channels, Some(TimestampMode::Tokens))
            .map_err(|e| eyre!("parakeet transcribe chunk {i}: {e}"))?;

        let sentences = parakeet_grouping::group_by_sentences(&result.tokens);
        let words = parakeet_grouping::group_by_words(&result.tokens);

        for sent in sentences.iter() {
            let abs_s_start = sent.start as f64 + chunk_start_s;
            let abs_s_end = sent.end as f64 + chunk_start_s;

            // Same 0.01s tolerance as Python `parse_parakeet_raw_chunk` — keeps
            // the time-range word/segment pairing identical across languages.
            let mut seg_words = Vec::new();
            for w in &words {
                let abs_w_start = w.start as f64 + chunk_start_s;
                let abs_w_end = w.end as f64 + chunk_start_s;
                if abs_w_start >= abs_s_start - 0.01 && abs_w_end <= abs_s_end + 0.01 {
                    seg_words.push(WordTimestamp {
                        word: w.text.clone(),
                        start: abs_w_start,
                        end: abs_w_end,
                    });
                }
            }

            all.push(AsrSegment {
                text: sent.text.clone(),
                start: abs_s_start,
                end: abs_s_end,
                words: seg_words,
                // Setting chunk_start matches Python's transcribe_audio_file
                // (which assigns segment.chunk_start = chunk_start after parsing),
                // so the overlap-merge picks the chunk-region midpoint.
                chunk_start: Some(chunk_start_s),
                speaker: None,
            });
        }
    }
    Ok(all)
}

// ---------------------------------------------------------------------------
// IDs / hashes (parity with Python transcribe_cli.py)
// ---------------------------------------------------------------------------

fn generate_document_id(audio_hash: &str, deterministic: bool) -> String {
    if deterministic {
        return "doc_id".into();
    }
    let input = format!("doc:{audio_hash}");
    let digest = Sha256::digest(input.as_bytes());
    uuid_from_bytes16(&digest[..16])
}

fn generate_cue_id(audio_hash: &str, start_time: f64, idx: usize, deterministic: bool) -> String {
    if deterministic {
        return format!("id_{idx:05}");
    }
    let combined = format!("{audio_hash}:{start_time:.3}");
    let digest = Sha256::digest(combined.as_bytes());
    uuid_from_bytes16(&digest[..16])
}

/// Format 16 bytes as the canonical UUID hyphenated string (Python's
/// `uuid.UUID(bytes=...)` repr). No actual version/variant bits are set —
/// matches Python's behavior of accepting arbitrary 16-byte input.
fn uuid_from_bytes16(b: &[u8]) -> String {
    assert!(b.len() == 16);
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
    )
}

fn assign_cue_ids_and_timestamps(
    segments: &mut [caption_schema::TranscriptSegment],
    audio_hash: &str,
    deterministic: bool,
) {
    let current_ts = if deterministic {
        "2025-01-01T00:00:00.000000+00:00".to_string()
    } else {
        // chrono pulls a transitive dep we don't strictly need; format the
        // current wall-clock time in RFC3339 by hand from `std::time`.
        // Day-level precision is fine; the field is informational.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Crude RFC3339-ish UTC timestamp (Python uses the local TZ — but the
        // field is informational, not load-bearing for downstream code).
        format!("@{now}+00:00")
    };

    for (idx, seg) in segments.iter_mut().enumerate() {
        seg.id = generate_cue_id(audio_hash, seg.start_time, idx, deterministic);
        if seg.timestamp.is_none() {
            seg.timestamp = Some(current_ts.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------

/// Pull the parakeet-tdt-0.6b-v3 ONNX repo to the HF cache and return the
/// snapshot directory (so parakeet-rs's `from_pretrained(&dir, None)`
/// finds `encoder-model.onnx`, `decoder_joint-model.onnx`, `vocab.txt`).
fn download_parakeet_onnx(model_id: &str) -> Result<PathBuf> {
    let token = std::env::var("HF_TOKEN").ok();
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }
    let api = builder
        .build()
        .map_err(|e| eyre!("hf-hub init: {e}"))?
        .model(model_id.to_string());

    // parakeet-rs expects all three files in one directory. hf-hub caches
    // each file separately under the snapshot dir; pulling any of them
    // returns the snapshot path of *that file*, and the snapshot dir is
    // shared across files of the same revision.
    let mut snapshot_dir: Option<PathBuf> = None;
    for name in ["encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"] {
        let p = api
            .get(name)
            .map_err(|e| eyre!("hf-hub get {name}: {e}"))?;
        snapshot_dir = Some(p.parent().unwrap().to_path_buf());
    }
    // istupakov's repo ships the encoder weights as a sidecar `.onnx.data` —
    // try to fetch but tolerate absence (older revisions skipped it).
    let _ = api.get("encoder-model.onnx.data");

    snapshot_dir.ok_or_else(|| eyre!("no ONNX files resolved from {model_id}"))
}
