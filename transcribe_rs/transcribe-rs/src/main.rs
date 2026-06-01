//! Drop-in Rust replacement for `transcribe/transcribe_cli.py` (spike, not prod).
//!
//! Goal: take a WAV, run Parakeet-TDT via parakeet-rs, and emit the same raw
//! ASRSegment JSON the Python `--dump-raw-asr` mode produces, so the existing
//! Python snapshot tests in `asr_results_to_captions_post_processing_pipeline_test.py`
//! can be retargeted at this binary as a second source of truth.
//!
//! Not implemented yet (spike scope):
//!   - chunked driver (parakeet-rs caps TDT at ~5min/call — needs the same
//!     60s-with-5s-overlap loop the Python `transcribe_audio_file` uses).
//!   - hf-hub download of `istupakov/parakeet-tdt-0.6b-v3-onnx` (currently
//!     expects a local `--model-dir` path).
//!   - audio extraction from non-WAV containers (Python uses ffmpeg via
//!     `audio_utils.extract_audio_to_wav`).

use caption_schema::{AsrSegment, WordTimestamp};
use clap::Parser;
use eyre::Result;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Input WAV file (16kHz mono).
    wav: PathBuf,
    /// Directory containing the parakeet-tdt ONNX files
    /// (encoder-model.onnx, decoder_joint-model.onnx, vocab.txt).
    #[clap(long)]
    model_dir: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut reader = hound::WavReader::open(&args.wav)?;
    let spec = reader.spec();
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<Result<Vec<_>, _>>()?,
    };

    let mut parakeet = ParakeetTDT::from_pretrained(&args.model_dir, None)?;
    let result = parakeet.transcribe_samples(
        audio,
        spec.sample_rate,
        spec.channels,
        Some(TimestampMode::Words),
    )?;

    // parakeet-rs returns word-mode tokens with start/end. Roll them up into
    // a single ASRSegment for now; chunked driver + sentence segmentation
    // come in the real port (see asr_results_to_captions.parse_nemo_segment).
    let words: Vec<WordTimestamp> = result
        .tokens
        .iter()
        .map(|t| WordTimestamp {
            word: t.text.clone(),
            start: t.start as f64,
            end: t.end as f64,
        })
        .collect();

    let (start, end) = words
        .first()
        .zip(words.last())
        .map(|(a, b)| (a.start, b.end))
        .unwrap_or((0.0, 0.0));

    let seg = AsrSegment {
        text: result.text,
        start,
        end,
        words,
        chunk_start: Some(0.0),
        speaker: None,
    };

    println!("{}", serde_json::to_string_pretty(&[seg])?);
    Ok(())
}
