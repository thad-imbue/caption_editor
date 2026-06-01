//! Spike replacement for `transcribe/embed_cli.py`.
//!
//! Computes a wespeaker-voxceleb-resnet34-LM embedding for a WAV using `ort`
//! 2.0 + `knf-rs` (kaldi-native-fbank). This is the same recipe pyannote-rs
//! uses; we inline the ~30 lines of session/inference glue so we can stay on
//! ort 2.0-rc.12 (matching parakeet-rs) and avoid pyannote-rs's stale rc.10
//! pin.

use clap::Parser;
use eyre::{Context, ContextCompat, Result};
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// Input WAV file (16kHz mono, single speaker segment).
    wav: PathBuf,
    /// Path to wespeaker-voxceleb-resnet34-LM ONNX file.
    #[clap(long)]
    model: PathBuf,
}

fn create_session(path: &std::path::Path) -> Result<Session> {
    // Map ort's parameterized SessionBuilder errors to plain eyre — rc.12's
    // `Error<SessionBuilder>` isn't Send+Sync because of operator boxes, so
    // `?` -> eyre auto-conversion fails (this is what bit pyannote-rs).
    let s = Session::builder()
        .map_err(|e| eyre::eyre!("ort session builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| eyre::eyre!("opt level: {e}"))?
        .with_intra_threads(1)
        .map_err(|e| eyre::eyre!("intra threads: {e}"))?
        .with_inter_threads(1)
        .map_err(|e| eyre::eyre!("inter threads: {e}"))?
        .commit_from_file(path)
        .map_err(|e| eyre::eyre!("commit_from_file: {e}"))?;
    Ok(s)
}

fn read_wav_i16(path: &std::path::Path) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::open(path)?;
    Ok(reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?)
}

fn compute_embedding(session: &mut Session, samples_i16: &[i16]) -> Result<Vec<f32>> {
    let mut samples_f32 = vec![0.0f32; samples_i16.len()];
    knf_rs::convert_integer_to_float_audio(samples_i16, &mut samples_f32);

    // knf-rs pins ndarray 0.16; ort rc.12 pulls 0.17. Cross the version gap
    // by copying through raw shape + Vec — tiny tensor, irrelevant overhead.
    let feats_v016 = knf_rs::compute_fbank(&samples_f32)?;
    let (n_frames, n_bins) = feats_v016.dim();
    let flat: Vec<f32> = feats_v016.into_raw_vec();
    let feats: Array2<f32> = Array2::from_shape_vec((n_frames, n_bins), flat)?;
    let feats = feats.insert_axis(ndarray::Axis(0)); // batch dim

    let inputs = ort::inputs!["feats" => Tensor::from_array(feats)?];
    let outs = session.run(inputs)?;
    let out = outs
        .get("embs")
        .context("output 'embs' missing")?
        .try_extract_tensor::<f32>()
        .context("extract embs")?;
    Ok(out.1.iter().copied().collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let samples = read_wav_i16(&args.wav)?;
    let mut session = create_session(&args.model)?;
    let emb = compute_embedding(&mut session, &samples)?;
    println!("{}", serde_json::to_string(&emb)?);
    Ok(())
}
