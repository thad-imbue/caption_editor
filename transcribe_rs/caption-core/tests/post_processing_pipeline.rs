//! Rust mirror of
//! `transcribe/asr_results_to_captions_post_processing_pipeline_test.py`.
//!
//! Same fixtures, same pipeline arguments. Snapshots live alongside this
//! file as `__snapshots__/<test>.snap` (managed by `insta`). When the
//! Python and Rust pipelines agree, the *shape* of the snapshot files
//! (segment count, start/end/duration/text per segment) will match
//! 1:1 with the Python Amber files — that's the cross-language
//! parity check.
//!
//! Fixtures are embedded with `include_bytes!` so the test runs identically
//! under `cargo test` and `bazel test` (no runfiles dance).

use caption_core::{
    parse_parakeet_raw_chunk, parse_whisper_raw_chunk, post_process_asr_segments,
    raw_asr_segments_to_raw_asr_output, AsrSegment,
};
use serde::Serialize;
use serde_json::Value;

const WHISPER_10: &[u8] =
    include_bytes!("../../../transcribe/test_fixtures/whisper_chunked_10s_raw_output.json");
const WHISPER_60: &[u8] =
    include_bytes!("../../../transcribe/test_fixtures/whisper_chunked_60s_raw_output.json");
const PARAKEET_10: &[u8] =
    include_bytes!("../../../transcribe/test_fixtures/parakeet_chunked_10s_raw_output.json");
const PARAKEET_60: &[u8] =
    include_bytes!("../../../transcribe/test_fixtures/parakeet_chunked_60s_raw_output.json");

#[derive(Serialize)]
struct ProcessedSegment {
    start_time: f64,
    end_time: f64,
    duration: f64,
    text: String,
}

#[derive(Serialize)]
struct Processed {
    num_segments: usize,
    segments: Vec<ProcessedSegment>,
}

#[derive(Serialize)]
struct Payload {
    processed: Processed,
    #[serde(rename = "rawAsrOutput")]
    raw_asr_output: caption_schema::RawAsrOutput,
}

fn load_and_parse(
    bytes: &[u8],
    parser: fn(&Value, f64) -> Vec<AsrSegment>,
    chunk_size: f64,
    overlap: f64,
) -> Vec<AsrSegment> {
    let chunks: Vec<Value> = serde_json::from_slice(bytes).expect("fixture JSON");
    let mut out = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_start = (i as f64) * (chunk_size - overlap);
        out.extend(parser(chunk, chunk_start));
    }
    out
}

fn round_floats(value: &mut Value, digits: i32) {
    match value {
        // Only touch numbers that are actually floats — leave integers alone so
        // `num_segments` and `version` stay ints (matches the Python Amber output).
        Value::Number(n) if n.is_f64() => {
            if let Some(f) = n.as_f64() {
                if !f.is_nan() && !f.is_infinite() {
                    let factor = 10f64.powi(digits);
                    let rounded = (f * factor).round() / factor;
                    if let Some(num) = serde_json::Number::from_f64(rounded) {
                        *n = num;
                    }
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|v| round_floats(v, digits)),
        Value::Object(o) => o.values_mut().for_each(|v| round_floats(v, digits)),
        _ => {}
    }
}

fn build_payload(model: &str, chunk_size: i64) -> Value {
    let chunk_size_f = chunk_size as f64;
    let overlap = 5.0;
    let raw = match (model, chunk_size) {
        ("whisper", 10) => load_and_parse(WHISPER_10, parse_whisper_raw_chunk, chunk_size_f, overlap),
        ("whisper", 60) => load_and_parse(WHISPER_60, parse_whisper_raw_chunk, chunk_size_f, overlap),
        ("parakeet", 10) => load_and_parse(PARAKEET_10, parse_parakeet_raw_chunk, chunk_size_f, overlap),
        ("parakeet", 60) => load_and_parse(PARAKEET_60, parse_parakeet_raw_chunk, chunk_size_f, overlap),
        other => panic!("unknown case {other:?}"),
    };

    let gap_threshold = if model == "whisper" { 0.2 } else { 2.0 };
    let processed = post_process_asr_segments(
        raw.clone(),
        chunk_size_f,
        overlap,
        gap_threshold,
        10.0,
        model == "whisper",
    );

    let raw_asr_output = raw_asr_segments_to_raw_asr_output(&raw);

    let payload = Payload {
        processed: Processed {
            num_segments: processed.len(),
            segments: processed
                .iter()
                .map(|s| ProcessedSegment {
                    start_time: s.start_time,
                    end_time: s.end_time,
                    duration: s.end_time - s.start_time,
                    text: s.text.clone(),
                })
                .collect(),
        },
        raw_asr_output,
    };

    // Round floats to 5 digits before snapshotting — matches the Python
    // `rounded_floats_matcher(ndigits=5)` so trivial FP jitter doesn't fail
    // CI on either language.
    let mut value = serde_json::to_value(&payload).expect("serialize payload");
    round_floats(&mut value, 5);
    value
}

#[test]
fn whisper_chunk_10s() {
    insta::assert_json_snapshot!(build_payload("whisper", 10));
}

#[test]
fn whisper_chunk_60s() {
    insta::assert_json_snapshot!(build_payload("whisper", 60));
}

#[test]
fn parakeet_chunk_10s() {
    insta::assert_json_snapshot!(build_payload("parakeet", 10));
}

#[test]
fn parakeet_chunk_60s() {
    insta::assert_json_snapshot!(build_payload("parakeet", 60));
}
