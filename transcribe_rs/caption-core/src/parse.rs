//! Parsers for raw chunked ASR JSON — port of
//! `parse_whisper_raw_chunk` / `parse_parakeet_raw_chunk` in
//! `transcribe/asr_results_to_captions.py`.
//!
//! Input shape: a single chunk's raw output as `serde_json::Value` (we keep
//! it untyped so we don't lock in the exact ASR-runtime JSON shape — Whisper
//! and Parakeet emit different schemas under the same `segments`/`words`
//! keys). Output: a list of `AsrSegment` with `chunk_start` carried so the
//! later overlap-merge pass can compute midpoints correctly.

use caption_schema::{AsrSegment, WordTimestamp};
use serde_json::Value;

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64().or_else(|| v.as_i64().map(|i| i as f64))
}

fn trimmed_str<'a>(v: &'a Value) -> Option<&'a str> {
    v.as_str().map(|s| s.trim()).filter(|s| !s.is_empty())
}

/// Whisper raw chunk → list of single-word `AsrSegment`s.
///
/// Whisper's per-chunk output has a `segments` array where each entry is
/// actually word-level. We mirror the Python `parse_whisper_raw_chunk`:
/// drop entries missing `text`/`start`/`end`, emit one tiny segment per
/// word so the downstream gap-grouping pass can rejoin them into sentences.
pub fn parse_whisper_raw_chunk(chunk: &Value, chunk_start: f64) -> Vec<AsrSegment> {
    let Some(segments) = chunk.get("segments").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(segments.len());
    for word in segments {
        let Some(text) = word.get("text").and_then(trimmed_str) else {
            continue;
        };
        let Some(start) = word.get("start").and_then(as_f64) else {
            continue;
        };
        let Some(end) = word.get("end").and_then(as_f64) else {
            continue;
        };
        let w = WordTimestamp {
            word: text.to_string(),
            start: chunk_start + start,
            end: chunk_start + end,
        };
        out.push(AsrSegment {
            text: text.to_string(),
            start: chunk_start + start,
            end: chunk_start + end,
            words: vec![w],
            // Matches Python: parse_*_raw_chunk leaves `chunk_start` unset.
            // The live transcribe driver (transcribe_cli.transcribe_audio_file)
            // assigns it afterwards; the snapshot tests parse captured fixtures
            // and never set it, so the overlap-merge falls back to the
            // segment-times midpoint instead of the chunk-region midpoint.
            chunk_start: None,
            speaker: None,
        });
    }
    out
}

/// Parakeet raw chunk → list of sentence-level `AsrSegment`s.
///
/// Parakeet emits sentence-level `segments` and word-level `words` separately.
/// We match words to segments by time-range overlap, mirroring the Python
/// 0.01s tolerance (kept identical so snapshots match).
pub fn parse_parakeet_raw_chunk(chunk: &Value, chunk_start: f64) -> Vec<AsrSegment> {
    let Some(sentences) = chunk.get("segments").and_then(Value::as_array) else {
        return Vec::new();
    };
    let empty = Vec::new();
    let all_words = chunk
        .get("words")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    let mut out = Vec::with_capacity(sentences.len());
    for seg in sentences {
        let Some(text) = seg.get("text").and_then(trimmed_str) else {
            continue;
        };
        let Some(seg_start) = seg.get("start").and_then(as_f64) else {
            continue;
        };
        let Some(seg_end) = seg.get("end").and_then(as_f64) else {
            continue;
        };

        let mut seg_words = Vec::new();
        for word in all_words {
            let Some(wt) = word.get("word").and_then(trimmed_str) else {
                continue;
            };
            let Some(ws) = word.get("start").and_then(as_f64) else {
                continue;
            };
            let Some(we) = word.get("end").and_then(as_f64) else {
                continue;
            };
            // Same 0.01s tolerance as the Python implementation.
            if ws >= seg_start - 0.01 && we <= seg_end + 0.01 {
                seg_words.push(WordTimestamp {
                    word: wt.to_string(),
                    start: chunk_start + ws,
                    end: chunk_start + we,
                });
            }
        }
        out.push(AsrSegment {
            text: text.to_string(),
            start: chunk_start + seg_start,
            end: chunk_start + seg_end,
            words: seg_words,
            // Matches Python: parse_*_raw_chunk leaves `chunk_start` unset.
            // The live transcribe driver (transcribe_cli.transcribe_audio_file)
            // assigns it afterwards; the snapshot tests parse captured fixtures
            // and never set it, so the overlap-merge falls back to the
            // segment-times midpoint instead of the chunk-region midpoint.
            chunk_start: None,
            speaker: None,
        });
    }
    out
}
