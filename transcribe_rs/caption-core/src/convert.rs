//! AsrSegment → TranscriptSegment / RawAsrOutput. Port of the same-named
//! helpers in `transcribe/asr_results_to_captions.py`.
//!
//! Note: this Rust port leaves `id` empty and `index` at 0 to match the
//! Python contract — IDs (a deterministic hash) and indices are assigned
//! later by the CLI driver, not by this pure-logic crate.

use caption_schema::{
    AsrSegment, RawAsrOutput, RawAsrSegmentSnapshot, RawAsrWord, TranscriptSegment, TranscriptWord,
};

pub fn asr_segments_to_transcript_segments(
    segments: Vec<AsrSegment>,
    asr_model: Option<&str>,
) -> Vec<TranscriptSegment> {
    let mut out = Vec::with_capacity(segments.len());
    for seg in segments {
        let trimmed = seg.text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let words = if seg.words.is_empty() {
            None
        } else {
            Some(
                seg.words
                    .iter()
                    .map(|w| TranscriptWord {
                        text: w.word.clone(),
                        start_time: Some(w.start),
                        end_time: Some(w.end),
                    })
                    .collect::<Vec<_>>(),
            )
        };
        out.push(TranscriptSegment {
            id: String::new(),
            index: 0,
            start_time: seg.start,
            end_time: seg.end,
            text: trimmed.to_string(),
            words,
            speaker_name: seg.speaker.clone(),
            rating: None,
            timestamp: None,
            verified: Some(false),
            asr_model: asr_model.map(str::to_string),
            notes: None,
        });
    }
    out
}

pub fn raw_asr_segments_to_raw_asr_output(segments: &[AsrSegment]) -> RawAsrOutput {
    RawAsrOutput {
        version: 1,
        segments: segments
            .iter()
            .map(|s| RawAsrSegmentSnapshot {
                text: s.text.clone(),
                start: s.start,
                end: s.end,
                chunk_start: s.chunk_start,
                words: s
                    .words
                    .iter()
                    .map(|w| RawAsrWord {
                        word: w.word.clone(),
                        start: w.start,
                        end: w.end,
                    })
                    .collect(),
            })
            .collect(),
    }
}
