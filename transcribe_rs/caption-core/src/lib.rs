//! Post-processing pipeline — Rust port of
//! `transcribe/asr_results_to_captions.py`.
//!
//! Pure-logic crate (no audio, no ML). Consumes `AsrSegment` values, produces
//! `AsrSegment` or `TranscriptSegment` values. Snapshot-tested against the
//! same fixtures the Python suite uses (`transcribe/test_fixtures/*.json`).
//!
//! Module map mirrors the Python callables:
//!   - `parse`     ← parse_{whisper,parakeet}_raw_chunk
//!   - `pipeline`  ← resolve_overlap_conflicts, group_segments_by_gap,
//!                   split_segments_by_word_gap, split_long_segments,
//!                   post_process_{,raw_}asr_segments
//!   - `convert`   ← asr_segments_to_transcript_segments,
//!                   raw_asr_segments_to_raw_asr_output

pub mod convert;
pub mod parse;
pub mod pipeline;

pub use caption_schema::{AsrSegment, WordTimestamp};
pub use convert::{asr_segments_to_transcript_segments, raw_asr_segments_to_raw_asr_output};
pub use parse::{parse_parakeet_raw_chunk, parse_whisper_raw_chunk};
pub use pipeline::{
    group_segments_by_gap, post_process_asr_segments, post_process_raw_asr_segments,
    resolve_overlap_conflicts, split_long_segments, split_segments_by_word_gap,
};
