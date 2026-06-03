//! Recognizer abstraction over Parakeet (RNN-T/TDT via parakeet-rs) and
//! Whisper (encoder-decoder via whisper.cpp).
//!
//! Both engines run per-chunk and emit `AsrSegment`s into the same
//! `caption-core` post-processing pipeline. The differences they expose
//! to the caller:
//!
//!   - `transcribe_chunk`: feed 16 kHz mono f32 samples for a single
//!     chunk, get back a list of `AsrSegment`s (with absolute timing,
//!     chunk_start set so `resolve_overlap_conflicts` can compute
//!     midpoints).
//!   - `is_whisper`: drives the `is_whisper` flag passed into
//!     `post_process_raw_asr_segments`. Whisper emits one segment per
//!     word (matching the Python `parse_whisper_raw_chunk` path) so
//!     post-processing needs `group_segments_by_gap`; Parakeet emits
//!     already-grouped sentences and needs `split_segments_by_word_gap`.

use caption_core::AsrSegment;
use eyre::Result;

pub trait Recognizer {
    fn transcribe_chunk(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        chunk_start_s: f64,
    ) -> Result<Vec<AsrSegment>>;

    /// True when the recognizer emits word-level segments that need
    /// `group_segments_by_gap` to reassemble into sentences.
    fn is_whisper(&self) -> bool;
}

/// Pick a recognizer family from the user-facing model id. The model id
/// comes from `--model` (or the renderer's `__ASR_MODEL_OVERRIDE`).
///
/// Matching is intentionally lenient ("whisper" anywhere in the id wins)
/// so both `openai/whisper-tiny` and a local path like
/// `/tmp/ggml-whisper-large-v3.bin` route to Whisper.
pub fn is_whisper_model(model_id: &str) -> bool {
    model_id.to_ascii_lowercase().contains("whisper")
}
