//! Post-processing pipeline — Rust port of
//! `transcribe/asr_results_to_captions.py`.
//!
//! Filled in by Phase 3 of the port. Public surface mirrors the Python
//! callables that downstream code (`transcribe_cli`, the post-processing
//! snapshot test) actually consumes:
//!   - `post_process_raw_asr_segments`
//!   - `post_process_asr_segments`
//!   - `asr_segments_to_transcript_segments`
//!   - `raw_asr_segments_to_raw_asr_output`
