//! Whisper (encoder-decoder) recognizer via whisper.cpp (whisper-rs).
//!
//! Emits one `AsrSegment` per whisper *word* (matching Python's
//! `parse_whisper_raw_chunk` convention). caption-core's
//! `group_segments_by_gap` (is_whisper=true) reassembles sentences
//! downstream.
//!
//! Model resolution:
//!   - `openai/whisper-tiny` / `openai/whisper-base` / ... → fetch the
//!     matching `ggml-<name>.bin` from `ggerganov/whisper.cpp`.
//!   - `<path>/whisper-something.bin` (or any local file path) → use as-is.
//!   - Anything else → assume it's a HF repo id with a `ggml-model.bin`
//!     inside (fallback).
//!
//! Token grouping:
//!   - Skip whisper special tokens (`<|startoftranscript|>`, `<|en|>`,
//!     `<|transcribe|>`, timestamp tokens `<|0.00|>`, etc.) — anything
//!     that whisper.cpp surfaces with `<|...|>` shape.
//!   - Group consecutive subword tokens into words on SentencePiece-style
//!     leading-space boundary (whisper's BPE prepends a space to word
//!     starts).

use crate::recognizer::Recognizer;
use caption_core::{AsrSegment, WordTimestamp};
use eyre::{eyre, Context, Result};
use hf_hub::api::sync::ApiBuilder;
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

pub struct WhisperRecognizer {
    // The C++ context owns model weights; state holds inference scratch
    // space and is reused across chunks for efficiency.
    state: WhisperState,
    language: Option<String>,
    // Stash the context to keep it alive (state borrows from it).
    _ctx: WhisperContext,
}

impl WhisperRecognizer {
    pub fn from_model(model_path: &Path, language: Option<&str>) -> Result<Self> {
        eprintln!("Loading whisper model: {}", model_path.display());
        let ctx = WhisperContext::new_with_params(
            model_path
                .to_str()
                .ok_or_else(|| eyre!("non-utf8 model path: {}", model_path.display()))?,
            WhisperContextParameters::default(),
        )
        .map_err(|e| eyre!("WhisperContext::new_with_params: {e}"))?;

        // SAFETY-ish: WhisperState holds a borrow on the WhisperContext.
        // We keep `ctx` alive by storing it next to `state` in the struct.
        // The borrow checker doesn't model this self-reference, so we
        // transmute the lifetime away. Same pattern whisper-rs's own
        // examples use for long-lived state.
        let state = ctx
            .create_state()
            .map_err(|e| eyre!("WhisperContext::create_state: {e}"))?;
        let state_static: WhisperState = unsafe { std::mem::transmute(state) };
        Ok(Self {
            state: state_static,
            language: language.map(str::to_string),
            _ctx: ctx,
        })
    }
}

impl Recognizer for WhisperRecognizer {
    fn is_whisper(&self) -> bool {
        true
    }

    fn transcribe_chunk(
        &mut self,
        samples: Vec<f32>,
        sample_rate: u32,
        _channels: u16,
        chunk_start_s: f64,
    ) -> Result<Vec<AsrSegment>> {
        if sample_rate != 16_000 {
            return Err(eyre!(
                "whisper requires 16 kHz mono audio; got {sample_rate} Hz"
            ));
        }

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        if let Some(lang) = self.language.as_deref() {
            params.set_language(Some(lang));
        }
        // Quiet stdout — we have our own progress logging.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Enable per-token timestamps so we can pull t0/t1 off each
        // WhisperTokenData below for word-level word_timing.
        params.set_token_timestamps(true);
        // Split on whitespace boundaries so segments don't smear across
        // word boundaries (matches Python NeMo's per-word output shape).
        params.set_split_on_word(true);

        self.state
            .full(params, &samples)
            .map_err(|e| eyre!("whisper full(): {e}"))?;

        let n_segments = self
            .state
            .full_n_segments()
            .map_err(|e| eyre!("full_n_segments: {e}"))?;

        let mut out: Vec<AsrSegment> = Vec::new();
        // For each whisper segment, walk its tokens and accumulate them
        // into one-word-per-AsrSegment entries — matching Python's
        // parse_whisper_raw_chunk shape.
        for s in 0..n_segments {
            let n_tokens = self
                .state
                .full_n_tokens(s)
                .map_err(|e| eyre!("full_n_tokens({s}): {e}"))?;

            // Buffer: (text, start_centisec, end_centisec) for tokens
            // belonging to the current in-progress word.
            let mut cur_text = String::new();
            let mut cur_start: i64 = 0;
            let mut cur_end: i64 = 0;

            let flush = |out: &mut Vec<AsrSegment>,
                         text: &str,
                         start_cs: i64,
                         end_cs: i64| {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    return;
                }
                // whisper.cpp timestamps are centiseconds since the start
                // of the chunk audio. Convert to seconds and offset by the
                // chunk's absolute start.
                let abs_start = chunk_start_s + (start_cs as f64) / 100.0;
                let abs_end = chunk_start_s + (end_cs as f64) / 100.0;
                let word = WordTimestamp {
                    word: trimmed.to_string(),
                    start: abs_start,
                    end: abs_end,
                };
                out.push(AsrSegment {
                    text: trimmed.to_string(),
                    start: abs_start,
                    end: abs_end,
                    words: vec![word],
                    chunk_start: Some(chunk_start_s),
                    speaker: None,
                });
            };

            for t in 0..n_tokens {
                let tok_text = self
                    .state
                    .full_get_token_text(s, t)
                    .map_err(|e| eyre!("full_get_token_text({s},{t}): {e}"))?;
                // Filter whisper special tokens. whisper.cpp surfaces them
                // in either the original `<|...|>` form or its own
                // bracket-and-underscore alias (`[_BEG_]`, `[_TT_200]`,
                // `[_EOT_]`, etc.). Drop both.
                let is_special = (tok_text.starts_with('<')
                    && tok_text.ends_with('>'))
                    || (tok_text.starts_with("[_") && tok_text.ends_with(']'));
                if is_special {
                    continue;
                }
                let data = self
                    .state
                    .full_get_token_data(s, t)
                    .map_err(|e| eyre!("full_get_token_data({s},{t}): {e}"))?;

                // Leading space (or first token of a new word) → flush
                // and start a fresh word. whisper.cpp's BPE uses a space
                // prefix to mark word starts; with set_split_on_word(true)
                // each tokenized segment line tends to be one word, but
                // we still defensively glue subword pieces.
                let starts_word = tok_text.starts_with(' ') || cur_text.is_empty();
                if starts_word && !cur_text.is_empty() {
                    flush(&mut out, &cur_text, cur_start, cur_end);
                    cur_text.clear();
                }
                if cur_text.is_empty() {
                    cur_start = data.t0 as i64;
                }
                cur_end = data.t1 as i64;
                cur_text.push_str(&tok_text);
            }
            if !cur_text.is_empty() {
                flush(&mut out, &cur_text, cur_start, cur_end);
            }
        }

        Ok(out)
    }
}

/// Resolve the user-facing model id (`openai/whisper-tiny`, etc.) to a
/// local path to a whisper.cpp GGML `.bin` file. Downloads via hf-hub
/// if needed.
pub fn resolve_whisper_model_path(model_id: &str) -> Result<PathBuf> {
    // If they passed a local path to a .bin file directly, use it as-is.
    let local = PathBuf::from(model_id);
    if local.is_file() {
        return Ok(local);
    }

    // openai/whisper-<size> → ggerganov/whisper.cpp:ggml-<size>.bin
    // We accept a few of the well-known sizes; anything else falls
    // through to the "give us the literal HF id + filename" path below.
    if let Some(size) = model_id.strip_prefix("openai/whisper-") {
        let filename = format!("ggml-{size}.bin");
        return fetch_from_hf("ggerganov/whisper.cpp", &filename);
    }

    // Direct ggerganov-style ids (e.g. ggerganov/whisper.cpp) — assume
    // ggml-model.bin or ggml-tiny.bin etc. and hope the caller knows.
    // Most users won't hit this path; print a clear error if it fails.
    fetch_from_hf(model_id, "ggml-model.bin").or_else(|_| {
        Err(eyre!(
            "Could not resolve whisper model id {:?}. Try a known alias like \
             `openai/whisper-tiny` or pass an absolute path to a GGML .bin file.",
            model_id,
        ))
    })
}

fn fetch_from_hf(repo: &str, filename: &str) -> Result<PathBuf> {
    let token = std::env::var("HF_TOKEN").ok();
    let mut builder = ApiBuilder::new();
    if let Some(t) = token {
        builder = builder.with_token(Some(t));
    }
    let api = builder
        .build()
        .map_err(|e| eyre!("hf-hub init: {e}"))?
        .model(repo.to_string());
    api.get(filename)
        .with_context(|| format!("hf-hub get {repo}:{filename}"))
}
