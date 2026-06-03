//! Inline port of parakeet-rs's `process_timestamps` helpers.
//!
//! Reason for vendoring: the upstream `group_by_words` and `group_by_sentences`
//! are `pub(crate)` and the wrapper `process_timestamps` is `pub` inside a
//! private `timestamps` module, so they aren't reachable from outside the
//! crate. We need *both* word- and sentence-level groupings from a single raw
//! token list (so we don't have to run the model twice per chunk), which the
//! public API forces a model pass per mode.
//!
//! Source: parakeet-rs v0.3.5 src/timestamps.rs (MIT OR Apache-2.0). Logic
//! preserved verbatim; only the `TimedToken` type alias changes (we use
//! parakeet-rs's re-exported type via `parakeet_rs::TimedToken`).

use parakeet_rs::TimedToken;

/// Group raw subword tokens into words.
///
/// Rules (diverges from upstream parakeet-rs to give cleaner semantics):
///   - SentencePiece marker (▁) or leading space → new word.
///   - Pure-punctuation tokens (`.`, `?`, `!`, `,`, etc.) **attach to the
///     previous word's text** but do *not* affect its timing. So a period
///     coming after "Tim" produces a single word `{text: "Tim.", start: T0,
///     end: T1}` where T1 is the end of the spoken "Tim" sound, not the
///     end of the punctuation token's extent.
///   - A pure-punctuation token with no preceding content is dropped.
///   - Apostrophes and hyphens still attach (so contractions and hyphenated
///     compounds stay together).
///
/// Why this differs from upstream parakeet-rs (which emits standalone `.`
/// words) and from NeMo's PyTorch path (which DOES attach but uses the
/// punctuation token's end as the word end):
///
/// 1. Punctuation is typographic — there's no "audio" being pronounced. A
///    word's `[start, end]` should be the duration of the spoken sound, not
///    silence plus a marker glued onto the end. Burying chunk-boundary
///    silence into the word's end-time was the proximate cause of our
///    spurious overlap merges (fixed in the previous commit at the sentence
///    level; this finishes the same job at the word level).
/// 2. `words[].text` still concatenates back to the formatted sentence
///    (`" ".join(...)` of `["My", "name", "is", "Tim."]` is
///    `"My name is Tim."`), so the data model is internally consistent.
pub fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current_word_text = String::new();
    let mut current_word_start = 0.0;
    // `last_content_end` is the end-time of the last *non-punctuation* token
    // we accumulated into the current word. We close the word with this
    // value rather than `tokens[i-1].end` so trailing punctuation doesn't
    // stretch the word's timestamp into the silence after.
    let mut last_content_end = 0.0;
    let mut last_word_lower = String::new();

    // Helper to emit (or dedup) the current word. Returns true if a word
    // was actually pushed; resets the accumulator either way.
    let flush_word =
        |words: &mut Vec<TimedToken>,
         text: &mut String,
         start: f32,
         end: f32,
         last_lower: &mut String| {
            if text.is_empty() {
                return;
            }
            // Drop entries that are pure punctuation with no content (no
            // spoken word attached): no audio, no place in the words list.
            if text.chars().all(|c| c.is_ascii_punctuation()) {
                text.clear();
                return;
            }
            let word_lower = text.to_lowercase();
            if word_lower != *last_lower {
                words.push(TimedToken {
                    text: text.clone(),
                    start,
                    end,
                });
                *last_lower = word_lower;
            }
            text.clear();
        };

    for (i, token) in tokens.iter().enumerate() {
        // Whitespace-only tokens act as word boundaries (and contribute no text).
        if token.text.trim().is_empty() {
            flush_word(
                &mut words,
                &mut current_word_text,
                current_word_start,
                last_content_end,
                &mut last_word_lower,
            );
            continue;
        }

        let is_pure_punctuation =
            !token.text.is_empty() && token.text.chars().all(|c| c.is_ascii_punctuation());

        let token_without_marker = token.text.trim_start_matches('▁').trim_start_matches(' ');
        let is_contraction = token_without_marker.starts_with('\'');
        let is_hyphenation = token_without_marker.starts_with('-');

        // Pure-punctuation NO LONGER starts a new word. It attaches to the
        // previous word's text (and is dropped if there's nothing to attach to).
        let starts_word = (token.text.starts_with('▁') || token.text.starts_with(' '))
            && !is_contraction
            && !is_hyphenation
            || i == 0;

        if starts_word && !current_word_text.is_empty() {
            flush_word(
                &mut words,
                &mut current_word_text,
                current_word_start,
                last_content_end,
                &mut last_word_lower,
            );
        }

        if current_word_text.is_empty() {
            // For a first-of-word punctuation that survived to here (i==0
            // case), there's nothing to attach to — give it the token's
            // start so a later content token sets the real start.
            current_word_start = token.start;
        }

        let token_text = token.text.trim_start_matches('▁').trim_start_matches(' ');
        current_word_text.push_str(token_text);

        // Punctuation contributes text but NOT timing — keep last_content_end.
        if !is_pure_punctuation {
            last_content_end = token.end;
            // Edge case: if the very first token of the word was punctuation
            // (we kept its start as current_word_start above), but the first
            // real content arrives now, anchor the start here so the word's
            // [start, end] still describes spoken audio only.
            if current_word_text.trim_end_matches(|c: char| c.is_ascii_punctuation())
                == token_text
            {
                current_word_start = token.start;
            }
        }
    }

    flush_word(
        &mut words,
        &mut current_word_text,
        current_word_start,
        last_content_end,
        &mut last_word_lower,
    );

    words
}

/// Group words into sentences by terminator punctuation (`. ? !`).
pub fn group_by_sentences(tokens: &[TimedToken]) -> Vec<TimedToken> {
    let words = group_by_words(tokens);
    if words.is_empty() {
        return Vec::new();
    }

    let mut sentences = Vec::new();
    let mut current_sentence: Vec<TimedToken> = Vec::new();

    for word in words {
        current_sentence.push(word.clone());

        let ends_sentence =
            word.text.contains('.') || word.text.contains('?') || word.text.contains('!');

        if ends_sentence {
            let sentence_text = format_sentence(&current_sentence);
            let start = current_sentence.first().unwrap().start;
            // NeMo gives punctuation tokens zero duration (end == start), so
            // a NeMo segment ends at the end of the last *spoken* word.
            // parakeet-rs makes the punctuation token span from the end of
            // the previous token to the start of the next, which artificially
            // extends our sentence end by 0.1–0.3s. That extra slop makes
            // adjacent sentences look like they overlap in
            // resolve_overlap_conflicts, triggering spurious merges across
            // chunk boundaries. Use the prior word's end (i.e. the start of
            // the terminator) so our sentence intervals match NeMo's.
            let end = sentence_end_excluding_terminator(&current_sentence);

            if !sentence_text.is_empty() {
                sentences.push(TimedToken {
                    text: sentence_text,
                    start,
                    end,
                });
            }
            current_sentence.clear();
        }
    }

    if !current_sentence.is_empty() {
        let sentence_text = format_sentence(&current_sentence);
        let start = current_sentence.first().unwrap().start;
        // Trailing-fragment case (no terminator at all): keep the last word's
        // actual end; there's no spurious-punctuation slop to trim.
        let end = current_sentence.last().unwrap().end;

        if !sentence_text.is_empty() {
            sentences.push(TimedToken {
                text: sentence_text,
                start,
                end,
            });
        }
    }

    sentences
}

/// End time for a sentence that ends in punctuation. Trims the punctuation
/// token's extent (parakeet-rs assigns it the gap between the previous and
/// next tokens; NeMo treats it as zero-duration). Walks backward from the
/// end of `words` skipping pure-punctuation entries; returns the end of the
/// first content word. Falls back to the last word's end (matches NeMo
/// when no content word precedes the terminator — e.g. a chunk that starts
/// mid-sentence with just `.`).
fn sentence_end_excluding_terminator(words: &[TimedToken]) -> f32 {
    for w in words.iter().rev() {
        if !is_pure_punctuation(&w.text) {
            return w.end;
        }
    }
    words.last().map(|w| w.end).unwrap_or(0.0)
}

fn is_pure_punctuation(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_punctuation())
}

/// Join words with punctuation spacing (no space before `. , ! ? ; : )`).
fn format_sentence(words: &[TimedToken]) -> String {
    let mut output = String::new();
    for (i, word) in words.iter().enumerate() {
        let is_standalone_punct = word.text.len() == 1
            && word
                .text
                .chars()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
        if i > 0 && !is_standalone_punct {
            output.push(' ');
        }
        output.push_str(&word.text);
    }
    output
}
