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

/// Group raw subword tokens into words. SentencePiece marker (▁) or leading
/// space starts a new word; trailing punctuation gets attached.
pub fn group_by_words(tokens: &[TimedToken]) -> Vec<TimedToken> {
    if tokens.is_empty() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current_word_text = String::new();
    let mut current_word_start = 0.0;
    let mut last_word_lower = String::new();

    for (i, token) in tokens.iter().enumerate() {
        if token.text.trim().is_empty() {
            if !current_word_text.is_empty() {
                let word_lower = current_word_text.to_lowercase();
                if word_lower != last_word_lower {
                    words.push(TimedToken {
                        text: current_word_text.clone(),
                        start: current_word_start,
                        end: if i > 0 { tokens[i - 1].end } else { token.end },
                    });
                    last_word_lower = word_lower;
                }
                current_word_text.clear();
            }
            continue;
        }

        let is_pure_punctuation =
            !token.text.is_empty() && token.text.chars().all(|c| c.is_ascii_punctuation());

        let token_without_marker = token.text.trim_start_matches('▁').trim_start_matches(' ');
        let is_contraction = token_without_marker.starts_with('\'');
        let is_hyphenation = token_without_marker.starts_with('-');

        let starts_word =
            (token.text.starts_with('▁') || token.text.starts_with(' ') || is_pure_punctuation)
                && !is_contraction
                && !is_hyphenation
                || i == 0;

        if starts_word && !current_word_text.is_empty() {
            let word_lower = current_word_text.to_lowercase();
            if word_lower != last_word_lower {
                words.push(TimedToken {
                    text: current_word_text.clone(),
                    start: current_word_start,
                    end: tokens[i - 1].end,
                });
                last_word_lower = word_lower;
            }
            current_word_text.clear();
        }

        if current_word_text.is_empty() {
            current_word_start = token.start;
        }

        let token_text = token.text.trim_start_matches('▁').trim_start_matches(' ');
        current_word_text.push_str(token_text);
    }

    if !current_word_text.is_empty() {
        let word_lower = current_word_text.to_lowercase();
        if word_lower != last_word_lower {
            words.push(TimedToken {
                text: current_word_text,
                start: current_word_start,
                end: tokens.last().unwrap().end,
            });
        }
    }

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
