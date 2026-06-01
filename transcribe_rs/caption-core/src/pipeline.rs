//! Three-pass post-processing pipeline:
//! overlap-merge → gap-split-or-group → long-segment-split.
//!
//! Port of:
//!   - resolve_overlap_conflicts + zip_words_in_overlapping_segments
//!   - group_segments_by_gap        (Whisper path: rejoin words → sentences)
//!   - split_segments_by_word_gap   (Parakeet path: split sentences on big gaps)
//!   - split_long_segments
//!   - post_process_raw_asr_segments / post_process_asr_segments
//!
//! Float math is byte-for-byte identical to Python (same arithmetic, same
//! comparisons) so the snapshot tests can compare against the same fixtures.

use caption_schema::{AsrSegment, WordTimestamp};

fn join_words(words: &[WordTimestamp]) -> String {
    words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Resolve overlaps from chunked transcription. Segments sorted by start
/// time; if two consecutive results overlap in time, merge them by splitting
/// at the midpoint of the chunk-overlap region.
pub fn resolve_overlap_conflicts(
    segments: Vec<AsrSegment>,
    chunk_size: f64,
    overlap: f64,
) -> Vec<AsrSegment> {
    let mut sorted = segments;
    sorted.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<AsrSegment> = Vec::with_capacity(sorted.len());
    for seg in sorted {
        match result.last() {
            None => result.push(seg),
            Some(prev) if prev.end <= seg.start => result.push(seg),
            Some(_) => {
                let prev = result.pop().expect("just matched Some");
                result.push(zip_words_in_overlapping_segments(
                    prev, seg, chunk_size, overlap,
                ));
            }
        }
    }
    result
}

/// Merge two overlapping segments. Words from the earlier chunk are kept up
/// to the midpoint of the overlap window; words from the later chunk are
/// kept from the midpoint onward.
pub fn zip_words_in_overlapping_segments(
    mut seg1: AsrSegment,
    mut seg2: AsrSegment,
    _chunk_size: f64,
    overlap: f64,
) -> AsrSegment {
    // Same swap rule as the Python: ensure seg1 is the earlier chunk so the
    // midpoint formula (seg2.chunk_start + overlap/2) works.
    if let (Some(c1), Some(c2)) = (seg1.chunk_start, seg2.chunk_start) {
        if c1 > c2 {
            std::mem::swap(&mut seg1, &mut seg2);
        }
    }

    let midpoint = match (seg1.chunk_start, seg2.chunk_start) {
        (Some(_), Some(c2)) => c2 + overlap / 2.0,
        _ => (seg2.start + seg1.end) / 2.0,
    };

    let mut merged: Vec<WordTimestamp> = seg1
        .words
        .iter()
        .filter(|w| w.start < midpoint)
        .cloned()
        .collect();
    merged.extend(seg2.words.iter().filter(|w| w.start >= midpoint).cloned());

    AsrSegment {
        text: join_words(&merged),
        start: seg1.start.min(seg2.start),
        end: seg1.end.max(seg2.end),
        words: merged,
        // Drop chunk_start on the merged result (matches the Python — its
        // ASRSegment(...) constructor in zip_words_in_overlapping_segments
        // doesn't pass chunk_start, so it defaults to None).
        chunk_start: None,
        speaker: None,
    }
}

/// Whisper-style: group adjacent single-word segments when the inter-word
/// gap is small enough (default 0.5s in Python). Pre-sorted by start time
/// to match the Python.
pub fn group_segments_by_gap(segments: Vec<AsrSegment>, max_gap_seconds: f64) -> Vec<AsrSegment> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut sorted = segments;
    sorted.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<AsrSegment> = Vec::new();
    let mut current: Vec<AsrSegment> = vec![sorted[0].clone()];

    for i in 1..sorted.len() {
        let prev = &sorted[i - 1];
        let curr = &sorted[i];
        let gap = curr.start - prev.end;

        if gap <= max_gap_seconds {
            current.push(curr.clone());
        } else {
            result.push(flush_group(&current));
            current = vec![curr.clone()];
        }
    }

    if !current.is_empty() {
        result.push(flush_group(&current));
    }

    result
}

fn flush_group(group: &[AsrSegment]) -> AsrSegment {
    let mut all_words = Vec::new();
    for seg in group {
        all_words.extend(seg.words.iter().cloned());
    }
    AsrSegment {
        text: join_words(&all_words),
        start: group.first().expect("non-empty group").start,
        end: group.last().expect("non-empty group").end,
        words: all_words,
        chunk_start: None,
        speaker: None,
    }
}

/// Parakeet-style: split sentence segments wherever a word-to-word gap
/// exceeds `max_gap_seconds`.
pub fn split_segments_by_word_gap(
    segments: Vec<AsrSegment>,
    max_gap_seconds: f64,
) -> Vec<AsrSegment> {
    let mut result = Vec::new();
    for segment in segments {
        if segment.words.len() <= 1 {
            result.push(segment);
            continue;
        }

        let mut split_indices = Vec::new();
        for i in 0..segment.words.len() - 1 {
            let gap = segment.words[i + 1].start - segment.words[i].end;
            if gap > max_gap_seconds {
                split_indices.push(i + 1);
            }
        }

        if split_indices.is_empty() {
            result.push(segment);
            continue;
        }

        let mut boundaries = Vec::with_capacity(split_indices.len() + 2);
        boundaries.push(0);
        boundaries.extend(split_indices.iter().copied());
        boundaries.push(segment.words.len());

        for pair in boundaries.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let sub_words = segment.words[lo..hi].to_vec();
            if sub_words.is_empty() {
                continue;
            }
            result.push(AsrSegment {
                text: join_words(&sub_words),
                start: sub_words.first().unwrap().start,
                end: sub_words.last().unwrap().end,
                words: sub_words,
                chunk_start: None,
                speaker: None,
            });
        }
    }
    result
}

/// Greedy split of segments whose duration exceeds `max_duration_seconds`.
/// New cut points fall after the last word whose end time keeps the
/// running duration under the threshold.
pub fn split_long_segments(
    segments: Vec<AsrSegment>,
    max_duration_seconds: f64,
) -> Vec<AsrSegment> {
    let mut result = Vec::new();

    for segment in segments {
        let duration = segment.end - segment.start;
        if duration <= max_duration_seconds || segment.words.len() <= 1 {
            result.push(segment);
            continue;
        }

        let mut current: Vec<WordTimestamp> = Vec::new();
        for word in &segment.words {
            if let Some(first) = current.first() {
                let potential = word.end - first.start;
                if potential > max_duration_seconds {
                    result.push(AsrSegment {
                        text: join_words(&current),
                        start: current.first().unwrap().start,
                        end: current.last().unwrap().end,
                        words: current.clone(),
                        chunk_start: None,
                        speaker: None,
                    });
                    current = vec![word.clone()];
                } else {
                    current.push(word.clone());
                }
            } else {
                current.push(word.clone());
            }
        }

        if !current.is_empty() {
            result.push(AsrSegment {
                text: join_words(&current),
                start: current.first().unwrap().start,
                end: current.last().unwrap().end,
                words: current,
                chunk_start: None,
                speaker: None,
            });
        }
    }
    result
}

/// Full pipeline: overlap merge → gap split/group → long-segment split.
/// Returns post-processed `AsrSegment`s (still pre-`TranscriptSegment`).
#[allow(clippy::too_many_arguments)]
pub fn post_process_raw_asr_segments(
    segments: Vec<AsrSegment>,
    chunk_size: f64,
    overlap: f64,
    max_intra_segment_gap_seconds: f64,
    max_segment_duration_seconds: f64,
    is_whisper: bool,
) -> Vec<AsrSegment> {
    let s = resolve_overlap_conflicts(segments, chunk_size, overlap);
    let s = if is_whisper {
        group_segments_by_gap(s, max_intra_segment_gap_seconds)
    } else {
        split_segments_by_word_gap(s, max_intra_segment_gap_seconds)
    };
    split_long_segments(s, max_segment_duration_seconds)
}

/// Convenience wrapper: run `post_process_raw_asr_segments`, then convert
/// to `TranscriptSegment`s. Mirrors `post_process_asr_segments` in Python.
pub fn post_process_asr_segments(
    segments: Vec<AsrSegment>,
    chunk_size: f64,
    overlap: f64,
    gap_threshold: f64,
    max_duration: f64,
    is_whisper: bool,
) -> Vec<caption_schema::TranscriptSegment> {
    let asr = post_process_raw_asr_segments(
        segments,
        chunk_size,
        overlap,
        gap_threshold,
        max_duration,
        is_whisper,
    );
    crate::convert::asr_segments_to_transcript_segments(asr, None)
}
