//! Serde mirror of `transcribe/schema.py` and `src/types/schema.ts`.
//!
//! Wire format: camelCase JSON (matches Pydantic's `alias` + the TS interface
//! field names). All field renames are explicit so the file reads like a
//! diff against `schema.py`.
//!
//! Cross-reference these files together when changing any field:
//!   - `transcribe/schema.py` (Pydantic, snake_case + alias to camelCase)
//!   - `src/types/schema.ts` (TS interfaces, camelCase)
//!   - this file (serde, snake_case Rust idents + rename to camelCase)
//!
//! Optionality discipline: `Optional[X]` in Python = `Option<X>` in Rust with
//! `#[serde(default, skip_serializing_if = "Option::is_none")]` so we don't
//! emit `"field": null` (Pydantic `model_dump(exclude_none=True)` and the TS
//! writer both drop them).

use serde::{Deserialize, Serialize};

fn skip_if_none<T>(v: &Option<T>) -> bool {
    v.is_none()
}

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HistoryAction {
    #[serde(rename = "modified")]
    Modified,
    #[serde(rename = "deleted")]
    Deleted,
    #[serde(rename = "speakerRenamed")]
    SpeakerRenamed,
}

// ---------------------------------------------------------------------------
// Document-level types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptWord {
    pub text: String,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub start_time: Option<f64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub end_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptSegment {
    pub id: String,
    #[serde(default)]
    pub index: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub text: String,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub words: Option<Vec<TranscriptWord>>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub speaker_name: Option<String>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub rating: Option<i32>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub timestamp: Option<String>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub verified: Option<bool>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub asr_model: Option<String>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptMetadata {
    pub id: String,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub media_file_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentHistoryEntry {
    pub id: String,
    pub action: HistoryAction,
    pub action_timestamp: String,
    pub segment: TranscriptSegment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SegmentSpeakerEmbedding {
    pub segment_id: String,
    /// Base64-encoded little-endian float32 vector. Use `encode_embedding` /
    /// `decode_embedding` to convert to `Vec<f32>`.
    pub speaker_embedding: String,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub umap_embeddings: Option<std::collections::BTreeMap<String, Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GridColumnState {
    pub col_id: String,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub width: Option<i64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub hide: Option<bool>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub sort: Option<String>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub sort_index: Option<i64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub flex: Option<f64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub pinned: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UIState {
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub column_state: Option<Vec<GridColumnState>>,
    /// Free-form AG Grid filter model — kept as `serde_json::Value` so this
    /// crate doesn't grow a flag for every filter shape AG Grid invents.
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub filter_model: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub left_panel_width: Option<f64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub caption_height: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawAsrWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawAsrSegmentSnapshot {
    pub text: String,
    pub start: f64,
    pub end: f64,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub chunk_start: Option<f64>,
    pub words: Vec<RawAsrWord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawAsrOutput {
    /// Defaults to 1 (matches Pydantic default) so older readers stay valid.
    #[serde(default = "raw_asr_version_default")]
    pub version: i32,
    pub segments: Vec<RawAsrSegmentSnapshot>,
}

fn raw_asr_version_default() -> i32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionsDocument {
    pub metadata: TranscriptMetadata,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub title: Option<String>,
    pub segments: Vec<TranscriptSegment>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub history: Option<Vec<SegmentHistoryEntry>>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub embeddings: Option<Vec<SegmentSpeakerEmbedding>>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub ui_state: Option<UIState>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub raw_asr_output: Option<RawAsrOutput>,
}

// ---------------------------------------------------------------------------
// ASR pipeline-internal types (mirror of asr_results_to_captions.ASRSegment).
// Distinct from `TranscriptSegment` above: this is the chunked-ASR-output
// shape, snake_case on the wire (Python dataclass), pre-post-processing.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WordTimestamp {
    pub word: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsrSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Vec<WordTimestamp>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub chunk_start: Option<f64>,
    #[serde(default, skip_serializing_if = "skip_if_none")]
    pub speaker: Option<String>,
}

// ---------------------------------------------------------------------------
// Embedding codec (matches transcribe/schema.py encode/decode_embedding)
// ---------------------------------------------------------------------------

pub fn encode_embedding(values: &[f32]) -> String {
    use base64::Engine as _;
    let mut raw = Vec::with_capacity(values.len() * 4);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(raw)
}

pub fn decode_embedding(b64: &str) -> Result<Vec<f32>, base64::DecodeError> {
    use base64::Engine as _;
    let raw = base64::engine::general_purpose::STANDARD.decode(b64)?;
    let mut out = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_roundtrip() {
        let values = vec![1.0f32, -2.5, 3.14, 0.0, f32::MIN_POSITIVE];
        let b64 = encode_embedding(&values);
        let back = decode_embedding(&b64).unwrap();
        assert_eq!(values, back);
    }

    #[test]
    fn segment_roundtrips_camelcase_json() {
        let json = r#"{
            "id": "abc",
            "index": 0,
            "startTime": 1.5,
            "endTime": 2.0,
            "text": "hi",
            "speakerName": "alice"
        }"#;
        let seg: TranscriptSegment = serde_json::from_str(json).unwrap();
        assert_eq!(seg.id, "abc");
        assert_eq!(seg.start_time, 1.5);
        assert_eq!(seg.speaker_name.as_deref(), Some("alice"));

        let re = serde_json::to_string(&seg).unwrap();
        // No nulls leak out for the unset Optionals.
        assert!(!re.contains("null"));
        assert!(re.contains("\"startTime\":1.5"));
    }
}
