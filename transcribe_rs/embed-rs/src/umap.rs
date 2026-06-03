//! UMAP dimensionality reduction for speaker embeddings.
//!
//! Mirrors the Python `umap_dimensions=[1,2]` default in `embed_cli.py`:
//! produce a 1-D and a 2-D coordinate per segment, stored in
//! `SegmentSpeakerEmbedding.umap_embeddings`.
//!
//! Approach:
//!   - Brute-force k-NN with cosine distance (Python uses `metric="cosine"`).
//!     N is small (segments per doc, typically < 200), so O(N^2) is fine.
//!   - Random init (Python falls back to "random" when N is small relative to
//!     n_components anyway).
//!   - Pass to `umap-rs` which handles the gradient-descent optimization.
//!
//! Numerical parity vs Python is NOT a goal: umap-learn and umap-rs have
//! different random seeds, KNN backends (NN-descent vs brute force), and
//! initialization. We just produce coordinates of the right shape.

use eyre::{eyre, Result};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use umap_rs::{GraphParams, Umap, UmapConfig};

const DEFAULT_N_NEIGHBORS: usize = 15;
const RNG_SEED: u64 = 0xCAFEC0DE;

/// Reduce `embeddings` (rows = samples, columns = features) to `dim`
/// dimensions via UMAP. Returns `Vec<Vec<f32>>` of length N where each
/// inner vec has length `dim`.
///
/// Returns `Ok(None)` when N is too small to meaningfully UMAP (matches
/// Python's `len(segment_ids) > 1` guard and silent skip on UMAP failure).
pub fn compute_umap(embeddings: &[Vec<f32>], dim: usize) -> Result<Option<Vec<Vec<f32>>>> {
    let n = embeddings.len();
    if n < 2 || dim < 1 {
        return Ok(None);
    }
    let feat_dim = embeddings[0].len();
    if feat_dim == 0 {
        return Ok(None);
    }
    if !embeddings.iter().all(|e| e.len() == feat_dim) {
        return Err(eyre!("embeddings have inconsistent feature dimensions"));
    }

    // Flatten to ndarray Array2<f32>(N, D).
    let flat: Vec<f32> = embeddings.iter().flatten().copied().collect();
    let data = Array2::from_shape_vec((n, feat_dim), flat)?;

    // L2-normalize a copy for cosine-distance KNN. Cosine distance ranking is
    // preserved by Euclidean distance on unit-norm vectors, so we can hand
    // umap-rs the simpler metric without changing neighbor ordering.
    let mut normed = data.clone();
    for mut row in normed.rows_mut() {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }

    // Python clamps `n_neighbors = min(15, N-1)`; same here.
    let n_neighbors = DEFAULT_N_NEIGHBORS.min(n.saturating_sub(1));
    if n_neighbors == 0 {
        return Ok(None);
    }

    let (knn_indices, knn_dists) = brute_force_knn(normed.view(), n_neighbors);

    // Random init in [-10, 10] (umap-learn's default range).
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let init: Array2<f32> = Array2::from_shape_fn((n, dim), |_| rng.random_range(-10.0..10.0));

    let config = UmapConfig {
        n_components: dim,
        graph: GraphParams {
            n_neighbors,
            ..Default::default()
        },
        ..Default::default()
    };
    let umap = Umap::new(config);
    let model = umap.fit(
        normed.view(),
        knn_indices.view(),
        knn_dists.view(),
        init.view(),
    );
    let embedding = model.into_embedding();

    let mut out = Vec::with_capacity(n);
    for row in embedding.rows() {
        out.push(row.iter().copied().collect::<Vec<f32>>());
    }
    Ok(Some(out))
}

/// Brute-force k-nearest-neighbor on L2-normalized vectors → equivalent to
/// cosine-distance KNN. Returns (indices, dists) of shape (N, k) — the row
/// for sample i is `i`'s nearest neighbors *excluding itself*.
fn brute_force_knn(data: ArrayView2<f32>, k: usize) -> (Array2<u32>, Array2<f32>) {
    let n = data.nrows();
    let mut indices = Array2::<u32>::zeros((n, k));
    let mut dists = Array2::<f32>::zeros((n, k));

    for i in 0..n {
        let me = data.row(i);
        // (distance, neighbor_index) for all j != i
        let mut pairs: Vec<(f32, u32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let other = data.row(j);
                let d = me
                    .iter()
                    .zip(other.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>()
                    .sqrt();
                (d, j as u32)
            })
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (slot, (d, idx)) in pairs.iter().take(k).enumerate() {
            indices[[i, slot]] = *idx;
            dists[[i, slot]] = *d;
        }
    }

    (indices, dists)
}
