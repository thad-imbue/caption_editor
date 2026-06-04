//! HF download wrapper that emits visible-in-electron-log progress.
//!
//! `hf_hub`'s built-in `ProgressBar` (indicatif) auto-disables when stderr
//! isn't a TTY — which is exactly the Electron-spawned-child-process case
//! we ship in. That turned the user-facing log into a long unexplained
//! stall at `Loading parakeet model: ...` while ~600 MB of ONNX weights
//! downloaded silently in the background.
//!
//! `fetch_with_log` here mirrors `ApiRepo::get`'s cache-first behavior:
//!   - cache hit  → return path immediately, no log noise.
//!   - cache miss → eprintln "Downloading <file>" then call
//!                  `download_with_progress` with a `LogProgress` that
//!                  emits one throttled `eprintln!` per 5% (or per 2s)
//!                  with MB downloaded + MB/s. Stderr-only, plays well
//!                  with the Electron log tail.

use eyre::{eyre, Result};
use hf_hub::api::sync::ApiRepo;
use hf_hub::api::Progress;
use hf_hub::{Cache, Repo};
use std::path::PathBuf;
use std::time::Instant;

pub fn fetch_with_log(api: &ApiRepo, model_id: &str, filename: &str) -> Result<PathBuf> {
    // Same cache lookup hf-hub's `get()` does internally; avoids any HTTP
    // when the file is already on disk.
    let cache = Cache::from_env();
    let cache_repo = cache.repo(Repo::model(model_id.to_string()));
    if let Some(path) = cache_repo.get(filename) {
        return Ok(path);
    }
    eprintln!("Downloading {filename} from huggingface.co/{model_id}...");
    api.download_with_progress(filename, LogProgress::default())
        .map_err(|e| eyre!("hf-hub download {filename}: {e}"))
}

#[derive(Default)]
struct LogProgress {
    filename: String,
    total: usize,
    current: usize,
    last_logged_pct: i32,
    start: Option<Instant>,
    last_log: Option<Instant>,
}

impl Progress for LogProgress {
    fn init(&mut self, size: usize, filename: &str) {
        // hf-hub calls init() twice — once in download_tempfile and again
        // in download_from — so skip the second header line.
        let first_call = self.start.is_none();
        self.filename = filename.to_string();
        self.total = size;
        self.current = 0;
        self.last_logged_pct = -1;
        let now = Instant::now();
        self.start = Some(now);
        self.last_log = Some(now);
        if first_call {
            eprintln!(
                "  {} — {:.1} MB to download",
                filename,
                size as f64 / 1_048_576.0
            );
        }
    }

    fn update(&mut self, size: usize) {
        self.current += size;
        let pct = if self.total > 0 {
            (self.current as u64 * 100 / self.total as u64) as i32
        } else {
            0
        };
        // Throttle: log on every 5% milestone, but never more than once
        // every 2 seconds (avoids spam on tiny files / slow links).
        let now = Instant::now();
        let since_last = self
            .last_log
            .map(|t| now.duration_since(t).as_secs_f64())
            .unwrap_or(0.0);
        let milestone = pct >= self.last_logged_pct + 5;
        if milestone && since_last >= 1.0 {
            let elapsed = self
                .start
                .map(|t| now.duration_since(t).as_secs_f64())
                .unwrap_or(0.001)
                .max(0.001);
            let mb_done = self.current as f64 / 1_048_576.0;
            let mb_total = self.total as f64 / 1_048_576.0;
            let mb_per_s = mb_done / elapsed;
            eprintln!(
                "  {}: {}% ({:.1} / {:.1} MB, {:.1} MB/s)",
                self.filename, pct, mb_done, mb_total, mb_per_s
            );
            self.last_logged_pct = pct;
            self.last_log = Some(now);
        }
    }

    fn finish(&mut self) {
        let elapsed = self
            .start
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        let mb = self.total as f64 / 1_048_576.0;
        eprintln!(
            "  {} done — {:.1} MB in {:.1}s",
            self.filename, mb, elapsed
        );
    }
}
