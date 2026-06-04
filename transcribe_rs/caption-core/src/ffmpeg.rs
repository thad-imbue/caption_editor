//! Resolve the **bundled** `ffmpeg` binary for Rust CLIs (`transcribe-rs`, `embed-rs`).
//!
//! We never invoke `ffmpeg` from the user's PATH. The binary must be either:
//!   - `CAPTION_EDITOR_FFMPEG` (set by Electron to the packaged path), or
//!   - `ffmpeg` in the same directory as the running executable (`dist-rust/` in
//!     dev, `Contents/Resources/bin/` in the .app — staged by `npm run build:ffmpeg`).

use std::path::{Path, PathBuf};

/// Bundled ffmpeg was not found at the expected locations.
#[derive(Debug, Clone, thiserror::Error)]
#[error(
    "bundled ffmpeg not found — expected 'ffmpeg' next to this program \
     (run `npm run build:ffmpeg`) or set CAPTION_EDITOR_FFMPEG to the packaged binary"
)]
pub struct BundledFfmpegNotFound;

/// Resolve the ffmpeg executable path (bundled only).
pub fn resolve_ffmpeg() -> Result<PathBuf, BundledFfmpegNotFound> {
    let env = std::env::var("CAPTION_EDITOR_FFMPEG")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let exe_parent = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()));
    resolve_ffmpeg_from(exe_parent.as_deref(), env.as_deref())
}

/// Core resolution logic (testable without depending on the real `current_exe`).
pub fn resolve_ffmpeg_from(
    exe_parent: Option<&Path>,
    env_ffmpeg: Option<&str>,
) -> Result<PathBuf, BundledFfmpegNotFound> {
    if let Some(p) = env_ffmpeg {
        let path = PathBuf::from(p.trim());
        if path.is_file() {
            return Ok(path);
        }
    }

    if let Some(parent) = exe_parent {
        for name in ["ffmpeg", "ffmpeg.exe"] {
            let candidate = parent.join(name);
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }

    Err(BundledFfmpegNotFound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn env_override_must_exist_on_disk() {
        let dir = tempdir().unwrap();
        let fake = dir.path().join("ffmpeg");
        fs::write(&fake, b"").unwrap();
        let got = resolve_ffmpeg_from(None, Some(fake.to_str().unwrap())).unwrap();
        assert_eq!(got, fake);
    }

    #[test]
    fn sibling_next_to_exe_parent() {
        let dir = tempdir().unwrap();
        let fake = dir.path().join("ffmpeg");
        fs::write(&fake, b"").unwrap();
        let got = resolve_ffmpeg_from(Some(dir.path()), None).unwrap();
        assert_eq!(got, fake);
    }

    #[test]
    fn fails_when_nothing_bundled() {
        let dir = tempdir().unwrap();
        assert!(resolve_ffmpeg_from(Some(dir.path()), None).is_err());
        assert!(resolve_ffmpeg_from(None, None).is_err());
    }

    #[test]
    fn ignores_empty_env() {
        let dir = tempdir().unwrap();
        assert!(resolve_ffmpeg_from(Some(dir.path()), Some("   ")).is_err());
    }
}
