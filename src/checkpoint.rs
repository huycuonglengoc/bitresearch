//! Checkpoint persistence (resume state).

use std::path::Path;

use anyhow::Result;

/// Format: a single line with the 16-char hex of the next key to scan.
pub fn save(path: &Path, next_key: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, format!("{next_key:016x}\n"))?;
    Ok(())
}

/// Load a checkpoint. Returns `None` if the file does not exist.
pub fn load(path: &Path) -> Result<Option<u64>> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(None);
            }
            let v = u64::from_str_radix(t, 16)
                .map_err(|e| anyhow::anyhow!("invalid checkpoint '{t}': {e}"))?;
            Ok(Some(v))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}
