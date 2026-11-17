//! Argument parsing and validation.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "bitsearch",
    version,
    about = "Search a hex key range for the private key matching a Bitcoin P2PKH address"
)]
pub struct Args {
    /// Start of the private-key range (hex, up to 64-bit, optional 0x prefix).
    #[arg(long)]
    pub start: Option<String>,

    /// End of the range (hex, inclusive).
    #[arg(long)]
    pub end: Option<String>,

    /// Target P2PKH address (starts with '1').
    #[arg(long)]
    pub target: Option<String>,

    /// Number of CPU threads to use (0 = all logical cores).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,

    /// Force the CPU search path (default uses the GPU when the `cuda` feature
    /// is enabled; falls back to CPU automatically otherwise).
    #[arg(long)]
    pub cpu: bool,

    /// Path to the checkpoint file (resume state).
    #[arg(long, default_value = "checkpoint.json")]
    pub checkpoint: PathBuf,

    /// Run the GPU self-test (validates kernel vs CPU reference) and exit.
    #[arg(long)]
    pub selftest: bool,
}

pub struct Config {
    pub start: u64,
    pub end: u64,
    pub target: String,
    pub threads: usize,
    pub cpu: bool,
    pub checkpoint: PathBuf,
    pub selftest: bool,
}

impl Config {
    pub fn from_args(a: Args) -> Result<Self> {
        let start_str = a.start.ok_or_else(|| anyhow::anyhow!("--start is required"))?;
        let end_str = a.end.ok_or_else(|| anyhow::anyhow!("--end is required"))?;
        let target = a.target.ok_or_else(|| anyhow::anyhow!("--target is required"))?;
        let start = parse_hex_u64(&start_str)?;
        let end = parse_hex_u64(&end_str)?;
        if start == 0 {
            anyhow::bail!("start must be >= 1 (private key 0 is the point at infinity, not a valid key)");
        }
        if start > end {
            anyhow::bail!("invalid range: start (0x{start:x}) > end (0x{end:x})");
        }
        Ok(Config {
            start,
            end,
            target,
            threads: a.threads,
            cpu: a.cpu,
            checkpoint: a.checkpoint,
            selftest: a.selftest,
        })
    }

    pub fn thread_count(&self) -> usize {
        if self.threads > 0 {
            self.threads
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }
    }
}

pub fn parse_hex_u64(s: &str) -> Result<u64> {
    let t = s.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    if t.is_empty() {
        anyhow::bail!("empty hex value");
    }
    u64::from_str_radix(t, 16).map_err(|e| anyhow::anyhow!("invalid hex u64 '{t}': {e}"))
}
