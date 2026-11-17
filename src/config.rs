//! Argument parsing and validation.
//!
//! Hand-rolled on purpose: `clap` (+ `clap_derive`/`syn`/`quote`/`anstream`) is
//! a large compile-time cost that makes the from-scratch build very slow on
//! Colab's 2 vCPU. Our CLI has only a handful of trivial flags, so a small
//! manual parser is clearer and much faster to build.

use std::path::PathBuf;

use anyhow::Result;

#[derive(Debug)]
pub struct Args {
    pub start: Option<String>,
    pub end: Option<String>,
    pub target: Option<String>,
    pub threads: usize,
    pub cpu: bool,
    pub checkpoint: PathBuf,
    pub selftest: bool,
}

fn print_help() {
    eprintln!(
        "\
bitsearch {version} — search a hex key range for the private key matching a P2PKH address

USAGE:
    bitsearch --start <HEX> --end <HEX> --target <ADDRESS> [OPTIONS]

OPTIONS:
    --start <HEX>       Start of the private-key range (hex, up to 64-bit, 0x optional)
    --end <HEX>         End of the range (hex, inclusive)
    --target <ADDRESS>  Target P2PKH address (starts with '1')
    --threads <N>       Number of CPU threads (0 = all logical cores) [default: 0]
    --cpu               Force the CPU search path (default: GPU when built with CUDA)
    --checkpoint <PATH> Checkpoint file for resume [default: checkpoint.json]
    --selftest          Run the GPU self-test and exit (no range/target needed)
    -h, --help          Print help
    -V, --version       Print version",
        version = env!("CARGO_PKG_VERSION")
    );
}

impl Args {
    /// Parse command-line arguments, printing help/errors and exiting on
    /// malformed input (mirrors the ergonomics of a `clap::Parser`).
    pub fn parse() -> Args {
        let mut args = Args {
            start: None,
            end: None,
            target: None,
            threads: 0,
            cpu: false,
            checkpoint: PathBuf::from("checkpoint.json"),
            selftest: false,
        };

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            // Support both `--flag value` and `--flag=value`.
            let (key, inline_val) = match arg.split_once('=') {
                Some((k, v)) => (k.to_string(), Some(v.to_string())),
                None => (arg, None),
            };

            match key.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--version" | "-V" => {
                    println!("bitsearch {}", env!("CARGO_PKG_VERSION"));
                    std::process::exit(0);
                }
                "--start" => args.start = Some(take_value(&mut it, inline_val, "--start")),
                "--end" => args.end = Some(take_value(&mut it, inline_val, "--end")),
                "--target" => args.target = Some(take_value(&mut it, inline_val, "--target")),
                "--threads" => {
                    let v = take_value(&mut it, inline_val, "--threads");
                    args.threads = v.parse().unwrap_or_else(|_| {
                        eprintln!("error: invalid value '{v}' for '--threads': expected an integer");
                        std::process::exit(2);
                    });
                }
                "--checkpoint" => {
                    args.checkpoint = PathBuf::from(take_value(&mut it, inline_val, "--checkpoint"))
                }
                "--cpu" => args.cpu = true,
                "--selftest" => args.selftest = true,
                other => {
                    eprintln!("error: unknown argument '{other}'");
                    print_help();
                    std::process::exit(2);
                }
            }
        }

        args
    }
}

/// Return the value for `flag`, either from `--flag=value` or the next
/// positional argument, or exit with an error if it is missing.
fn take_value(it: &mut impl Iterator<Item = String>, inline: Option<String>, flag: &str) -> String {
    if let Some(v) = inline {
        return v;
    }
    it.next().unwrap_or_else(|| {
        eprintln!("error: missing value for '{flag}'");
        print_help();
        std::process::exit(2);
    })
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
