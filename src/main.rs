use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

use bitsearch::config::{self, Config};
use bitsearch::{checkpoint, cpu, target};

fn main() -> Result<()> {
    let args = config::Args::parse();

    // GPU self-test (no range/target needed).
    #[cfg(feature = "cuda")]
    if args.selftest {
        return bitsearch::gpu::selftest();
    }
    #[cfg(not(feature = "cuda"))]
    if args.selftest {
        anyhow::bail!("self-test requires building with `--features cuda`");
    }

    let cfg = Config::from_args(args)?;

    let target_hash = target::address_to_hash160(&cfg.target)?;
    eprintln!("target HASH160 = {}", hex::encode(target_hash));

    // Resume from checkpoint if present.
    let mut start = cfg.start;
    if let Some(cp) = checkpoint::load(&cfg.checkpoint)? {
        if cp > start {
            eprintln!("resuming from checkpoint 0x{cp:016x}");
            start = cp;
        }
    }
    if start > cfg.end {
        anyhow::bail!("range already exhausted (checkpoint past end)");
    }

    let threads = cfg.thread_count();
    let mode = if cfg.cpu || !cfg!(feature = "cuda") {
        format!("CPU ({threads} threads)")
    } else {
        "GPU (CUDA)".to_string()
    };
    eprintln!(
        "scanning range [0x{start:016x}, 0x{:016x}] ({} keys) using {mode}",
        cfg.end,
        cfg.end as u128 - start as u128 + 1
    );

    // Shared frontier + stop flag.
    let frontier = Arc::new(AtomicU64::new(start));
    let done = Arc::new(AtomicBool::new(false));

    // Periodic checkpoint writer (survives hard kills within ~5s). Woken early
    // via a channel so the process exits promptly.
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let cp_path = cfg.checkpoint.clone();
    let cp_frontier = frontier.clone();
    let cp_threads = threads;
    let cp_end = cfg.end;
    let cp_start = cfg.start;
    let writer = std::thread::spawn(move || loop {
        match stop_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
        }
        // Conservative: re-scan up to `threads * BLOCK` in-flight keys.
        let f = cp_frontier.load(Ordering::Relaxed);
        let safe = f.saturating_sub(cp_threads as u64 * cpu::BLOCK).max(cp_start);
        if safe <= cp_end {
            let _ = checkpoint::save(&cp_path, safe);
        }
    });

    // Progress reporter.
    let rep_frontier = frontier.clone();
    let rep_start = start;
    let rep_end = cfg.end;
    let (rep_tx, rep_rx) = std::sync::mpsc::channel::<()>();
    let reporter = std::thread::spawn(move || {
        let t0 = Instant::now();
        let mut last = 0u64;
        let total = rep_end.saturating_sub(rep_start) as f64 + 1.0;
        loop {
            match rep_rx.recv_timeout(Duration::from_secs(2)) {
                Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            }
            let f = rep_frontier.load(Ordering::Relaxed);
            let scanned = f.saturating_sub(rep_start);
            let rate = scanned.saturating_sub(last) as f64 / 2.0;
            last = scanned;
            let pct = (scanned as f64 / total) * 100.0;
            eprintln!(
                "scanned {scanned} keys | ~{:.1} Mkeys/s | {:>5.1}% | elapsed {:.0}s",
                rate / 1e6,
                pct,
                t0.elapsed().as_secs()
            );
        }
    });

    // Run the search. GPU is the default when compiled with the `cuda` feature;
    // `--cpu` (or a non-CUDA build) uses the CPU path.
    let found = if cfg.cpu || !cfg!(feature = "cuda") {
        cpu::search(start, cfg.end, &target_hash, threads, &frontier, &done)
    } else {
        #[cfg(feature = "cuda")]
        {
            bitsearch::gpu::search(start, cfg.end, &target_hash, &frontier, &done)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    };

    done.store(true, Ordering::Relaxed);
    let _ = stop_tx.send(());
    let _ = rep_tx.send(());
    let _ = writer.join();
    let _ = reporter.join();

    match found {
        Some((key, addr)) => {
            println!("FOUND");
            println!("private_key = 0x{key:064x}");
            println!("address     = {addr}");
            checkpoint::save(&cfg.checkpoint, key)?;
        }
        None => {
            println!("NOT_FOUND");
            checkpoint::save(&cfg.checkpoint, cfg.end + 1)?;
        }
    }

    Ok(())
}
