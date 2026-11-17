//! CPU search: sequential scan of [start, end] using incremental point addition.
//!
//! Work is distributed with a shared atomic "frontier" (work-stealing): each
//! worker claims a contiguous block of keys via `fetch_add`, scans it with a
//! single scalar-mult base point + repeated `P += G` additions, then claims the
//! next block. This keeps the frontier a trivial, correct checkpoint (keys below
//! the frontier are fully claimed).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::PrimeField;
use k256::{FieldBytes, ProjectivePoint, Scalar};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::target::hash160;

/// Number of keys each worker claims per block (2^20 ≈ 1M keys).
pub const BLOCK: u64 = 1 << 20;
/// Convert a u64 (< 2^64, always < curve order n) to a `Scalar`.
fn scalar_from_u64(v: u64) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&v.to_be_bytes());
    Scalar::from_repr(FieldBytes::from(bytes)).unwrap()
}

/// Scan one contiguous block [block_start, block_end] (inclusive).
fn scan_block(block_start: u64, block_end: u64, target: &[u8; 20]) -> Option<(u64, String)> {
    let mut p = ProjectivePoint::GENERATOR * scalar_from_u64(block_start);
    let g_proj = ProjectivePoint::GENERATOR;
    let mut key = block_start;

    loop {
        let aff = p.to_affine();
        let enc = aff.to_encoded_point(true); // 33-byte compressed pubkey
        let h = hash160(enc.as_bytes());
        if &h == target {
            return Some((key, crate::target::hash160_to_address(&h)));
        }
        if key == block_end {
            return None;
        }
        p += g_proj; // P += G
        key += 1;
    }
}

/// Multi-threaded CPU search. Returns the found `(key, address)`.
///
/// `frontier` is advanced atomically; on return it points past the last claimed
/// key. The caller can persist it as a checkpoint. `stop` is set when a key is
/// found so other workers can bail out early.
pub fn search(
    start: u64,
    end: u64,
    target: &[u8; 20],
    threads: usize,
    frontier: &AtomicU64,
    stop: &AtomicBool,
) -> Option<(u64, String)> {
    frontier.store(start, Ordering::Relaxed);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build thread pool");

    pool.install(|| {
        rayon::iter::repeat_n((), threads)
            .into_par_iter()
            .find_map_any(|_| loop {
                if stop.load(Ordering::Relaxed) {
                    return None;
                }
                let s = frontier.fetch_add(BLOCK, Ordering::Relaxed);
                if s > end {
                    return None;
                }
                let e = (s + BLOCK - 1).min(end);
                if let Some(found) = scan_block(s, e, target) {
                    stop.store(true, Ordering::Relaxed);
                    return Some(found);
                }
            })
    })
}
