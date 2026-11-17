//! End-to-end test: derive the address for a known key, then confirm the search
//! finds it. Uses a key that crosses work-stealing block boundaries.

use std::sync::atomic::{AtomicBool, AtomicU64};

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::PrimeField;
use k256::{FieldBytes, ProjectivePoint, Scalar};

fn scalar_from_u64(v: u64) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&v.to_be_bytes());
    Scalar::from_repr(FieldBytes::from(bytes)).unwrap()
}

fn address_for_key(key: u64) -> String {
    let p = ProjectivePoint::GENERATOR * scalar_from_u64(key);
    let enc = p.to_affine().to_encoded_point(true);
    let h = bitsearch::target::hash160(enc.as_bytes());
    bitsearch::target::hash160_to_address(&h)
}

#[test]
fn finds_key_1() {
    let addr = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH";
    let hash = bitsearch::target::address_to_hash160(addr).unwrap();
    let frontier = AtomicU64::new(1);
    let stop = AtomicBool::new(false);
    let found = bitsearch::cpu::search(1, 1000, &hash, 4, &frontier, &stop);
    assert_eq!(found.map(|(k, _)| k), Some(1));
}

#[test]
fn finds_key_crossing_blocks() {
    // Key > 2^20 (BLOCK) forces the worker to scan multiple blocks.
    let key = 0x2F_0000u64; // ~3.08M
    let addr = address_for_key(key);
    let hash = bitsearch::target::address_to_hash160(&addr).unwrap();

    let start = key - 0x10_0000; // 1M before
    let end = key + 0x10_0000; // 1M after
    let frontier = AtomicU64::new(start);
    let stop = AtomicBool::new(false);
    let found = bitsearch::cpu::search(start, end, &hash, 8, &frontier, &stop);
    assert_eq!(found.map(|(k, _)| k), Some(key));
}

#[test]
fn not_found_returns_none() {
    let addr = "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH"; // key 1
    let hash = bitsearch::target::address_to_hash160(addr).unwrap();
    let frontier = AtomicU64::new(1000);
    let stop = AtomicBool::new(false);
    let found = bitsearch::cpu::search(1000, 2000, &hash, 4, &frontier, &stop);
    assert!(found.is_none());
}
