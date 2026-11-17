//! Bitcoin P2PKH private-key search over a hex range.
//!
//! CPU path (always available) + optional CUDA path behind the `cuda` feature.

pub mod checkpoint;
pub mod config;
pub mod cpu;
pub mod target;

#[cfg(feature = "cuda")]
pub mod gpu;
