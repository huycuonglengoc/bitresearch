//! CUDA search path (behind the `cuda` feature).
//!
//! Uses `cudarc` with runtime NVRTC compilation: the `.cu` kernel source is
//! compiled to PTX at startup, so no `nvcc` build step is required (only the
//! CUDA driver + libnvrtc, both present on Google Colab).
//!
//! The host precomputes the `2^i * G` table and the per-launch base/step points
//! using `k256`; the kernel does only incremental point additions + hashing.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use anyhow::Result;
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

use k256::elliptic_curve::sec1::ToEncodedPoint;
use k256::elliptic_curve::PrimeField;
use k256::{AffinePoint, FieldBytes, ProjectivePoint, Scalar};

use crate::target::{hash160, hash160_to_address};

const BLOCK_DIM: u32 = 256;
/// Max threads per kernel launch (covers `GPU_CHUNK` with stride).
const MAX_THREADS: u64 = 1 << 20;
/// Keys scanned per kernel launch (~1s on a fast GPU). Each launch is a
/// natural checkpoint boundary.
const GPU_CHUNK: u64 = 1 << 30;

fn scalar_from_u64(v: u64) -> Scalar {
    let mut bytes = [0u8; 32];
    bytes[24..].copy_from_slice(&v.to_be_bytes());
    Scalar::from_repr(FieldBytes::from(bytes)).unwrap()
}

fn point_from_u64(v: u64) -> AffinePoint {
    AffinePoint::from(ProjectivePoint::GENERATOR * scalar_from_u64(v))
}

/// Big-endian 32 bytes -> 8 little-endian u32 limbs (matching the kernel layout).
fn limbs_from_be(be: &[u8; 32]) -> [u32; 8] {
    let mut out = [0u32; 8];
    for i in 0..8 {
        let s = 28 - 4 * i;
        out[i] = u32::from_be_bytes([be[s], be[s + 1], be[s + 2], be[s + 3]]);
    }
    out
}

/// Affine point -> [x limbs (8), y limbs (8)].
fn point_limbs(p: &AffinePoint) -> [u32; 16] {
    // Uncompressed encoding: 0x04 || x(32) || y(32).
    let enc = p.to_encoded_point(false);
    let b = enc.as_bytes();
    let xl = limbs_from_be(b[1..33].try_into().unwrap());
    let yl = limbs_from_be(b[33..65].try_into().unwrap());
    let mut out = [0u32; 16];
    out[0..8].copy_from_slice(&xl);
    out[8..16].copy_from_slice(&yl);
    out
}

fn hash_to_u32s(h: &[u8; 20]) -> [u32; 5] {
    let mut out = [0u32; 5];
    for i in 0..5 {
        out[i] = u32::from_be_bytes([h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]]);
    }
    out
}

fn build_g_tab() -> Vec<u32> {
    let mut tab = Vec::with_capacity(64 * 16);
    let mut cur = AffinePoint::GENERATOR; // 1 * G
    for _ in 0..64 {
        tab.extend_from_slice(&point_limbs(&cur));
        cur = AffinePoint::from(ProjectivePoint::from(cur).double());
    }
    tab
}

fn address_from_key(key: u64) -> String {
    let p = ProjectivePoint::GENERATOR * scalar_from_u64(key);
    let enc = p.to_affine().to_encoded_point(true);
    let h = hash160(enc.as_bytes());
    hash160_to_address(&h)
}

struct Gpu {
    _ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    _module: std::sync::Arc<CudaModule>,
    search_fn: cudarc::driver::CudaFunction,
    g_tab: CudaSlice<u32>,
    target: CudaSlice<u32>,
}

fn setup(target: &[u8; 20]) -> Result<Gpu> {
    eprintln!("[gpu] initializing CUDA device 0");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    eprintln!("[gpu] compiling kernel (NVRTC)");
    let ptx = compile_ptx(include_str!("../kernel/kernel.cu"))
        .map_err(|e| anyhow::anyhow!("NVRTC compile failed: {e}"))?;
    eprintln!("[gpu] loading module");
    let module = ctx.load_module(ptx)?;
    let search_fn = module.load_function("search_kernel")?;
    let g_tab = build_g_tab();
    let g_tab_dev = stream.memcpy_stod(&g_tab)?;
    let target_dev = stream.memcpy_stod(&hash_to_u32s(target))?;
    eprintln!("[gpu] ready");
    Ok(Gpu {
        _ctx: ctx,
        stream,
        _module: module,
        search_fn,
        g_tab: g_tab_dev,
        target: target_dev,
    })
}

impl Gpu {
    /// Launch one kernel over [start, end]; return the found key if any.
    fn launch_chunk(&self, start: u64, end: u64) -> Result<Option<u64>> {
        let total = end - start + 1;
        let t = total.min(MAX_THREADS);
        let grid = (t as u32).div_ceil(BLOCK_DIM);

        let p_start = point_limbs(&point_from_u64(start));
        let step = point_limbs(&point_from_u64(t));
        let p_start_dev = self.stream.memcpy_stod(&p_start)?;
        let step_dev = self.stream.memcpy_stod(&step)?;

        let mut found_key_dev = self.stream.alloc_zeros::<u64>(1)?;
        let mut found_flag_dev = self.stream.alloc_zeros::<u32>(1)?;

        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: (2 * BLOCK_DIM as usize * 8 * 4) as u32,
        };

        let mut builder = self.stream.launch_builder(&self.search_fn);
        builder.arg(&self.g_tab);
        builder.arg(&p_start_dev);
        builder.arg(&step_dev);
        builder.arg(&self.target);
        builder.arg(&start);
        builder.arg(&end);
        builder.arg(&mut found_key_dev);
        builder.arg(&mut found_flag_dev);
        eprintln!("[gpu] launching chunk [{start:#x}, {end:#x}] (T={t}, grid={grid})");
        unsafe { builder.launch(cfg) }?;
        self.stream.synchronize()?;
        eprintln!("[gpu] chunk done");

        let flag = self.stream.memcpy_dtov(&found_flag_dev)?[0];
        if flag == 1 {
            let key = self.stream.memcpy_dtov(&found_key_dev)?[0];
            Ok(Some(key))
        } else {
            Ok(None)
        }
    }
}

/// Scan [start, end] on the GPU. Claims `GPU_CHUNK`-sized blocks from `frontier`
/// (so the caller's checkpoint logic works identically to the CPU path).
pub fn search(
    start: u64,
    end: u64,
    target: &[u8; 20],
    frontier: &AtomicU64,
    stop: &AtomicBool,
) -> Result<Option<(u64, String)>> {
    frontier.store(start, Ordering::Relaxed);
    let gpu = setup(target)?;
    let mut cur = frontier.load(Ordering::Relaxed);
    while cur <= end {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let chunk_end = (cur + GPU_CHUNK - 1).min(end);
        if let Some(key) = gpu.launch_chunk(cur, chunk_end)? {
            stop.store(true, Ordering::Relaxed);
            return Ok(Some((key, address_from_key(key))));
        }
        frontier.store(chunk_end + 1, Ordering::Relaxed);
        cur = chunk_end + 1;
    }
    Ok(None)
}

/// Validate the GPU kernel against the host `k256` reference:
/// 1. HASH160 of a set of keys (field arithmetic, EC, hashes, encoding).
/// 2. A full mini-search over a small range (exercises the batch inversion).
pub fn selftest() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let ptx = compile_ptx(include_str!("../kernel/kernel.cu"))
        .map_err(|e| anyhow::anyhow!("NVRTC compile failed: {e}"))?;
    let module = ctx.load_module(ptx)?;

    // --- 1. HASH160 reference check ---
    let fn_ = module.load_function("selftest_kernel")?;
    let g_tab = build_g_tab();
    let g_tab_dev = stream.memcpy_stod(&g_tab)?;

    let keys: Vec<u64> = vec![
        1,
        2,
        3,
        5,
        7,
        10,
        100,
        1000,
        0xdead_beef,
        0xffff_ffff,
        1 << 40,
        0x1234_5678_9abc_def0,
    ];
    let keys_dev = stream.memcpy_stod(&keys)?;
    let n = keys.len();
    let mut out_hash_dev = stream.alloc_zeros::<u32>(n * 5)?;

    let cfg = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(BLOCK_DIM), 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i32 = n as i32;
    let mut b = stream.launch_builder(&fn_);
    b.arg(&g_tab_dev);
    b.arg(&keys_dev);
    b.arg(&mut out_hash_dev);
    b.arg(&n_i32);
    unsafe { b.launch(cfg) }?;
    stream.synchronize()?;
    let hash_out = stream.memcpy_dtov(&out_hash_dev)?;

    let mut ok = true;
    for (i, &key) in keys.iter().enumerate() {
        let p = ProjectivePoint::GENERATOR * scalar_from_u64(key);
        let enc = p.to_affine().to_encoded_point(true);
        let host_hash = hash_to_u32s(&hash160(enc.as_bytes()));
        let gpu_hash = &hash_out[i * 5..i * 5 + 5];
        if gpu_hash != host_hash.as_slice() {
            ok = false;
            eprintln!("HASH160 MISMATCH key={key} gpu={gpu_hash:08x?} host={host_hash:08x?}");
        }
    }

    // --- 2. Full search-kernel check (batch inversion) ---
    let probe = 0x0000_3039u64; // 12345
    let probe_hash: [u8; 20] = {
        let p = ProjectivePoint::GENERATOR * scalar_from_u64(probe);
        let enc = p.to_affine().to_encoded_point(true);
        hash160(enc.as_bytes())
    };
    let gpu = setup(&probe_hash)?;
    match gpu.launch_chunk(probe - 16, probe + 16)? {
        Some(k) if k == probe => {}
        other => {
            ok = false;
            eprintln!("SEARCH MISMATCH probe={probe} got={other:?}");
        }
    }

    if ok {
        println!("GPU self-test PASSED ({n} hash checks + 1 search check)");
        Ok(())
    } else {
        anyhow::bail!("GPU self-test FAILED");
    }
}
