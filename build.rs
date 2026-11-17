//! Build script: precompiles `kernel/kernel.cu` to PTX using `nvcc` when the
//! `cuda` feature is enabled.
//!
//! We deliberately avoid NVRTC (runtime compilation) because the NVRTC shipped
//! with CUDA 13 on Colab is missing `libnvrtc-builtins.so.13.0`, which makes
//! `nvrtcCompileProgram` hang. Precompiling with `nvcc` is fully offline and
//! has no such dependency.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn find_nvcc() -> Option<PathBuf> {
    // Explicit override.
    if let Ok(p) = std::env::var("NVCC") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    // CUDA install directories.
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(base) = std::env::var(var) {
            let p = PathBuf::from(&base).join("bin").join("nvcc");
            if p.exists() {
                return Some(p);
            }
        }
    }
    for dir in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
    ] {
        let p = PathBuf::from(dir);
        if p.exists() {
            return Some(p);
        }
    }
    // Search PATH (covers pip-installed `nvidia-cuda-nvcc-cu12` if its bin/
    // dir was added to PATH, and apt-installed nvcc).
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let p = PathBuf::from(dir).join("nvcc");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernel/kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BITSEARCH_GPU_ARCH");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");

    // Only build the kernel when the CUDA backend is enabled.
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Default to compute_89 (NVIDIA L4 / Ada). PTX for a virtual architecture
    // is JIT-compiled by the driver and is forward-compatible with newer GPUs.
    // Override with BITSEARCH_GPU_ARCH (e.g. "compute_80", "sm_90").
    let arch = std::env::var("BITSEARCH_GPU_ARCH").unwrap_or_else(|_| "compute_89".to_string());

    let nvcc = find_nvcc().unwrap_or_else(|| {
        panic!(
            "nvcc not found. Install it, e.g.:\n  \
             pip install nvidia-cuda-nvcc-cu12\n  \
             export PATH=\"$(python -c 'import site,os;print(os.path.join(site.getsitepackages()[0],\"nvidia\",\"cuda_nvcc\",\"bin\"))'):$PATH\"\n\
             or set NVCC=/path/to/nvcc."
        )
    });

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let ptx_path = PathBuf::from(&out_dir).join("kernel.ptx");

    eprintln!(
        "[build] nvcc = {} (arch {arch})",
        nvcc.display()
    );

    // Run nvcc with a timeout so a broken/hung nvcc fails loudly instead of
    // blocking `cargo build` forever.
    let mut child = Command::new(&nvcc)
        .arg("-ptx")
        .arg(format!("-arch={arch}"))
        .arg("-o")
        .arg(&ptx_path)
        .arg("kernel/kernel.cu")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("failed to run nvcc ({}): {e}", nvcc.display()));

    let deadline = Instant::now() + Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .unwrap_or_else(|e| panic!("failed to wait for nvcc: {e}"))
        {
            break status;
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            panic!(
                "nvcc timed out after 120s while compiling kernel/kernel.cu (arch {arch}).\n\
                 This usually means the nvcc install is broken (see pip nvidia-cuda-nvcc-cu12)."
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    };

    if !status.success() {
        panic!("nvcc failed to compile kernel/kernel.cu (arch {arch})");
    }

    println!("cargo:rerun-if-env-changed=PATH");
}
