# bitsearch

A high-performance Bitcoin **P2PKH** private-key search tool written in Rust, with an
optional CUDA path. Given a range `[start, end]` and a target address, it scans every
private key in the range — deriving the compressed public key via incremental point
addition (no per-key scalar multiplication) and comparing `HASH160` — until it finds the
matching key.

> **Intended use:** the Bitcoin puzzle challenge, recovering your own lost keys, and
> security research. Do not use against addresses you do not own.

---

## ⚠️ The hard truth about feasibility

A sequential scan is **O(range)** — no optimization changes that. Average time is
`range / 2` divided by your keys-per-second rate:

| Range | Keys | @300M/s (T4 free) | @1G/s (L4/A100) |
|---|---|---|---|
| 2^50 | 1.1e15 | ~22 days | ~6.5 days |
| 2^54 | 1.8e16 | ~1 year | ~105 days |
| 2^60 | 1.2e18 | ~62 years | ~19 years |
| 2^64 | 1.8e19 | ~1000 years | ~300 years |

Sequential scanning is only practical up to roughly **2^50–2^52** on a single GPU. Beyond
that you need a large GPU farm or Pollard's Kangaroo (which additionally requires the
*target public key*, not just the address).

---

## Quick start (CPU, anywhere Rust runs)

```bash
cargo run --release -- \
  --start 1 --end 1000000 \
  --target 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
```

The classic test vector (private key `1`) is found instantly.

## Quick start (CUDA, on Linux + NVIDIA GPU)

```bash
cargo build --release --features cuda
./target/release/bitsearch --start <hex> --end <hex> --target <P2PKH>
```

Validate the GPU kernel before trusting it:

```bash
./target/release/bitsearch --selftest
```

This computes the `HASH160` of a set of test keys **on the GPU** and compares byte-for-byte
against the host `k256` reference, verifying field arithmetic, EC point operations,
compressed-key encoding, SHA-256 and RIPEMD-160 end-to-end.

## One command on Google Colab

```python
!bash <(curl -sSL https://raw.githubusercontent.com/<user>/bitsearch/main/scripts/run_colab.sh) \
     -- 0xSTART_HEX 0xEND_HEX 1TARGET_P2PKH...
```

The script installs the Rust toolchain if needed, clones the repo, builds with CUDA
(Colab has `nvcc` + the CUDA runtime), and runs. Set the `BITSEARCH_REPO` env var to point
at your fork. The checkpoint is written to `/content/bitsearch_checkpoint.json` so a
runtime reset resumes instead of restarting.

---

## CLI

```
--start <HEX>      Start of range (hex, optional 0x prefix)
--end <HEX>        End of range (hex, inclusive)
--target <ADDR>    Target P2PKH address (starts with '1')
--threads <N>      CPU threads (default: all logical cores)
--cpu              Force the CPU search path (default is GPU when built with CUDA)
--checkpoint <P>   Checkpoint file path (default: checkpoint.json)
--selftest         Run GPU self-test and exit
```

Output:

```
FOUND
private_key = 0x0000...0001
address     = 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
```

---

## How it works

### CPU path (`src/cpu.rs`)
- **Incremental point addition** — the base point `start·G` is computed once; each key is
  one cheap `P += G` (Jacobian) instead of a ~512-op scalar multiplication.
- **Work-stealing** — a shared atomic frontier; each worker claims a ~1M-key block, scans
  it, then claims the next. This makes checkpointing trivial (keys below the frontier are
  done).
- Multi-threaded via `rayon`, with `k256` for the curve and `sha2`/`ripemd` for hashing.

### CUDA path (`kernel/kernel.cu`, `src/gpu.rs`)
- 256-bit field arithmetic on 8×`u32` little-endian limbs (no `__int128` dependency);
  reduction uses the identity `2^256 ≡ 2^32 + 977 (mod p)`.
- Jacobian + affine **mixed addition**; the host precomputes `2^i·G` and the per-launch
  base/step points, so the kernel never does a scalar multiplication in its hot loop.
- **Shared-memory Montgomery batch inversion** — one field inverse per *block* instead of
  per *thread* — the single biggest throughput win.
- SHA-256 + RIPEMD-160 written for a single 64-byte block, running entirely in registers.
- `cudarc` loads the `.cu` source and JIT-compiles it via NVRTC at startup (no `nvcc`
  build step needed; only the driver + `libnvrtc`, both present on Colab).

### Checkpointing
A background thread persists the frontier every 5s, so a Colab runtime reset loses at most
~5s of work. Re-run with the same `--checkpoint` to resume.

---

## Testing

```bash
cargo test --release        # unit + end-to-end (CPU)
# with a GPU:
cargo build --release --features cuda
./target/release/bitsearch --selftest
```

---

## Project layout

```
src/
  main.rs          CLI, orchestration, checkpoint writer, progress reporter
  cpu.rs           CPU search (work-stealing + incremental point addition)
  gpu.rs           CUDA search + self-test (cuda feature)
  target.rs        Base58Check + P2PKH <-> HASH160
  config.rs        argument parsing
  checkpoint.rs    resume state
kernel/kernel.cu   CUDA kernel (field arithmetic, EC, SHA-256, RIPEMD-160)
scripts/run_colab.sh
tests/e2e.rs
```

## Limitations

- **P2PKH only** (addresses starting `1`). Matching is done on `HASH160`, so extending to
  P2SH (`3...`) or P2WPKH (`bc1q...`) only requires changing the target decoder.
- Key range is limited to `u64` (2^64) — beyond that, sequential scan is infeasible anyway.
- The CUDA path needs building/testing on an NVIDIA Linux machine (Colab works); it is not
  compiled on non-Linux targets.
>>>>>>> 0f1bcb7 (Initial commit)
