#!/usr/bin/env bash
# One-command runner for bitsearch on Google Colab (and any Linux box with an
# NVIDIA GPU). Installs the Rust toolchain if needed, clones the repo, builds
# with CUDA, and runs.
#
# Usage (from a Colab cell):
#   # validate the GPU kernel:
#   !bash <(curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/run_colab.sh) \
#        -- --selftest
#   # real search:
#   !bash <(curl -sSL https://raw.githubusercontent.com/<user>/<repo>/main/scripts/run_colab.sh) \
#        -- --start 0xSTART_HEX --end 0xEND_HEX --target 1TARGET_P2PKH...
#
# All arguments after `--` are passed straight to the binary.
set -euo pipefail

REPO_URL="${BITSEARCH_REPO:-}"
REPO_DIR="bitsearch"

if [ -z "$REPO_URL" ]; then
  echo "ERROR: BITSEARCH_REPO is not set." >&2
  echo "Set it to your repository URL before running, e.g.:" >&2
  echo "  import os; os.environ['BITSEARCH_REPO'] = 'https://github.com/<user>/bitresearch.git'" >&2
  echo "or edit REPO_URL in scripts/run_colab.sh and push." >&2
  exit 1
fi

# Fail fast on a missing/private repo instead of prompting for credentials.
export GIT_TERMINAL_PROMPT=0

# --- 1. GPU check -----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "== GPU detected =="
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  USE_GPU=1
else
  echo "== No NVIDIA GPU detected; running CPU-only ==" >&2
  USE_GPU=0
fi

# --- 2. Rust toolchain ------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
  echo "== Installing Rust toolchain =="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

# --- 3. Clone repo ----------------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
  echo "== Cloning $REPO_URL =="
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
if [ ! -f "$REPO_DIR/Cargo.toml" ]; then
  echo "ERROR: clone failed or $REPO_DIR has no Cargo.toml." >&2
  echo "Check that BITSEARCH_REPO points at a public repo containing this project." >&2
  exit 1
fi
cd "$REPO_DIR"

# --- 4. Ensure nvcc (needed by build.rs to precompile the kernel to PTX) -----
if [ "$USE_GPU" = "1" ] && ! command -v nvcc >/dev/null 2>&1; then
  echo "== nvcc not found; installing nvidia-cuda-nvcc-cu12 =="
  pip install -q nvidia-cuda-nvcc-cu12
  NVCC_BIN="$(python - <<'PY'
import site, os
for base in site.getsitepackages():
    p = os.path.join(base, "nvidia", "cuda_nvcc", "bin")
    if os.path.isdir(p):
        print(p)
        break
PY
)"
  if [ -n "${NVCC_BIN:-}" ]; then
    export PATH="$NVCC_BIN:$PATH"
  fi
fi

# --- 5. Build ---------------------------------------------------------------
if [ "$USE_GPU" = "1" ]; then
  echo "== Building with CUDA =="
  cargo build --release --features cuda
else
  echo "== Building (CPU only) =="
  cargo build --release
fi

# --- 6. Run -----------------------------------------------------------------
CKPT="checkpoint.json"
[ -d /content ] && CKPT="/content/bitsearch_checkpoint.json"

# Pass all arguments straight through to the binary. Add a checkpoint default
# only for a real search (not --selftest).
ARGS=("$@")
if [[ " ${ARGS[*]} " != *"--checkpoint"* && " ${ARGS[*]} " != *"--selftest"* ]]; then
  ARGS+=(--checkpoint "$CKPT")
fi
exec ./target/release/bitsearch "${ARGS[@]}"
