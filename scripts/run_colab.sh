#!/usr/bin/env bash
# One-command runner for bitsearch on Google Colab (and any Linux box with an
# NVIDIA GPU). Installs the Rust toolchain if needed, clones the repo, builds
# with CUDA, and runs.
#
# Usage (from a Colab cell):
#   !bash <(curl -sSL https://raw.githubusercontent.com/<user>/bitsearch/main/scripts/run_colab.sh) \
#        -- 0xSTART_HEX 0xEND_HEX 1TARGET_P2PKH...
#
# Set BITSEARCH_REPO to override the repository URL.
set -euo pipefail

REPO_URL="${BITSEARCH_REPO:-https://github.com/yourname/bitsearch.git}"
REPO_DIR="bitsearch"

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
cd "$REPO_DIR"

# --- 4. Build ---------------------------------------------------------------
if [ "$USE_GPU" = "1" ]; then
  echo "== Building with CUDA =="
  cargo build --release --features cuda
else
  echo "== Building (CPU only) =="
  cargo build --release
fi

# --- 5. Run -----------------------------------------------------------------
CKPT="checkpoint.json"
[ -d /content ] && CKPT="/content/bitsearch_checkpoint.json"

# GPU is the default when built with `--features cuda` (done above when a GPU
# is present); no flag needed.
exec ./target/release/bitsearch \
  --start "$1" --end "$2" --target "$3" --checkpoint "$CKPT"
