#!/usr/bin/env bash
set -euo pipefail

BASE_RUNTIME_ROOT="${1:-${MJLAB_BASE_RUNTIME_ROOT:-${MJLAB_RUNTIME_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp/${USER:-user}/mjlab-runtime}}}}}"
RUNTIME_SUFFIX="${2:-${MJLAB_RUNTIME_SUFFIX:-}}"

# Keep the cluster's existing HOME/TMPDIR by default. This script only redirects
# shared cache/log/artifact paths unless MJLAB_HOME_DIR / MJLAB_TMP_DIR are
# explicitly provided.
HOME_ROOT="${MJLAB_HOME_DIR:-}"
TMP_ROOT="${MJLAB_TMP_DIR:-}"
LOG_ROOT="${MJLAB_OUTPUT_DIR:-$BASE_RUNTIME_ROOT/logs}"

CACHE_ROOT="${MJLAB_CACHE_DIR:-$BASE_RUNTIME_ROOT/cache}"
WANDB_ROOT="${MJLAB_WANDB_DIR:-$BASE_RUNTIME_ROOT/wandb}"
WANDB_ARTIFACT_ROOT="${MJLAB_WANDB_ARTIFACT_DIR:-$WANDB_ROOT/artifacts}"

mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/uv" \
  "$CACHE_ROOT/uv-python" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/wandb" \
  "$CACHE_ROOT/wandb_cache" \
  "$CACHE_ROOT/wandb_config" \
  "$CACHE_ROOT/wandb_data" \
  "$CACHE_ROOT/hf" \
  "$CACHE_ROOT/warp" \
  "$CACHE_ROOT/pycache" \
  "$CACHE_ROOT/mjlab/wandb_motions" \
  "$LOG_ROOT" \
  "$WANDB_ROOT" \
  "$WANDB_ARTIFACT_ROOT"

if [[ -n "$HOME_ROOT" ]]; then
  mkdir -p "$HOME_ROOT"
  export HOME="$HOME_ROOT"
fi

if [[ -n "$TMP_ROOT" ]]; then
  mkdir -p "$TMP_ROOT"
  export TMPDIR="$TMP_ROOT"
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
fi

export MJLAB_RUNTIME_ROOT="$BASE_RUNTIME_ROOT"
export MJLAB_BASE_RUNTIME_ROOT="$BASE_RUNTIME_ROOT"
export MJLAB_RUNTIME_SUFFIX="$RUNTIME_SUFFIX"
export MJLAB_OUTPUT_DIR="$LOG_ROOT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$CACHE_ROOT/xdg}"
export XDG_STATE_HOME="${XDG_STATE_HOME:-$CACHE_ROOT/xdg}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$CACHE_ROOT/uv-python}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_ROOT/cuda}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/mpl}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$CACHE_ROOT/hf}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CACHE_ROOT/warp}"
export WANDB_DIR="${WANDB_DIR:-$WANDB_ROOT}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$CACHE_ROOT/wandb_cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$CACHE_ROOT/wandb_config}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$WANDB_ARTIFACT_ROOT}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$CACHE_ROOT/wandb_data}"
export MJLAB_WANDB_CACHE_DIR="${MJLAB_WANDB_CACHE_DIR:-$CACHE_ROOT/mjlab/wandb_motions}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$CACHE_ROOT/pycache}"

echo "Overlay base runtime root: $BASE_RUNTIME_ROOT"
echo "Overlay runtime suffix: ${RUNTIME_SUFFIX:-<none>}"
echo "HOME=${HOME:-<unchanged>}"
echo "TMPDIR=${TMPDIR:-<unchanged>}"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"
echo "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
echo "TORCH_HOME=$TORCH_HOME"
echo "WARP_CACHE_PATH=$WARP_CACHE_PATH"
echo "WANDB_DIR=$WANDB_DIR"
echo "WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR"
echo "MJLAB_WANDB_CACHE_DIR=$MJLAB_WANDB_CACHE_DIR"
echo "MJLAB_OUTPUT_DIR=$MJLAB_OUTPUT_DIR"
