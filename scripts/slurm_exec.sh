#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNTIME_ROOT_ARG="${MJLAB_RUNTIME_ROOT:-}"
RUNTIME_SUFFIX_ARG="${SLURM_JOB_ID:-${MJLAB_RUNTIME_SUFFIX:-}}"

if [[ $# -gt 0 && "$1" != "--" ]]; then
  RUNTIME_ROOT_ARG="$1"
  shift
fi
if [[ $# -gt 0 && "$1" != "--" ]]; then
  RUNTIME_SUFFIX_ARG="$1"
  shift
fi
if [[ $# -gt 0 && "$1" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [runtime_root] [runtime_suffix] -- <command...>" >&2
  echo "Example: $0 /scratch/\$USER/mjlab \"\$SLURM_JOB_ID\" -- uv run train Mjlab-Velocity-Flat-Unitree-G1" >&2
  exit 2
fi

if [[ -n "$RUNTIME_ROOT_ARG" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/slurm_overlay_env.sh" "$RUNTIME_ROOT_ARG" "$RUNTIME_SUFFIX_ARG"
else
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/slurm_overlay_env.sh"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${MJLAB_UV_SYNC:-0}" == "1" ]]; then
  uv sync --frozen --group dev
fi

if [[ -n "${SLURM_JOB_ID:-}" && -n "${MJLAB_SLURM_NOTIFY_TO:-}" ]]; then
  export MJLAB_SLURM_NOTIFY_CWD="$PWD"
  export MJLAB_SLURM_NOTIFY_COMMAND="$(printf '%q ' "$@")"
  if ! python3 "$SCRIPT_DIR/slurm_email_notify.py"; then
    echo "Warning: failed to send Slurm allocation email notification" >&2
  fi
fi

exec "$@"
