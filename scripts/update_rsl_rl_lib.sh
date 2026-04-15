#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/update_rsl_rl_lib.sh [--no-sync]

Updates the locked rsl-rl-lib Git commit in uv.lock by resolving the current
pyproject.toml source, then optionally syncs the local environment.

Options:
  --no-sync   Update uv.lock only. Skip uv sync.
  -h, --help  Show this help text.
EOF
}

DO_SYNC=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync)
      DO_SYNC=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

echo "[INFO] Updating rsl-rl-lib in uv.lock"
uv lock --upgrade-package rsl-rl-lib

if [[ "$DO_SYNC" == "1" ]]; then
  echo "[INFO] Syncing environment"
  uv sync
fi
