#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv interpreter: $PY" >&2
  echo "Create it with: python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt" >&2
  echo "If venv creation fails on Debian/Ubuntu, install: sudo apt-get install -y python3-venv" >&2
  exit 1
fi

export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache/huggingface"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" 2>/dev/null || true

# Some systems export CUDA libs globally (e.g., /usr/local/cuda/lib64) which can
# shadow PyTorch's bundled CUDA dependencies and break imports. Prefer the
# venv's `nvidia/*/lib` directories when present.
shopt -s nullglob
NVIDIA_DIRS=("$ROOT"/.venv/lib/python*/site-packages/nvidia)
if (( ${#NVIDIA_DIRS[@]} )); then
  NVIDIA_LIBS="$(find "${NVIDIA_DIRS[@]}" -maxdepth 2 -type d -name lib 2>/dev/null | sort | paste -sd: -)"
  if [[ -n "$NVIDIA_LIBS" ]]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
shopt -u nullglob

exec "$PY" "$@"
