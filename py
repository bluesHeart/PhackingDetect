#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/.venv/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "Missing venv interpreter: $PY" >&2
  echo "Create it with: python3 -m venv --without-pip .venv && .venv/bin/python /tmp/get-pip.py && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

exec "$PY" "$@"

