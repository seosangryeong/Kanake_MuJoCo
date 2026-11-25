#!/usr/bin/env bash
set -euo pipefail
rl_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${KANAKE_PYTHON:-python3}" -m venv --system-site-packages "$rl_dir/.venv"
"$rl_dir/.venv/bin/python" -m pip install -r "$rl_dir/requirements.txt" "$@"
