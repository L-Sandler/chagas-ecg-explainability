#!/usr/bin/env bash
# Build the pinned environment on a pod, from uv.lock. Run from the repo root, once per volume:
# the pinned uv and the venv both live on /workspace, so later pods find them in place.
#
#   bash scripts/setup_runpod.sh
#
# What it does: install the uv version named by pyproject.toml's required-version into
# /workspace/bin, `uv sync --frozen` (never re-resolves), confirm the venv equals the lock,
# then scripts/env_check.py --require-gpu (CUDA wheel vs driver, one AMP forward/backward).
# It does not touch data (scripts/download_data.sh) and does not train.
set -euo pipefail
cd "$(dirname "$0")/.."

UV_VERSION="$(sed -n 's/^required-version *= *"==\(.*\)"/\1/p' pyproject.toml)"
[ -n "$UV_VERSION" ] || { echo "pyproject.toml has no required-version = \"==x.y.z\""; exit 1; }
BIN=/workspace/bin
export PATH="$BIN:$PATH"

echo "=== 1. GPU and driver ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true

echo ""
echo "=== 2. uv ${UV_VERSION} (pinned) ==="
if ! command -v uv >/dev/null 2>&1 || ! uv --version | grep -q "^uv ${UV_VERSION} "; then
    mkdir -p "$BIN"
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | env UV_INSTALL_DIR="$BIN" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

echo ""
echo "=== 3. uv sync --frozen ==="
if [ -d .venv ]; then
    echo "note: .venv already exists; --frozen will bring it to exactly uv.lock."
    echo "      To keep a hand-installed venv for the record, mv it aside before running this."
fi
uv sync --frozen
uv sync --frozen --check
echo "venv matches uv.lock"

echo ""
echo "=== 4. env_check (CUDA wheel vs driver, AMP forward/backward) ==="
.venv/bin/python scripts/env_check.py --require-gpu

echo ""
echo "=== Environment ready. Next: bash scripts/download_data.sh (verifies against spec/data-manifest.json) ==="
