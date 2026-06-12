#!/usr/bin/env bash
# Run once on a fresh RunPod pod after cloning the repo.
# Assumes:
#   - Working directory is the repo root
#   - Data volume is already mounted and data/ symlinked or present
# Usage: bash scripts/setup_runpod.sh
set -euo pipefail

echo "=== 1. Install uv ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "source \$HOME/.local/bin/env" >> ~/.bashrc
fi
echo "uv $(uv --version)"

echo ""
echo "=== 2. Install project dependencies ==="
uv sync

echo ""
echo "=== 3. Verify CUDA torch ==="
uv run python -c "
import torch
print(f'torch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

CUDA_AVAILABLE=$(uv run python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_AVAILABLE" = "False" ]; then
    echo ""
    echo "WARNING: CUDA not available after uv sync."
    echo "The lock file installs a CPU torch wheel from PyPI by default."
    echo "Re-install with the CUDA wheel matching your driver (check nvidia-smi):"
    echo ""
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | tr -d '.' | head -1 || echo "unknown")
    if [ "$CUDA_VER" != "unknown" ]; then
        echo "  Detected CUDA driver version: $(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')"
        echo "  Run: uv add torch --index https://download.pytorch.org/whl/cu${CUDA_VER}"
    else
        echo "  Run: uv add torch --index https://download.pytorch.org/whl/cu<version>"
        echo "  (replace <version> with CUDA version from nvidia-smi, e.g. cu124)"
    fi
    exit 1
fi

echo ""
echo "=== 4. Run data audit ==="
uv run python src/audit_preprocessing.py --n-samples 20

echo ""
echo "=== Setup complete. ==="
echo ""
echo "GPU smoke test (verify CUDA path, ~2 min):"
echo "  uv run python src/train.py \\"
echo "      --hdf5 data/code15/exams_part0.hdf5 \\"
echo "      --labels data/code15/code15_chagas_labels.csv \\"
echo "      --fast --epochs 2 --batch-size 64 \\"
echo "      --accelerator gpu --devices 1 --num-workers 4 --amp"
echo ""
echo "Full subset run (part0 + SaMi-Trop + PTB-XL, ~23 min on RTX 3090):"
echo "  uv run python src/train.py \\"
echo "      --hdf5 data/code15/exams_part0.hdf5 \\"
echo "      --labels data/code15/code15_chagas_labels.csv \\"
echo "      --samitrop data/samitrop/exams.hdf5 \\"
echo "      --ptbxl data/ptbxl \\"
echo "      --epochs 30 --batch-size 256 \\"
echo "      --accelerator gpu --devices 1 --num-workers 4 --amp \\"
echo "      --eval-test"
