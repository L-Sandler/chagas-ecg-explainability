#!/usr/bin/env bash
# Download the training data subset (part0 + SaMi-Trop + PTB-XL) to data/.
# Run once on a fresh pod when no network volume is attached.
# Total: ~7.5 GB. Expects a fast datacenter connection (~5-10 min).
# Usage: bash scripts/download_data.sh
set -euo pipefail

mkdir -p data/code15 data/samitrop data/ptbxl

echo "=== CODE-15% part0 + labels ==="
cd data/code15
if [ ! -f exams_part0.hdf5 ]; then
    curl -L "https://zenodo.org/records/4916206/files/exams_part0.zip?download=1" -o exams_part0.zip
    unzip -q exams_part0.zip && rm exams_part0.zip
else
    echo "  exams_part0.hdf5 already present, skipping"
fi
if [ ! -f code15_chagas_labels.csv ]; then
    curl -L "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip" -o code15_chagas_labels.zip
    unzip -q code15_chagas_labels.zip && rm code15_chagas_labels.zip
else
    echo "  code15_chagas_labels.csv already present, skipping"
fi
cd ../..

echo "=== SaMi-Trop ==="
cd data/samitrop
if [ ! -f exams.hdf5 ]; then
    curl -L "https://zenodo.org/records/4905618/files/exams.zip?download=1" -o exams.zip
    unzip -q exams.zip && rm exams.zip
    curl -L "https://zenodo.org/records/4905618/files/exams.csv?download=1" -o exams.csv
else
    echo "  exams.hdf5 already present, skipping"
fi
cd ../..

echo "=== PTB-XL ==="
if [ ! -f data/ptbxl/ptbxl_database.csv ]; then
    rsync -Cavz --progress physionet.org::ptb-xl/1.0.3/ data/ptbxl/
else
    echo "  PTB-XL already present, skipping"
fi

echo ""
echo "Download complete. Verify with:"
echo "  du -sh data/code15/exams_part0.hdf5 data/samitrop/exams.hdf5 data/ptbxl/"
echo "  uv run python src/audit_preprocessing.py --n-samples 20"
