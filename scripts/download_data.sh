#!/usr/bin/env bash
# Fetch every input file, then verify it against spec/data-manifest.json. Idempotent: files
# already present are skipped. Run from the repo root (on the pod, the data lives on /workspace).
#
#   bash scripts/download_data.sh
#
# Inputs (see spec/data-manifest.json for sizes, sha256 and sources):
#   CODE-15%   18 HDF5 parts + exams.csv (Zenodo 4916206 v1.0.0), labels CSV (PhysioNet Challenge 2025)
#   SaMi-Trop  exams.hdf5 + exams.csv (Zenodo 4905618 v1.0.0)
#   PTB-XL     v1.0.3 via the HTTPS zip (rsync, port 873, is blocked on RunPod)
# Total ~72 GB; the 18 CODE-15% zips are ~2.7 GB each. Zenodo's published md5 of each zip is
# compared with the download (the values are in scripts/data_manifest.py).
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=spec/data-manifest.json
Z15="https://zenodo.org/records/4916206/files"
CURL=(curl -fL --retry 5 --retry-delay 10 -C -)

published_md5() {  # $1 = repo path of the file the zip unpacks to; empty if Zenodo publishes none
    python3 scripts/data_manifest.py --published-md5 "$1"
}

fetch_zip() {  # $1 = url, $2 = zip file, $3 = manifest path it unpacks to
    "${CURL[@]}" -o "$2" "$1"
    local want got
    want="$(published_md5 "$3")"
    if [ -n "$want" ]; then
        got="$(md5sum "$2" | cut -d' ' -f1)"
        [ "$got" = "$want" ] || { echo "MD5 MISMATCH for $2: got $got, Zenodo publishes $want"; exit 1; }
        echo "  $2 md5 matches Zenodo"
    else
        echo "  $2: no published md5 known, not checked"
    fi
    unzip -q -o "$2"
    rm "$2"
}

mkdir -p data/code15 data/samitrop data/ptbxl

echo "=== CODE-15%: 18 parts, labels, exams.csv ==="
cd data/code15
for i in $(seq 0 17); do
    if [ -f "exams_part${i}.hdf5" ]; then
        echo "  exams_part${i}.hdf5 present, skipping"
    else
        echo "  fetching exams_part${i}.zip"
        fetch_zip "${Z15}/exams_part${i}.zip?download=1" "exams_part${i}.zip" "data/code15/exams_part${i}.hdf5"
    fi
done
if [ ! -f code15_chagas_labels.csv ]; then
    "${CURL[@]}" -o code15_chagas_labels.zip "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip"
    unzip -q -o code15_chagas_labels.zip
fi
# exams.csv (age, sex, patient_id) is the patient-split universe (notes/decisions/0004); training refuses to start without it.
if [ ! -f exams.csv ]; then
    "${CURL[@]}" -o exams.csv "${Z15}/exams.csv?download=1"
fi
cd ../..

echo "=== SaMi-Trop ==="
cd data/samitrop
if [ ! -f exams.hdf5 ]; then
    fetch_zip "https://zenodo.org/records/4905618/files/exams.zip?download=1" exams.zip "data/samitrop/exams.hdf5"
fi
if [ ! -f exams.csv ]; then
    "${CURL[@]}" -o exams.csv "https://zenodo.org/records/4905618/files/exams.csv?download=1"
fi
cd ../..

echo "=== PTB-XL 1.0.3 ==="
if [ ! -f data/ptbxl/ptbxl_database.csv ]; then
    "${CURL[@]}" -o data/ptbxl.zip "https://physionet.org/content/ptb-xl/get-zip/1.0.3/"
    mkdir -p data/ptbxl_unzip
    unzip -q -o data/ptbxl.zip -d data/ptbxl_unzip
    # the zip nests everything in one directory; the code expects the files directly in data/ptbxl/
    inner="$(find data/ptbxl_unzip -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner"/* data/ptbxl/
    rmdir "$inner" data/ptbxl_unzip
    rm data/ptbxl.zip
else
    echo "  PTB-XL already present, skipping"
fi

echo ""
echo "=== Verify against ${MANIFEST} ==="
python3 scripts/data_manifest.py --verify
