# Data Setup

All datasets are publicly available — no credentialed PhysioNet access required.

---

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (first time only)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync

# Run any script inside the environment
uv run python src/preprocess.py

# Or activate the venv manually
source .venv/bin/activate
```

`uv.lock` is committed to the repo — `uv sync` reproduces the exact environment on any machine.

**Important:** Always use `uv run python` rather than bare `python` to ensure you're using the project venv, not any active conda environment. Do not mix conda and uv for this project.

**Intel Mac note:** `torch==2.2.2` is pinned locally — the last version with Intel Mac (x86_64) support. On RunPod (Linux + CUDA), replace with: `uv add torch --index https://download.pytorch.org/whl/cu121`

---

## Directory Structure

```
data/
├── code15/          # CODE-15% dataset (raw HDF5 + converted WFDB)
├── ptbxl/           # PTB-XL dataset (WFDB format natively)
└── samitrop/        # SaMi-Trop dataset (confirm access before downloading)
```

Create this structure before downloading:
```bash
mkdir -p data/code15 data/ptbxl data/samitrop
```

---

## PTB-XL

Open access, Creative Commons 4.0. No signup required. Source: https://physionet.org/content/ptb-xl/1.0.3/

```bash
rsync -Cavz physionet.org::ptb-xl/1.0.3/ data/ptbxl/
```

Expected size: ~2.3 GB. Already in WFDB format — no conversion needed.

---

## CODE-15%

Source: https://zenodo.org/records/4916206

**For local prototyping, download part0 only (~2.7 GB, ~20k records):**
```bash
cd data/code15
curl -L "https://zenodo.org/records/4916206/files/exams.csv?download=1" -o exams.csv
curl -L "https://zenodo.org/records/4916206/files/exams_part0.zip?download=1" -o exams_part0.zip
curl -L "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip" -o code15_chagas_labels.zip
unzip exams_part0.zip
unzip code15_chagas_labels.zip
```

**For full dataset (RunPod only — 46 GB total, 18 parts):**
```bash
for i in $(seq 0 17); do
  curl -L "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1" -o exams_part${i}.zip
done
```

Convert HDF5 → WFDB format using the challenge script:
```bash
git clone --depth 1 https://github.com/physionetchallenges/python-example-2025.git
uv run python python-example-2025/prepare_code15_data.py \
  -i data/code15/exams_part0.hdf5 \
  -d data/code15/exams.csv \
  -l data/code15/code15_chagas_labels.csv \
  -o data/code15/wfdb
```

---

## SaMi-Trop

Source: https://zenodo.org/records/4905618 — open access, Creative Commons 4.0.

```bash
cd data/samitrop
curl -L "https://zenodo.org/records/4905618/files/exams.csv?download=1" -o exams.csv
curl -L "https://zenodo.org/records/4905618/files/exams.zip?download=1" -o exams.zip
unzip exams.zip
```

Expected size: ~265 MB total.

---

## Verifying the Download

After downloading, verify WFDB files are readable:
```python
import wfdb
record = wfdb.rdrecord('data/ptbxl/records500/00000/00001_hr')
print(record.p_signal.shape)  # should be (5000, 12) for PTB-XL at 500 Hz
```

---

## Notes

- `data/` is gitignored — raw data is never committed to the repo
- All preprocessing is handled in code — re-running `preprocess_ecg()` on the raw files reproduces any derived dataset
- PTB-XL is 500 Hz and must be resampled to 400 Hz during preprocessing (see Phase 2)
