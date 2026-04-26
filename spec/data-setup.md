# Data Setup

Three publicly-available 12-lead ECG datasets feed this project. All schema facts below are **verified** — either against the primary source page (Zenodo / PhysioNet) or against the official PhysioNet 2025 Challenge conversion scripts in `python-example-2025/`. After downloading, run the audit script to confirm everything matches:

```bash
uv run python src/audit_preprocessing.py
```

Treat any FAIL or unexpected WARN as a blocker before training.

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
```

`uv.lock` is committed — `uv sync` reproduces the exact environment.

**Always use `uv run python`** (not bare `python`) so an active conda env can't shadow the project venv.

**Intel Mac note:** `torch==2.2.2` is pinned locally (last x86_64 build). On RunPod (Linux + CUDA), replace with: `uv add torch --index https://download.pytorch.org/whl/cu121`

---

## Directory layout

```
data/
├── code15/          # CODE-15% — exams_partN.hdf5 + label CSV
├── ptbxl/           # PTB-XL  — WFDB records100/ + records500/
└── samitrop/        # SaMi-Trop — exams.hdf5 + exams.csv
```

```bash
mkdir -p data/code15 data/ptbxl data/samitrop
```

`data/` is gitignored — never commit raw datasets.

---

## CODE-15% (training source)

- **Primary source:** https://zenodo.org/records/4916206
- **Full dataset:** 345,779 exams from 233,770 patients, 18 HDF5 parts
- **For local prototyping use part0 only:** 20,001 records, ~3.7 GB unzipped
- **Format:** HDF5; keys `[exam_id, tracings]`; tracings shape `(N, 4096, 12)` float32
- **Sampling:** 400 Hz
- **Recording length:** 10s (4000 samples) OR 7s (2800 samples), zero-padded to 4096 per record (variable length is verified empirically; `prepare_code15_data.py:188` strips this padding before WFDB conversion)
- **Lead order in HDF5:** `[I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]` (Zenodo page calls Lead I "DI", Lead II "DII", Lead III "DIII"; the official conversion script `prepare_code15_data.py:139` uses the I/II/III spellings)
- **Units:** mV, asserted at `prepare_code15_data.py:141` (Zenodo page does not specify)
- **Chagas labels:** separate CSV (`exam_id, patient_id, chagas`) downloaded from the PhysioNet 2025 Challenge mirror

```bash
cd data/code15
curl -L "https://zenodo.org/records/4916206/files/exams.csv?download=1" -o exams.csv
curl -L "https://zenodo.org/records/4916206/files/exams_part0.zip?download=1" -o exams_part0.zip
curl -L "https://moody-challenge.physionet.org/2025/data/code15_chagas_labels.zip" -o code15_chagas_labels.zip
unzip exams_part0.zip
unzip code15_chagas_labels.zip
```

For the full dataset (RunPod only):

```bash
for i in $(seq 0 17); do
  curl -L "https://zenodo.org/records/4916206/files/exams_part${i}.zip?download=1" -o exams_part${i}.zip
done
```

**Known issue — silent label drop:** 62 of 20,001 part0 records have no entry in the labels CSV and are silently filtered out by `Code15Dataset.__init__` (see backlog).

---

## SaMi-Trop (positive-only Chagas cohort)

- **Primary source:** https://zenodo.org/records/4905618
- **Total:** 1,631 ECGs (one per patient) from a 1,959-patient cohort
- **Format:** HDF5; key `[tracings]` only — **no `exam_id` array**, records are positionally aligned to `exams.csv`
- **Tracings shape:** `(1631, 4096, 12)` **float64** (note: differs from CODE-15% float32)
- **Sampling:** 400 Hz; same 7s/10s zero-padding to 4096 as CODE-15%
- **Lead order, units:** same as CODE-15%, asserted at `prepare_samitrop_data.py:127–129`
- **CSV columns:** `[exam_id, age, is_male, normal_ecg, death, timey, nn_predicted_age]` — **no chagas column**
- **Chagas label is implicit, all positive** per `prepare_samitrop_data.py:178` ("All of the patients in the SaMi-Trop dataset are Chagas positive.")

```bash
cd data/samitrop
curl -L "https://zenodo.org/records/4905618/files/exams.csv?download=1" -o exams.csv
curl -L "https://zenodo.org/records/4905618/files/exams.zip?download=1" -o exams.zip
unzip exams.zip
```

---

## PTB-XL (negative-only control cohort)

- **Primary source:** https://physionet.org/content/ptb-xl/1.0.3/
- **Total:** 21,799 records from 18,869 patients
- **Format:** WFDB (`.dat` + `.hea` pairs); two resolutions provided:
  - `records100/…/<id>_lr` at **100 Hz**, 1000 samples (10s)
  - `records500/…/<id>_hr` at **500 Hz**, 5000 samples (10s)
- **Storage:** 1 μV/LSB raw integer values; WFDB header `units` field is `'mV'`, so `wfdb.rdrecord(...).p_signal` returns mV-scaled float64 — verified by audit
- **Lead order in WFDB header (verified):** `[I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]` (the PhysioNet page text reads "AVL, AVR" but that's prose — trust the header)
- **Chagas label is implicit, all negative** per `prepare_ptbxl_data.py:132` (`label = False`)

```bash
rsync -Cavz physionet.org::ptb-xl/1.0.3/ data/ptbxl/
```

**Known issue — `RECORDS` file:** PTB-XL 1.0.3 ships a `RECORDS` file with a missing newline between the last `_lr` line (`records100/21000/21837_lr`) and the first `_hr` line (`records500/00000/00001_hr`). Naive `for line in f` parsing concatenates them. The audit script handles this with a regex; downstream code that walks `RECORDS` should do the same.

---

## Optional: convert CODE-15% HDF5 → WFDB

Useful if you want a uniform WFDB pipeline across all three datasets. The conversion script is the authoritative reference for lead order, fs, and units claims above.

```bash
git clone --depth 1 https://github.com/physionetchallenges/python-example-2025.git
uv run python python-example-2025/prepare_code15_data.py \
  -i data/code15/exams_part0.hdf5 \
  -d data/code15/exams.csv \
  -l data/code15/code15_chagas_labels.csv \
  -o data/code15/wfdb
```

---

## Verifying the download

After any download or update, run:

```bash
uv run python src/audit_preprocessing.py
```

Expected on a healthy setup:
- `PASS` on every schema, count, units, bandpass, normalization, determinism, and step-C check
- `WARN` only on the four known issues:
  1. CODE-15% has 7s recordings whose zero-padding survives preprocessing
  2. SaMi-Trop has 7s recordings (same story)
  3. SaMi-Trop CSV has no chagas column (implicit positive)
  4. CODE-15% has 62 part0 records with no chagas label (silently dropped today)
  5. PTB-XL `RECORDS` has the missing-newline quirk above

Anything else FAIL or WARN means something has changed upstream and the audit + this doc need updating before training.

---

## Preprocessing contract (what every dataset becomes)

`preprocess_signal()` in `src/preprocess.py` is the single shared entry point. All three datasets must produce the same downstream tensor:

- shape `(12, 4000)` float32
- canonical lead order: `[I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]`
- bandpass [0.5, 40] Hz (4th-order Butterworth, applied via `filtfilt` → 8th-order magnitude response)
- resample to 400 Hz if needed (PTB-XL is 500 Hz, others are already 400 Hz)
- per-lead z-score (after a ±5 mV clip)
- deterministic (verified)

The audit script enforces every property above on real samples from each dataset and on a synthetic frequency-response test. PTB-XL signals at 500 Hz get resampled to 400 Hz inside `preprocess_signal`; CODE-15% and SaMi-Trop are passed `fs=400` directly.
