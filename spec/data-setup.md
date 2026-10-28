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

**Torch:** the lock resolves `torch 2.2.2` (CPU) on the Intel Mac and `2.6.0+cu124` on Linux via `[tool.uv.sources]`. Do not `uv add` on a pod; see `spec/gpu-setup.md` §2.

---

## Data inventory

Paths, sizes and sha256 of every file are in `spec/data-manifest.json`. Download and verify with
`scripts/download_data.sh` and `scripts/data_manifest.py --verify` (`spec/gpu-setup.md` §3). Local
machines may hold only part0 for smoke tests; training uses all 18 parts.

`data/code15/wfdb/` (part0 converted to WFDB) is optional and not used by the training code, which reads HDF5.

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
- **part0:** 20,001 records; the other parts are listed in `spec/data-manifest.json`
- **Format:** HDF5; keys `[exam_id, tracings]`; tracings shape `(N, 4096, 12)` float32
- **Sampling:** 400 Hz
- **Recording length:** every record is stored as 4096 rows. Short (7 s) recordings hold 2,934 real samples with ~581 zero rows on each side; long (10 s) recordings fill all 4096 rows. Measured 2026-10-28 on every 20th part0 record (1,000 records): 2,934 real samples in 608, 4,096 in 364, 28 other lengths including 2 all-zero. `prepare_code15_data.py:188` strips the padding before WFDB conversion
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

**Known issue — records without a label:** 62 of 20,001 part0 records have no entry in the labels CSV. `Code15Dataset.__init__` excludes them and prints the count (`[Code15Dataset] N/M HDF5 records had no label row; excluded.`). Records with fewer than 28 real samples are also excluded (`notes/decisions/0008-min-real-length-28.md`).

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

Expect `PASS` on every schema, count, units, bandpass, normalization and determinism check. `WARN`s
are data-dependent (padded 7 s recordings, SaMi-Trop with no chagas column, part0 records with no
label, flat leads, the PTB-XL `RECORDS` quirk above): read each one against the known issues in this
file. Anything else FAIL or WARN means something has changed upstream.

> The audit's padding checks still assume a 4,000-sample window (written before
> `notes/decisions/0006-crop-to-common-length.md`). Treat their wording as stale until the script is revisited.

---

## Preprocessing contract (what every dataset becomes)

`preprocess_signal()` in `src/preprocess.py` is the single shared entry point. Rationale for each step is in `notes/decisions/`. All three datasets produce the same downstream tensor:

- shape `(12, TARGET_LEN)` float32, `TARGET_LEN = 2934` (0006); constants live in `src/preprocess.py`
- canonical lead order: `[I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]`
- zero-padding stripped at both ends of the recording (0001), then cropped to the first 2,934 samples at native fs, before the bandpass (0006); shorter records are zero-padded afterwards
- bandpass [0.5, 40] Hz (4th-order Butterworth, applied via `filtfilt`)
- resample to 400 Hz if needed (PTB-XL is 500 Hz, others are 400 Hz)
- per-lead z-score after a ±5 mV clip
- CODE-15% records with fewer than 28 real samples are excluded (0008)
- deterministic

`src/audit_preprocessing.py` checks these properties on real samples and on a synthetic frequency-response test.
