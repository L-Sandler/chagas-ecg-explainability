# Dataset Descriptive Statistics

All numbers below are verified against actual files on disk (not claimed from docs).
Generated 2026-05-15. Re-run `uv run python src/audit_preprocessing.py` to revalidate.

---

## CODE-15% — full dataset (all 18 parts) and part0

The full CODE-15% dataset spans 18 HDF5 files (exams_part0 … exams_part17).
**Only part0 is downloaded locally** (prototyping). For GPU training, download all 18 parts.

### Full dataset (all 18 parts, from the label CSV)

| Property | Value |
|---|---|
| Total labeled exams | 343,424 |
| Unique patients | 233,513 |
| **Chagas positive exams** | **6,561 (1.91%)** |
| Chagas negative exams | 336,863 |

Training on part0 alone uses only **6% of all available positives** (403 of 6,561). Positive class statistics are the primary reason to use all parts.

### part0 only (`data/code15/exams_part0.hdf5`)

| Property | Value |
|---|---|
| HDF5 records | 20,001 |
| HDF5 tensor shape | (20001, 4096, 12) float32 |
| Records with chagas label | 19,939 (62 unlabeled, silently dropped) |
| Unique patients | 19,349 |
| **Chagas positive exams** | **403 / 19,939 (2.02%)** |
| Chagas positive patients | 388 / 19,349 (2.01%) |
| Exams per patient | min=1, max=3, mean=1.03, median=1 |
| Patients with >1 exam | 571 |

**Split: patient-level stratified 70/15/15, seed=42**

| Split | Exams | Patients | Positive | Prevalence |
|---|---|---|---|---|
| train | 13,974 | 13,543 | 283 | 2.03% |
| val | 2,972 | 2,902 | 59 | 1.99% |
| test | 2,993 | 2,904 | 61 | 2.04% |

- Stratified on whether a patient has any chagas-positive exam, so prevalence is uniform across splits.
- Zero patient overlap across all three splits (verified by audit).
- The old 90/10 index split had 109 patients leaking between train and val — fixed.

**Known issues:**
- 62 of 20,001 part0 records have no entry in the labels CSV. They are dropped silently in `Code15Dataset.__init__` (see backlog).
- ~50% of records are 7-second recordings (2800 samples), zero-padded to 4096 in HDF5. `preprocess_signal` strips trailing zeros before bandpass and normalization.

---

## SaMi-Trop (`data/samitrop/exams.hdf5`)

| Property | Value |
|---|---|
| HDF5 records | 1,631 |
| HDF5 tensor shape | (1631, 4096, 12) **float64** (unlike CODE-15% float32) |
| CSV rows | 1,631 (positionally aligned, no exam_id in HDF5) |
| CSV columns | exam_id, age, is_male, normal_ecg, death, timey, nn_predicted_age |
| **Chagas label** | **ALL POSITIVE** — implicit, per `prepare_samitrop_data.py:178` |
| Unique patients | Assumed 1 exam/patient (no patient_id column) |

**Split:** Index-based 70/15/15 in `WFDBDataset` (no patient ID available for stratification).

**Notes:**
- This is the only serologically confirmed positive cohort — strong supervision signal.
- The spec calls for oversampling or upweighting SaMi-Trop in training to compensate for CODE-15% weak labels. Not yet implemented.
- ~20% of records are 7s zero-padded, same as CODE-15%.

---

## PTB-XL 1.0.3 (`data/ptbxl/`)

| Property | Value |
|---|---|
| Total WFDB records | 43,598 paths (21,799 lr + 21,799 hr) |
| Unique recordings | 21,799 |
| Unique patients | 18,869 (per PhysioNet page) |
| Sampling rates | 100 Hz (records100/_lr), 500 Hz (records500/_hr) |
| Used resolution | **500 Hz (_hr)** — higher quality, resampled to 400 Hz in preprocess_signal |
| **Chagas label** | **ALL NEGATIVE** — geographic assumption, per `prepare_ptbxl_data.py:132` |

**Known issue:** PTB-XL 1.0.3 `RECORDS` file has a missing newline at the lr/hr boundary. Audit and any code reading `RECORDS` must use a regex, not line-split (handled).

---

## Cross-dataset summary

| Dataset | Records | Chagas | Source |
|---|---|---|---|
| CODE-15% (part0 only) | 19,939 | 2.0% weak labels | Brazilian public health system |
| SaMi-Trop | 1,631 | 100% (all positive) | Serological confirmed |
| PTB-XL | 21,799 | 0% (all negative) | European cohort, geographic assumption |

**Total (part0 only):** 43,369 records, 2,034 positives (4.7%).

**Domain shift risk:** All three datasets are different populations. PTB-XL is European; CODE-15% and SaMi-Trop are Brazilian public health. The challenge hidden test sets (REDS-II, SaMi-Trop 3, ELSA-Brasil) are additional Brazilian cohorts with different demographic distributions. A model overfit to CODE-15% prevalence and waveform characteristics may not generalize.

---

## On the challenge hidden test set

The PhysioNet 2025 Challenge test datasets (REDS-II, SaMi-Trop 3, ELSA-Brasil) were **never released publicly**. They were held by the organizers for scoring submitted Docker containers. They are not downloadable.

The held-out `test` split in this project (2,993 exams from CODE-15% part0) is our internal stand-in for test evaluation — it gives a leakage-free estimate of in-distribution performance, but it does NOT test cross-population generalization the way the challenge test set did.

To measure domain shift, evaluate on the full SaMi-Trop dataset held out from training, or on a regional holdout from CODE-15% if metadata permits (e.g., by exam date).
