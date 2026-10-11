---
status: approved
applies_to: src/dataset.py
---
# 0008 — Exclude CODE-15% records with fewer than 28 real samples

DRAFT by Claude from your 2026-10-11 answer. Edit freely; only you set `status: approved`.

**Decision:**
- A CODE-15% record whose real length is under 28 samples (0.07 s at 400 Hz) is excluded from
  `Code15Dataset`, in every split. Real length is the span from the first to the last non-zero
  row, as in `preprocess._strip_zero_padding`; an all-zero record has length 0.
- The filter runs after patient-split assignment, like the existing "no label row" exclusion, so
  split membership does not change.
- Affects 692 of 345,797 records (scan 2026-10-11): 688 all-zero, 4 of length 27. By split:
  485 train, 90 val, 98 locked, 19 without a label row. 2 positives in train+val.

**Why:**
- 28 is the lowest length the bandpass (`butter(4, [0.5, 40])`, `filtfilt` padlen 27) can process.
  The 4 length-27 records raised `ValueError` in `baseline-crop2934`.
- All-zero records contain no signal; today they become an all-zero input and receive one
  identical score.
- 28 is a pipeline floor, not a physiological one. The data cannot set a better threshold:
  4,280 train+val records are shorter than the 2,934 crop and 7 of them (0.16%) are positive.

**Alternatives considered / rejected:**
- Keep the all-zero records (the pipeline runs on them): rejected, no information to learn from.
- A minimum duration in seconds (e.g. one beat, or 2 s): not chosen, no basis yet to pick the
  number. Revisit after the first run if `length_diagnostic.py` or a clinician gives one.

**Effect on reported numbers or checkpoints:**
- Removes 0.2% of records and 2 train+val positives; locked-split labels were masked in the scan.
- Reported metrics apply to ECGs with a signal. Limitation to state: the model is not defined for
  records without signal; deployment would need a quality gate that rejects them.
- No earlier checkpoint is affected (earlier runs trained on these records).

**How it is verified:** `scan_lengths.py` counts (above); after the change `len(Code15Dataset)`
drops by the matching per-split counts; `baseline-crop2934` relaunch runs past epoch 0 batch 124.
