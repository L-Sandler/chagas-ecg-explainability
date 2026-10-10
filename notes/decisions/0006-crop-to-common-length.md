---
status: approved
applies_to: src/preprocess.py, src/dataset.py, src/model.py
---
# 0006 — Crop every input to the first 2,934 samples (7.3 s)

DRAFT by Claude from your 2026-10-10 answers. Edit freely; only you set `status: approved`.

**Decision:**
- All sources (CODE-15%, SaMi-Trop, PTB-XL) keep the first 2,934 samples at 400 Hz, the real length of a short recording. `TARGET_LEN` goes 4000 → 2934. Records shorter than that are still zero-padded (rare).
- The crop happens at native fs, right after padding is stripped and **before** the bandpass (`ceil(2934 * fs / 400)` samples; then resample; then truncate to 2,934).
- Diagnostic: on val, report performance by original length (<2,934 | 7 s | 10 s). `Code15Dataset.real_lengths()` supplies the original length.

**Why:** 93% of CODE-15% positives vs 60% of negatives are 7 s recordings; after 0001 a short record still ended in 1,066 exact zeros, which the model can read as a label proxy. A common length removes that by construction. Cropping before the filter puts filter edges at both ends of every window; filtering first would leave an end transient inside short windows only.

**Alternatives considered:** keep length and only diagnose (model likely learns the cue); centre or random crop (not chosen: start crop is deterministic and keeps one time axis for attributions).

**Effect on reported numbers or checkpoints:** model inputs change for every record; long records lose ~27% of their signal. All earlier checkpoints are incompatible in meaning (the architecture accepts any length). Device/clinic differences that travel with length can remain; the diagnostic is how we look for them.

**How it is verified:** short and long part0 records and a PTB-XL record all give (12, 2934) with no exact-zero tail; `audit_preprocessing.py` passes; the diagnostic runs on a smoke-run `val_preds.csv`.
