---
status: superseded
applies_to: src/dataset.py, src/explain.py
---
# 0003 — Keep the existing patient split; make it independent of which HDF5 parts are on disk

DRAFT by Claude from the 2026-10-09 chat decision ("keep existing split"). Edit freely; set `status: approved` yourself.

**Decision:** `Code15Dataset` gets an optional split universe: compute `_patient_level_splits` over all labelled exams in `data/code15/exams.csv`, then restrict to the exam_ids present in the loaded HDF5 files. `explain.py` uses it and rejects exam_ids not in the requested split. Also (no output change): `GradCAM1D` hook handles are stored and removed.

**Why:** `explain.py` rebuilt the split from part0 only, so its "test" exams were really in train or val (Aug 28 "TP" exam 2905181 is in val; "FP" exam 784010 is in train). Offline reconstruction over the exams.csv universe reproduces the full-run split exactly: 240,775 / 51,120 / 51,529, 991 test positives. Keeping it means old and new runs share one test set.

**Alternatives considered:** hash-based per-patient bucketing (part-independent, but a new split, so old results are not comparable).

**Effect on reported numbers or checkpoints:** none on training (default behaviour unchanged for training runs); explainability figures from 2026-08-28 are invalidated as held-out examples.

**How it is verified:** split sizes and positives above are reproduced; exams 2905181 and 784010 are rejected for `--split test`.
