---
status: approved
applies_to: src/dataset.py, src/train.py, src/explain.py
---
# 0004 — Re-split from scratch: train / val / locked-final (supersedes 0003)

DRAFT by Claude from your 2026-10-09 answers. Edit freely; only you set `status: approved`.

**Decision:**
- New patient-level three-way split of CODE-15%: **70 / 15 / 15** = train / val / locked-final, seed 42.
- Stratified on **chagas (patient has any positive exam) x sex x age band**. Age bands (10-year, young end merged because under-40 positives are rare): under 40 | 40-49 | 50-59 | 60-69 | 70-79 | 80+ (edges left-closed: [17,40) [40,50) ... [80,100]). Each patient's age band is taken from their first exam. (To check at implementation: patients whose recorded sex differs across exams.)
- Computed over the full labelled CODE-15% universe (`data/code15/exams.csv`), then restricted to the HDF5 parts that are loaded, so membership never depends on which parts are on disk (closes the old split BUG).
- **Locked-final is read once, at the very end.** Development, model selection, hyperparameters, explainability exploration and the equity audit all use val. `train.py` no longer evaluates the locked set (the old `--eval-test` becomes a val-only report), and `explain.py` rejects the locked split unless explicitly unlocked. A hook-level unlock file (`.guardrails/unlock-final-test`) will be drafted as a separate guardrail patch.
- SaMi-Trop (all positive) and PTB-XL (all negative) remain **train-only** for the baseline (PTB-XL keeps its official folds only for the specificity check). A with/without ablation is deferred.

**Why:**
- The old test set cannot be treated as untouched: metrics from it were printed after every run and used to compare runs.
- Earlier issues (padding bug, last-epoch evaluation) already invalidate old numbers, so comparability with them has little value.
- Stratifying on sex and age gives every subgroup proportional positives in the locked set, which the equity report needs.

**Alternatives considered:** carve locked-final out of the old test patients (keeps train fixed, but the old test is already seen in aggregate); label-only stratification (subgroup counts left to chance); 60/15/25 (tighter subgroup intervals, less training data).

**Effect on reported numbers or checkpoints:** all earlier checkpoints and numbers (AUROC 0.8296 / 0.8276, TPR@top-5% 0.3804 / 0.3875) refer to the old split and are not comparable. Rough expected size (scaled from label-file totals, not measured on the new split): ~985 positives in locked-final; per band roughly 85 (under 40), 171, 203, 209, 208, 109 (80+), about half of each per sex.

**How it is verified:** zero patient overlap across the three splits; split sizes and per-stratum counts printed; membership identical whether one HDF5 part or all 18 are loaded; positives per sex/age band per split reported.
