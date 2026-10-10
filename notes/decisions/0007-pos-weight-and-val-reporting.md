---
status: approved
applies_to: src/train.py
---
# 0007 — pos_weight over the full training set; save val predictions; real PTB-XL check

DRAFT by Claude from your 2026-10-10 answers. Edit freely; only you set `status: approved`.

**Decision:**
- `pos_weight` = negatives / positives over everything the loss sees: (CODE-15% train negatives + PTB-XL train) / (CODE-15% train positives + SaMi-Trop). About 40.7 on the full data (was 51.3, CODE-15% only). Both values are printed and logged.
- After training, the best checkpoint's val predictions are written to `lightning_logs/<run>/val_preds.csv` (exam_id, patient_id, chagas, is_male, age, band, n_real, prob).
- The PTB-XL fold-10 block reports n, mean and median probability, and the fraction at or above the val top-5% cutoff.
- Selection stays on `val/auroc` (amend 0005's wording: select on AUROC, report TPR@top-5%).

**Why:** pos_weight should describe the data the loss is computed on. Only part0 is local, so the equity report, the Level-1 analysis and the 0006 length diagnostic need per-exam val predictions from the pod. The fold-10 check printed nothing because it has no positives; the fraction flagged at the val cutoff is a false-positive rate on an out-of-source set.

**Alternatives considered:** keep 51.3 (ignores 1,631 added positives and 17,418 added negatives); no weighting; select on TPR@top-5% (SD ≈ ±0.016 with ~982 val positives, so epoch choice is noisy).

**Effect on reported numbers or checkpoints:** the loss weighting changes, so new runs are not comparable with the 2026-08 runs (already non-comparable under 0001/0004). Reporting additions do not change the model.

**How it is verified:** a `--fast` smoke run prints both pos_weights, writes val_preds.csv, and prints the PTB-XL flagged fraction; `equity.py --preds` on that file reproduces the val AUROC printed by train.py.
