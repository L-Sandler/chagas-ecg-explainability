# Project Backlog

Items here are captured quickly during focused work. Triage regularly to keep this actionable.

> **Active plan:** see [spec/next-session-plan.md](../spec/next-session-plan.md) for the sequenced
> Phase 1 → 2 → 3 implementation plan (hardening → explainability → scale-up). The items below
> tagged **[P1]/[P2]/[P3]** map to that plan's phases.

---

- [x] **[P1][BUG]** Challenge metric is mislabeled/wrong — `_tpr_at_fpr` computes TPR@5%FPR, not TPR@top-5%-ranked
  - _Fixed: 2026-07-21 | Added `_tpr_at_top_k` in `src/train.py`, logged as `val/tpr_top5pct`; renamed the old metric to `val/tpr_at_5pct_fpr` everywhere it's logged/printed. Checkpoint monitor left on `val/auroc` per plan decision. Verified on `--fast` run: both metrics print, no crash._

- [x] **[P2][FEATURE]** Explainability harness (`src/explain.py`) — Grad-CAM then Integrated Gradients
  - _Fixed: 2026-07-21 | `src/explain.py` loads a checkpoint, fetches a record by exam_id (or auto-picks the highest-confidence TP/FP), and saves attribution overlays to `reports/explainability/`. Grad-CAM hooks `model.blocks[-1]` (last residual/SE block before global pooling); IG uses Captum with a zero baseline. Verified end-to-end on the epoch-12 checkpoint — 4 PNGs produced with no errors, overlays visually sane (attribution concentrates on QRS complexes)._


- [ ] **[FEATURE]** Build visualizations of data suitable for demoing the project
  - _Added: 2026-04-22 | Context: want something to show end-to-end pipeline output visually_

- [ ] **[IMPROVEMENT]** Add lightweight sample mode for training/testing that caps compute usage
  - _Added: 2026-04-22 | Context: need a fast iteration path without burning full training runs_

- [ ] **[TECH-DEBT]** Dockerize the project for reproducibility
  - _Added: 2026-04-22 | Context: ensure environment is portable and shareable_

- [ ] **[RESEARCH]** Simulate tech review sessions with a sr engineer or clinician persona to defend every design decision
  - _Added: 2026-04-22 | Context: build depth of understanding, not just working code — prep for volunteer interviews_

- [ ] **[INFRA]** Verify install from scratch
  - Test a clean `uv sync` + install on a fresh machine to catch missing or mis-pinned deps
  - _Added: 2026-04-23 | Context: split off from the broader "data + install" backlog item once data consistency was verified separately_

- [x] **[BUG]** Wire SaMi-Trop and PTB-XL into training
  - _Fixed: 2026-05-19 | `SamiTropDataset` and `PTBXLDataset` added to `dataset.py`; `train.py` wired with `--samitrop` and `--ptbxl` flags via `ConcatDataset`. `WFDBDataset` removed (was unused and used wrong split strategy)._

- [x] **[BUG]** Stratified splitting (chagas label + patient_id)
  - _Fixed (prior to 2026-05-19) | `_patient_level_splits()` in `dataset.py` implements patient-level stratified 70/15/15 with seed 42. Zero patient overlap verified by `check_split_leakage()` in `audit_preprocessing.py`. PTB-XL uses official `strat_fold` (folds 1–8/9/10)._

- [x] **[BUG]** Variable-length recordings keep zero-padding through preprocessing
  - _Fixed: 2026-05-02 | `_strip_zero_padding` added to `preprocess_signal`; truncation now happens before normalization so z-score stats reflect only real cardiac signal_

- [x] **[BUG]** `--fast` smoke test didn't cap auxiliary datasets
  - _Fixed: 2026-05-19 | Aux datasets (SaMi-Trop, PTB-XL) are skipped when `--fast` is set. Fast mode is for verifying the CODE-15% pipeline path only; aux sources add 19k records and make the "smoke test" take 22 min on CPU._

- [x] **[P3][INFRA]** Experiment tracking for GPU training runs
  - _Fixed: 2026-07-21 | Added `--run-name` to `train.py` (names the W&B run and namespaces the checkpoint dir); W&B config now logs `pos_weight`, per-source dataset sizes, batch size, epochs, lr, and git SHA; `WandbLogger(log_model=True)` saves the best checkpoint as a W&B artifact. Verified with `WANDB_MODE=offline` smoke run — no crash. Still needed before a real RunPod run: set `WANDB_API_KEY` in the pod environment._

- [x] **[P1][BUG]** Silent label drop in `Code15Dataset.__init__`
  - _Fixed: 2026-07-21 | Option (b): `Code15Dataset.__init__` now prints `"[Code15Dataset] N/M HDF5 records had no label row; excluded."` before filtering. Verified count matches audit: 62/20001 on part0._

