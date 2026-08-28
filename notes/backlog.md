# Project Backlog

Items here are captured quickly during focused work. Triage regularly to keep this actionable.

> **Active plan:** see [spec/next-session-plan.md](../spec/next-session-plan.md) for the sequenced
> Phase 1 → 2 → 3 implementation plan (hardening → explainability → scale-up). The items below
> tagged **[P1]/[P2]/[P3]** map to that plan's phases.

---

- [ ] **[P3][IMPROVEMENT]** Hyperparameter tuning for full-CODE-15% training
  - _Added: 2026-08-28 | Context: the 2026-08-28 full-data run (all 18 CODE-15% parts +
    SaMi-Trop + PTB-XL) used untuned defaults (lr=1e-3, batch=256, `CosineAnnealingLR`
    with `T_max` pinned to the planned 10-epoch budget) and early-stopped at epoch 1
    (`EarlyStopping(monitor="val/auroc", patience=5)`) — best AUROC 0.8296 / AUPRC 0.1403
    on the CODE-15% test split. Epoch-1-best means the LR schedule never annealed past
    ~60% of its cycle before patience ran out, so it's unclear whether this is a real
    optimum or an artifact of the schedule/patience interaction._
  - _Update 2026-08-28: code prerequisites landed in `src/train.py` — optimizer is now
    AdamW (Loshchilov & Hutter 2019) instead of Adam+L2, scheduler is `ReduceLROnPlateau`
    (matches Ribeiro et al. 2020's recipe on this same CODE dataset lineage) instead of a
    fixed-`T_max` cosine schedule, and `EarlyStopping` patience is 8 (> the scheduler's
    patience=3, so a run can't be killed right as an LR drop would help). Verified with a
    local CPU `--fast` smoke run — no crash, scheduler steps correctly. **Still open:**
    actually run the 3-point LR sweep ({3e-4, 1e-3, 3e-3}, log-spaced per Goodfellow et al.
    *Deep Learning* §11.4.1) on a pod against the full dataset; pick the winner by
    `val/tpr_top5pct` (challenge metric) with `val/auroc` as tie-break, then use that as the
    baseline before/alongside transformer work._

- [ ] **[BUG]** Patient-level split is not reproducible across different HDF5 subsets
  - _Added: 2026-08-28 | Context: `_patient_level_splits()` in `src/dataset.py` calls
    `sklearn.train_test_split(pids, random_state=42)` on whatever patient_id array is
    passed in. Because the shuffle depends on the array's contents/order, a given
    patient's train/val/test assignment can differ depending on which HDF5 parts are
    loaded — e.g. reconstructing the split from `exams_part0.hdf5` alone does NOT
    reproduce the same per-patient split as the full 18-part training run used. This
    only matters for two runs of the same code with different data scope (not for a
    single training run's own train/eval consistency, which is fine), but it means
    `src/explain.py` can't reconstruct the exact held-out test set for a checkpoint
    trained on more parts than are available locally. Fix: derive each patient's split
    bucket from a hash of `(patient_id, seed)` compared against fixed thresholds, so
    membership depends only on the patient, not on what else is in the array._

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

