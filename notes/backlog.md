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
    local CPU `--fast` smoke run — no crash, scheduler steps correctly._
  - _Update 2026-08-28 (later same day): ran 1 of 3 planned LR points on an RTX 4090 —
    lr=1e-3 stopped at epoch 9, AUROC 0.8276 / AUPRC 0.1366 / TPR@top-5% 0.3875 (challenge
    metric improved over the original 0.3804; AUROC/AUPRC dipped slightly). Best checkpoint
    still epoch 1 — looks like a real property of this data/model, not a schedule artifact.
    User paused the sweep here to control cost; **lr=3e-4 and lr=3e-3 were not run.**
    Checkpoint saved to `lightning_logs/full-code15-lr1e-3/`. Pod terminated. **Still open:**
    the remaining two LR points, if/when this gets picked back up._

## 2026-10-09 audit: fixed
- [x] **[BUG]** Leading zero-padding of short recordings never stripped (`preprocess.py`, decision 0001). Every earlier checkpoint was trained on it.
- [x] **[BUG]** Test metrics came from last-epoch weights, not the best checkpoint (`train.py`, decision 0002). `train.py` now evaluates the best checkpoint on val and never touches the locked-final split.
- [x] **[BUG]** `explain.py` rebuilt the split from part0 only, so "test" figures were drawn from train/val exams; replaced by the part-independent split of decision 0004 (supersedes the open split BUG below). `explain.py` now defaults to val, refuses exams outside the requested split, and refuses the locked split without an explicit unlock.
- [x] **[BUG]** `GradCAM1D` hooks were never removed (stacked on repeated use).
- [x] **[FEATURE]** Sex/age-band subgroup report with patient-level bootstrap CIs (`src/equity.py`, decision 0005). Verified on synthetic scores only; not yet run on a trained model.

## 2026-10-09 audit: open (decisions for the user, not tasks)
- [x] **[DECISION]** Recording length tracks the label (93.3% of positive vs 60.0% of negative CODE-15% exams are short). Resolved 2026-10-10 by decision 0006: crop every input to the first 2,934 samples before filtering; `src/length_diagnostic.py` reports performance by original length on val.
- [ ] **[DECISION]** SaMi-Trop (all positive) and PTB-XL (all negative) make the label perfectly predictable from source inside the training set. Baseline keeps both as train-only (0004); a with/without ablation is deferred.
- [ ] **[DECISION]** Two exams from the 2026-08-28 explainability figures (2905181, 784010) fall in the new locked-final split and their figures have been seen. Options: exclude those patients from the final evaluation, or disclose.
- [ ] **[DECISION]** Explainability method choices: example selection (`pick_interesting_records` scans only the first 1,000 records), IG visualisation (per-lead min-max on signed values loses sign and cross-lead comparability), IG zero baseline cannot see padding, Grad-CAM is shared across all 12 leads.
- [x] **[DECISION]** `pos_weight` ignored the added SaMi-Trop positives and PTB-XL negatives. Resolved 2026-10-10 by decision 0007: computed over the full train set (about 40.7, was 51.3). Selection stays on val/auroc.
- [x] **[BUG]** PTB-XL fold-10 "specificity check" printed nothing (no positives). Resolved by 0007: reports the fraction flagged at the val top-5% cutoff. `train.py` also writes `val_preds.csv`.
- [ ] **[VERIFY]** Residual length cue after the 0006 crop (decided 2026-10-10: keep all records, look first). In part0, by raw length: 2,934 samples 3.22% positive (11,624 exams), 2,935-4,095 0.42% (476), 4,096 0.35% (7,529); under 2,934: 272 exams, 0 positive, still zero-padded after the crop; all-zero records: 38 (1 positive). SaMi-Trop: 97 of 1,631 are under 2,934 (all positive). Padded records form a zero-tail cue, negative in CODE-15% and positive in SaMi-Trop. To check after the baseline: `python src/length_diagnostic.py lightning_logs/<run>/val_preds.csv` (the "<2934 (padded)" group, and AUROC within 7 s vs 10 s). If the model leans on it, options are excluding all-zero / under-2,934 exams (needs its own decision record).
- [ ] **[DATA]** 1,825 of 4,444 positive patients have exams with differing chagas labels (labels are per exam; training uses exam labels as the challenge does; the split stratifies on "any positive exam").
- [ ] **[INFO]** Val exams with a Level-1 finding (exams.csv labels): RBBB 1,388 (185 pos), AF 1,011 (76 pos), 1dAVb 826 (33 pos).
- [ ] **[INFRA]** The new split needs CODE-15% `exams.csv` (age, sex) next to the labels file; `scripts/download_data.sh` does not fetch it. Confirm it is on the RunPod volume before a run.
- [ ] **[INFRA]** `scripts/sweep_driver.sh` marks a run done by grepping for `TEST SET RESULTS`; `train.py` keeps that string in its header for now even though the report is on val. Update the driver's marker when convenient.
- [ ] **[DATA]** 372 of 233,513 patient_ids have an age spread of 10+ years across their exams (max 82), so some ids probably mix different people. Split age uses each patient's minimum recorded age.
- [ ] **[CLEANUP]** `reports/explainability/*.png` (2026-08-28) are not held-out and predate the fixes; regenerate or remove.

- [x] ~~**[BUG]** Patient-level split is not reproducible across different HDF5 subsets~~ (superseded: fixed by decision 0004, 2026-10-09)
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


- [ ] **[IMPROVEMENT]** Typeset the paper in LaTeX (arXiv single-column)
  - _Added: 2026-10-10 | Deferred; paper starts as `paper/draft.md`. Plan: `paper/main.tex` (article, 11pt, natbib/plainnat) with one file per section, `refs.bib`, figures exported as PDF. Build with tectonic via the VSCode LaTeX Workshop extension (already installed). Convert once the draft is stable or a venue is picked._

- [ ] **[TECH-DEBT]** Consolidate project markdown that predates the 2026-10 fixes
  - _Added: 2026-10-10 | Last-touched dates (git): `spec/spec.md` and `.claude/skills/*.md` Apr 22; `notes/learning/*` Apr 22-23; `spec/data-setup.md`, `spec/dataset-stats.md` May 15; `spec/gpu-setup.md` Jun 12; `spec/next-session-plan.md` Jul 21; `spec/fable-experiment-facilitator-prompt.md` Sep 11 (still describes the $15 wallet, the lr-sweep, and the Aug checkpoints as current). Several still mention the 4000-sample input, the part0-only split or "test set". Per CLAUDE.md the AI-written ones are not source material for the paper. Work: per file decide keep / update / archive / delete; the current-state record is `notes/decisions/`, `notes/runs.md`, `notes/backlog.md`. Protected files (facilitator prompt, CLAUDE.md, root README) are edited by the user._

- [ ] **[TECH-DEBT]** Redesign the guardrail hook so behaviour matches its stated intent
  - _Added: 2026-10-10 | User's assessment: the guardrails do not do what they were designed to do. To be scoped in its own session (plan mode). Seed observations from 2026-10-10, not an audit: (1) a shell command was blocked only because its text mentioned a guarded path while writing to an unguarded file (matching is on command text, not target); (2) `git add`/`git commit` of `spec/runs/` passes while `cp`/`sed -i` to it is blocked; (3) the code gate is per file (`applies_to`), not per change, and a script that writes a gated file from inside Python is not seen; (4) approvals are a stale mix: `approve-lr-sweep` (Sep 11, empty, past its 8 h TTL) and a `budget.json` global cap of $50 against a real balance of $25.87; (5) the facilitator prompt and `.guardrails/README.md` describe different states of the project. Before redesign: write down, per rule, the failure it is meant to prevent._
