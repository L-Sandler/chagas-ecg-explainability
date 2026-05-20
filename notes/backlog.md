# Project Backlog

Items here are captured quickly during focused work. Triage regularly to keep this actionable.

---

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

- [ ] **[INFRA]** Experiment tracking for GPU training runs
  - W&B integration already exists (`--wandb` flag in `train.py`), but there's no run naming, config logging, or artifact saving strategy
  - Before RunPod: set `WANDB_API_KEY` in the environment, decide on project/group naming convention, log `pos_weight` and dataset sizes as config, save best checkpoint as a W&B artifact
  - Consider adding `--run-name` arg so parallel hyperparameter sweeps are distinguishable
  - _Added: 2026-05-02 | Context: want reproducible experiment history before first real GPU training run_

- [ ] **[BUG]** Silent label drop in `Code15Dataset.__init__`
  - 62 of 20,001 CODE-15% part0 records have no entry in `code15_chagas_labels.csv` and are silently dropped by the inner-join filter at `dataset.py:44`. They never appear in train or val
  - Fix options: (a) error on init if any HDF5 record lacks a label, (b) explicitly mark unlabeled records and exclude with a warning count, (c) document this as expected behavior of the challenge label set
  - _Added: 2026-04-25 | Context: surfaced during data integrity audit_

