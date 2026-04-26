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

- [ ] **[BUG]** Wire SaMi-Trop and PTB-XL into training
  - `train.py` only loads `Code15Dataset`; the project's "multi-source training" framing isn't realized yet
  - SaMi-Trop is HDF5 (`exams.hdf5`, no `exam_id` key, label implicit-positive per `prepare_samitrop_data.py:178`); needs its own dataset class — `WFDBDataset` does not work for it
  - PTB-XL is WFDB but has no chagas column; label is implicit-negative per `prepare_ptbxl_data.py:132`. Needs a manifest CSV + adapter
  - _Added: 2026-04-25 | Context: surfaced during data integrity audit_

- [ ] **[BUG]** Stratified splitting (chagas label + patient_id)
  - Current 90/10 split in `Code15Dataset` and `WFDBDataset` is positional (`iloc[:cutoff]`). With ~2% positivity, val class balance is high-variance. CODE-15% labels CSV has `patient_id`, so the same patient can appear in both splits → leakage
  - PTB-XL ships a `strat_fold` column (10-fold CV) — use it directly
  - For CODE-15%: group-stratified split on `patient_id` with `chagas` as the strat target
  - _Added: 2026-04-25 | Context: surfaced when reviewing data setup_

- [ ] **[BUG]** Variable-length recordings keep zero-padding through preprocessing
  - CODE-15% and SaMi-Trop store recordings as 7s OR 10s padded to 4096 samples (per Zenodo docs and audit measurements: 5/10 sampled CODE-15% records had <4000 nonzero samples on lead II)
  - `preprocess_signal` truncates to 4000 samples but does not strip leading/trailing zero padding → bandpass + z-score distort the padded region; `prepare_code15_data.py:188` strips this padding before WFDB conversion
  - Decision needed: replicate the padding-strip step inside `preprocess_signal`, OR pre-strip in the dataset class, OR skip 7s recordings entirely
  - _Added: 2026-04-25 | Context: surfaced during data integrity audit_

- [ ] **[BUG]** Silent label drop in `Code15Dataset.__init__`
  - 62 of 20,001 CODE-15% part0 records have no entry in `code15_chagas_labels.csv` and are silently dropped by the inner-join filter at `dataset.py:44`. They never appear in train or val
  - Fix options: (a) error on init if any HDF5 record lacks a label, (b) explicitly mark unlabeled records and exclude with a warning count, (c) document this as expected behavior of the challenge label set
  - _Added: 2026-04-25 | Context: surfaced during data integrity audit_

