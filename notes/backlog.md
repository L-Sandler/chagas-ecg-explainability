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

- [ ] **[INFRA]** Verify install from scratch and data source consistency
  - Test a clean `uv sync` + install to catch any missing or mis-pinned deps
  - Audit whether preprocessing is applied consistently across CODE-15% (HDF5), SaMi-Trop (WFDB), and PTB-XL (WFDB) — currently `Code15Dataset` does inline normalization while `WFDBDataset` delegates to `preprocess_ecg`; confirm the same steps (bandpass, resample, clip, z-score) apply to all three or document intentional differences
  - _Added: 2026-04-23 | Context: raised during first training run_
