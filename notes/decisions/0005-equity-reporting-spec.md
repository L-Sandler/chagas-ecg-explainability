---
status: approved
applies_to: src/equity.py
---
# 0005 — Report model performance by sex and age band (no equity optimisation in v1)

DRAFT by Claude from your 2026-10-09 answers ("accept as is"). Edit freely; only you set `status: approved`.

**Decision:**
- Train to maximise **global** TPR@top-5%. Do not optimise for equity in this iteration; subgroup optimisation is a follow-up/ablation.
- **Report** performance in these groups: sex (2) and six age bands: under 40 | 40-49 | 50-59 | 60-69 | 70-79 | 80+ (same edges as 0004; the young end is merged because under-30 has only ~18 expected locked positives).
- Per group: AUROC; TPR at the **global** top-5% threshold (one threshold for everyone); fraction of the group flagged at that threshold; n, number of positives and prevalence, with AUPRC shown beside prevalence.
- 95% intervals from a bootstrap that resamples **patients**; groups with fewer than ~50 positives are flagged as too small to interpret.
- Development and audit on val; the locked-final set is read once at the end (0004).

**Why:** The global threshold is how the challenge metric works and the realistic deployment view; showing the flagged fraction exposes whether one group is flagged disproportionately. Prevalence differs strongly by age (about 0.2% under 30 and 1.0% in the 30s versus about 2.4-2.7% from 40-79 in the label file), so AUROC and AUPRC need prevalence beside them.

**Alternatives considered:** per-group thresholds (a different question, left for the ablation); sex x age intersections (too few positives per cell); optimising a fairness-constrained objective (deferred by choice).

**Effect on reported numbers or checkpoints:** none on training; adds a reporting module.

**How it is verified:** the overall (all-patient) row reproduces the headline AUROC and TPR@top-5% from the same predictions; group n and positives sum to the totals.

**Known limits to state in the write-up:** CODE-15% has only age and sex (no race or ethnicity) and comes from one country; labels are not individually confirmed.
