# Next-Session Implementation Plan

**Author of plan:** Opus session, 2026-07-21
**Intended implementer:** Sonnet (following this doc)
**Status of project:** Subset GPU pipeline validated 2026-06-13. Baseline on CODE-15% test
split: AUROC 0.7746, AUPRC 0.0868, `tpr_at_5pct` 0.3443 (⚠️ metric is mislabeled — see Phase 1).
Best checkpoint: `lightning_logs/run_2026-06-13_subset/checkpoints/best-epoch=12-val/auroc=0.829.ckpt`.

## Guiding decision for this stage

The focus of the next work is **wire-up and scaffolding, not valid results.** The current
checkpoint is trained on only 1/18 of CODE-15% (part0), so explainability output is not
expected to be clinically meaningful yet. We deliberately build the harness now on the
existing checkpoint and re-run it against a properly-trained model later (Phase 3).

Ordering (decided with user): **Phase 1 hardening → Phase 2 explainability → Phase 3 scale-up.**

---

## Phase 1 — Metric + data hardening

Goal: make the numbers trustworthy and the dataset honest before building on top of them.
This is the only phase that must be "correct"; it gates every comparison to the challenge
leaderboard.

### 1.1 Fix the challenge metric (highest priority)

**Problem:** `_tpr_at_fpr()` at [src/train.py:102-106](../src/train.py#L102-L106) computes
TPR at **5% false-positive-rate**. The PhysioNet Challenge 2025 metric (`spec/spec.md`) is
**TPR among the top-5% of patients ranked by predicted probability** — a different, harder
quantity. The current 0.3443 is therefore NOT comparable to the winner's 0.323.

**Fix:**
- Add a new function, e.g. `_tpr_at_top_k(labels, probs, k_frac=0.05)`:
  - Rank all samples by `probs` descending.
  - Take the top `ceil(k_frac * N)` as "sent for testing."
  - Return `(# true positives in that top slice) / (total positives)`.
- Log it as `val/tpr_top5pct` (keep the old `tpr_at_5pct` too, but **rename its label** to
  `tpr_at_5pct_fpr` everywhere it's logged/printed — [src/train.py:94](../src/train.py#L94),
  [src/train.py:136](../src/train.py#L136) — so the two are never confused again).
- Update the `=== TEST SET RESULTS ===` print block (~[src/train.py:130-140](../src/train.py#L130))
  to show both, with the top-5% one labeled as the **challenge score**.
- Set `ModelCheckpoint` monitor: consider switching from `val/auroc`
  ([src/train.py:220](../src/train.py#L220)) to `val/tpr_top5pct` since that's the true
  objective. **Decision for implementer:** keep AUROC as the checkpoint monitor for now
  (more stable early-training signal at 2% prevalence), but log both. Revisit at scale-up.

**Acceptance:** on a `--fast` run, both metrics print; top-5% metric is ≤ the FPR metric on
the same data (sanity: top-5%-ranked is generally the stricter number here); no crash on a
batch with zero positives (guard: return 0.0 or nan-safe).

### 1.2 Fix the silent label drop

**Problem:** [src/dataset.py:117](../src/dataset.py#L117) inner-joins the labels CSV to the
HDF5 exam_ids. 62 of 20,001 part0 records have no label row and vanish with no warning.

**Fix (option b from backlog):** after the filter, compute the count of HDF5 exam_ids that
had no label row and `print`/`warnings.warn` an explicit count. Do **not** hard-error
(option a) — the challenge label set legitimately doesn't cover every CODE-15% exam. Keep the
records excluded, but make the exclusion loud and logged, not silent.

**Acceptance:** constructing `Code15Dataset` on part0 prints something like
`"[Code15Dataset] 62/20001 HDF5 records had no label row; excluded."` Count matches the audit.

### 1.3 (Prerequisite for Phase 3, do now if cheap) W&B run hygiene

Not required for Phase 2. Wire this before Phase 3 scale-up:
- Add `--run-name` arg to [src/train.py](../src/train.py) so sweeps are distinguishable.
- Log as W&B config: `pos_weight`, per-source dataset sizes, batch size, epochs, git SHA.
- Save best checkpoint as a W&B artifact.
See backlog INFRA item for full checklist. Deferrable.

---

## Phase 2 — Explainability harness (wire-up focus)

Goal: a **reusable explainability module** that takes a trained checkpoint + an ECG record and
produces attribution overlays. Runs on the existing epoch-12 checkpoint. Correctness of the
*insights* is explicitly NOT the bar this stage — the wiring is.

Decided approach: **Grad-CAM first, then Integrated Gradients.**

### 2.1 New module `src/explain.py`

- Load a checkpoint into the `model.py` LightningModule; put in eval mode.
- Function to fetch a single record (by dataset + index) already preprocessed, plus its label
  and predicted probability.

### 2.2 Grad-CAM (1D)

- Register forward/backward hooks on the **last conv block before global pooling** in the
  1D ResNet ([src/model.py](../src/model.py) — locate the final residual/SE block; document
  which layer was chosen and why).
- Compute channel-weighted activation map, ReLU, upsample to the input time length (per lead
  or shared across leads — start shared over time axis).
- Output: a `(time,)` or `(leads, time)` importance array.

### 2.3 Integrated Gradients

- Use Captum `IntegratedGradients` (add `captum` to `pyproject.toml` deps). Baseline = zeros
  (flat signal). Attribute per-lead-per-timestep.
- Reuse the same record-fetch + checkpoint-load plumbing from 2.1.

### 2.4 Visualization

- Plot the 12-lead ECG with the attribution overlaid (heatmap under the trace, or color the
  trace by attribution). Reuse plotting conventions from
  [src/plot_preprocess.py](../src/plot_preprocess.py) where possible.
- Save to `reports/explainability/` (create dir). One figure per method.
- A tiny CLI or notebook entry point: pick a **true-positive** record (highest-confidence
  correct positive) and a **false-positive**, render both — these are the interesting cases
  to eyeball even on a weak model.

**Acceptance:** `python src/explain.py --checkpoint <path> --record <id> --method gradcam`
produces a saved overlay PNG without error, for both gradcam and integrated-gradients. No
claim about clinical validity required.

---

## Phase 3 — Scale-up training (later)

Prereq: Phase 1 metric fix + Phase 1.3 W&B done.
- Provision A100 40GB per `reference_runpod_gpu_selection` memory (datacenter EU-2 to match
  the `/workspace` network volume). RTX 4090 acceptable for a mid-size run.
- Download remaining CODE-15% parts (1–17) via `scripts/download_data.sh`; ~4-5 hr full run.
- Train full CODE-15% + SaMi-Trop + PTB-XL, log the corrected challenge metric to W&B.
- Re-run the Phase 2 explainability harness on the good checkpoint → this is where the
  explanations become worth interpreting.
- rsync `lightning_logs/` back before terminating the pod. Do NOT delete `/workspace`.

---

## Deferred / not in this plan
- Bandpass cutoff ablation (`notes/learning/open-questions.md`).
- Dockerization, clean-install verification (backlog).
- Demo visualizations of the pipeline (backlog) — partly covered by Phase 2 output.
