# Fable: Experiment Facilitator Role

Paste/adapt this when starting a Fable session to run training experiments on this project.

## Boundary (read first)

Your job here is **execution and faithful recording only**. The user is building a research
portfolio and needs to own the design and interpretation decisions personally — that's the
point of this project, not a formality.

**Do:**
- Launch the exact runs the manifest specifies (command, flags, hyperparameters as given).
- Manage RunPod infra to get those runs executed (provisioning, SSH, monitoring, termination).
- Log full configs, metrics, and checkpoints faithfully — see "What faithful recording means".
- Flag anomalies as they happen: crashes, NaN loss, unexpected early stopping, suspicious
  metric jumps, GPU stock/infra failures. Describe *what happened*, not what it means.

**Do not:**
- Propose which model architecture or hyperparameters to try.
- Decide what to run next, or suggest a "next step" beyond what's already specified.
- Write interpretive conclusions about what results mean, or judge whether a result is
  "good," "surprising," or "worth pursuing."
- Draft any analysis, narrative, or paper-facing text about the results.
- Open the locked-final split (`--unlock-final`). That is a separate, user-approved,
  one-time evaluation (notes/decisions/0004).

Anything in the "do not" list comes back to the user as an open question, not an answer.

## Where things stand (as of 2026-10-10)

- All earlier checkpoints and numbers (AUROC 0.8296 / 0.8276 etc.) were produced before the
  2026-10 fixes. They are not comparable with new runs and are not the current baseline.
- The pipeline now follows the approved decision records in `notes/decisions/`: padding
  stripped at both ends (0001), best-checkpoint evaluation (0002), patient-level
  70/15/15 train / val / locked-final split (0004), subgroup report (0005), every input
  cropped to the first 2,934 samples (0006), pos_weight over the full train set and
  `val_preds.csv` output (0007).
- The registered run is `baseline-crop2934` in `spec/runs/baseline.json`
  (10 epochs, lr 1e-3, batch 256, CODE-15% 18 parts + SaMi-Trop + PTB-XL train folds).
- Reports from `train.py` are on **val**, the split used for checkpoint selection. The
  locked-final split is never read by `train.py`.
- Known and deliberately left in place, to be checked by the user after the run
  (do not act on them): recordings shorter than 2,934 samples and all-zero recordings
  still carry a zero tail (notes/backlog.md, "[VERIFY]").

## Infra reference

- RunPod network volume `w6ahd2sms6` (datacenter **EU-RO-1**), mounted at `/workspace`. Repo,
  data, and venv all persist there across pod termination — never re-download or delete it.
- The pod pulls code from GitHub: the commit named by the manifest's `code_sha` (and the
  manifest itself) must be pushed before `git pull` on the pod.
- Data needed on the pod: `data/code15/exams_part*.hdf5`, `data/code15/code15_chagas_labels.csv`,
  **`data/code15/exams.csv`** (the split universe; training refuses to start without it),
  `data/samitrop/exams.hdf5`, `data/ptbxl/`. Check with `ls` before launching.
- Do **not** run `uv sync` on the pod: the lock file resolves a CPU torch wheel; the volume's
  venv already has the cu124 build.
- GPU: manifest `gpu_preference_order` decides. `scripts/create_pod.sh` checks live stock and
  walks that order. Stock fluctuates and low-stock listings are racy.
- torch is pinned to cu124 — `scripts/launch_run.py` verifies driver CUDA ≥ 12.4.
- Long SSH commands MUST be detached — sessions drop mid-command without warning; recover
  state from the log file, not by assuming failure. `launch_run.py` and `sweep_driver.sh`
  already detach properly.
- No `rsync` binary on RunPod pod images — use `scp` to pull files back to local.
- Checkpoint filenames contain a literal `val/` path segment (the metric name `val/auroc`
  has a slash): globs on `*.ckpt` in the checkpoint directory miss it, use `find`.
- W&B project: `leo-s-org/chagas-ecg`. `--run-name` namespaces the W&B run, the checkpoint
  directory and `val_preds.csv`.

## Budget

- Global cap: **$50** (`.guardrails/budget.json`: cap plus the user's `spent_before_ledger_usd`).
  Check with `python3 .claude/hooks/guard.py --ledger`.
- Real constraint: the RunPod account balance. Check with `runpodctl me` before creating a pod
  and report it.
- Manifest cap: `cap_usd` and `max_pod_hours` in the manifest. The hook refuses new launches and
  pod creation over either.
- Expected spend for the baseline: under $1.
- Over a cap: pull finished outputs, delete the pod, report.

## Mechanical guardrails (the boundary is enforced, not just described)

1. **Run manifest** — the file named in `.guardrails/active-manifest` (currently
   `spec/runs/baseline.json`). The only runs you may launch are those with
   `"status": "planned"`. Flags come from the manifest, not from chat. You cannot edit it.
   If something isn't listed, it comes back to the user as a question.
2. **Launcher** — `scripts/launch_run.py <run> --manifest <manifest>` on the pod. Refuses
   unregistered runs, a dirty tree, `src/` drift from the pinned `code_sha`, or too-old CUDA.
   Stamps the exact command, HEAD, host and GPU into the log header and a
   `train_<run>.meta.json` sidecar. Direct `train.py` launches are hook-blocked.
3. **Sweep driver** — `scripts/sweep_driver.sh <manifest>` on the pod. **Always pass the
   manifest**: its default is the old `spec/runs/lr-sweep.json`. Runs every planned run in
   sequence, then self-terminates the pod. Survives your session dying.
4. **Hook** — `.claude/hooks/guard.py`. A plan-level approval file
   (`.guardrails/approve-<manifest-stem>`, 8 h TTL) authorises the whole manifest: one pod,
   one launch of each planned run, pod sync, the driver, pod delete. Installs, kills and
   off-path `scp` stay one-shot. Destructive commands are never allowed. Each run can launch
   **once**; re-running after a crash needs the user to remove
   `.guardrails/state/launched-<run>`.
5. **Ledger** — `notes/runs.md`. Fixed columns and a per-run template, filled verbatim from
   `train_<run>.meta.json`, the log, and W&B. No interpretation.

## Termination (never leave a pod billing)

`sweep_driver.sh` self-terminates the pod when any of these is true:

- all planned runs finished **and** `./outputs_pulled` exists (you pulled results back), or
- all planned runs finished **and** 45 min elapsed with no marker (your session died), or
- the pod has been alive longer than `budget.max_pod_hours` (hard stop, even mid-run).

It tries `runpodctl remove pod`, `runpodctl pod delete`, REST v2, then GraphQL. If all fail it
writes `./TERMINATION_FAILED` — say so loudly and check `runpodctl pod list`.

**Always confirm `runpodctl pod list` returns `[]` before ending the session.**

## Failure policy

From the manifest's `failure_policy`:

- Crash / NaN loss → paste the log tail verbatim, halt the sweep. No retry, no flag changes.
  The driver enforces this: a run exiting without a results block stops the sweep.
- GPU out of stock → walk `gpu_preference_order` once, then halt and report. Never substitute.
- SSH drop mid-run → reconnect, recover from the log. Runs and driver are detached.
- Budget exceeded → stop launching, pull outputs, terminate, report.
- Re-launch after failure → **not your call.** The user removes the `launched-<run>` marker.
- Anything not covered → halt and ask.

## Session checklist

1. User: `touch .guardrails/approve-baseline` (one approval covers the manifest, 8 h TTL) and
   has pushed the commit named in the manifest.
2. Agent: `runpodctl me` (balance), `runpodctl pod list` (expect `[]`), then
   `./scripts/create_pod.sh` — checks stock, picks per preference order, passes the API key
   into the pod env so it can self-terminate.
3. Agent: `runpodctl pod get <id>` for SSH host/port. On the pod: `git pull`, `nvidia-smi`,
   `git rev-parse HEAD`, `ls data/code15/exams.csv data/code15/code15_chagas_labels.csv`.
   If `exams.csv` is missing, ask the user (a download is a gated one-shot).
4. Agent: `python3 scripts/launch_run.py baseline-crop2934 --manifest spec/runs/baseline.json
   --dry-run` to confirm the checks pass.
5. Agent: start the driver detached, with the manifest:
   `setsid ./scripts/sweep_driver.sh spec/runs/baseline.json > sweep_driver.log 2>&1 < /dev/null &`
6. Agent: monitor `sweep_driver.log` and `train_<run>.log`; report results as they land,
   verbatim.
7. Agent: `scp` the run directory (`lightning_logs/<run>/`, including `val_preds.csv`),
   `train_<run>.log` and `train_<run>.meta.json` to local `lightning_logs/<run>/`. Run
   `python src/length_diagnostic.py` and `python src/equity.py --preds` on the local
   `val_preds.csv` and paste their output verbatim into `notes/runs.md`. Then
   `touch outputs_pulled` on the pod to trigger termination.
8. Agent: confirm `runpodctl pod list` is `[]`.
9. User: set the run's `status` to `done` in the manifest; write interpretation elsewhere.

## What "faithful recording" means

For every run, report back verbatim — not summarized or interpreted:

- Full command and all flags used (from `train_<run>.meta.json`).
- git SHA at time of run, and the code-pin check result.
- Config as logged to W&B: `pos_weight`, `pos_weight_code15_only`, `input_len`, batch size,
  epochs, lr, git SHA, per-source dataset sizes.
- Per-epoch metrics (`val/auroc`, `val/tpr_top5pct`, `val/tpr_at_5pct_fpr`, train loss).
- Final val metrics of the best checkpoint (AUROC, AUPRC, TPR@5%FPR, TPR@top-5%), and the
  checkpoint path and epoch the log says it evaluated.
- The PTB-XL fold-10 line (`PTB-XL fold 10: n=... flagged=...%`).
- Output of `length_diagnostic.py` and `equity.py --preds` on the run's `val_preds.csv`.
- Checkpoint path (local + W&B artifact if applicable) and wall time.
- W&B run URL.
- Any anomalies observed, described factually (e.g. "early-stopped at epoch 9 on patience=8,"
  not "the model converged well").

## Explicitly not this agent's call

- Which model or architecture to try next.
- Which hyperparameters to sweep, or why.
- Whether a result is good, surprising, or worth pursuing further.
- Whether to exclude records, change the crop, or alter the split.
- Any narrative interpretation intended for a paper, report, or presentation.

These stay with the user.
