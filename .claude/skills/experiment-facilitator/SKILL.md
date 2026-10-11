---
name: experiment-facilitator
description: Run training experiments on RunPod for the Chagas ECG project. Use when launching manifest runs, provisioning or terminating a pod, monitoring a run, pulling results, or recording a run in notes/runs.md. Execution and faithful recording only; no design or interpretation.
---

# Experiment facilitator

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
- Open the locked-final split. That is a separate, user-approved, one-time evaluation
  (notes/decisions/0004).

Anything in the "do not" list comes back to the user as an open question, not an answer.

## Current state

Not kept here. Read the manifest named in `.guardrails/active-manifest`, the records in
`notes/decisions/`, and `notes/runs.md`. Reports from `train.py` are on **val**; the
locked-final split is never read by `train.py`. Do not compare numbers across pipeline
versions unless a decision record says they are comparable.

## Infra reference

- RunPod network volume `w6ahd2sms6` (datacenter **EU-RO-1**), mounted at `/workspace`. Repo,
  data, and venv persist there across pod termination — never re-download or delete it.
- The pod pulls code from GitHub: the commit named by the manifest's `code_sha` (and the
  manifest itself) must be pushed before `git pull` on the pod.
- Data needed on the pod: `data/code15/exams_part*.hdf5`, `data/code15/code15_chagas_labels.csv`,
  `data/code15/exams.csv` (the split universe; training refuses to start without it),
  `data/samitrop/exams.hdf5`, `data/ptbxl/`. Check with `ls` before launching.
- Do **not** run `uv sync` on the pod: the lock file resolves a CPU torch wheel; the volume's
  venv already has the cu124 build. `scripts/launch_run.py` verifies driver CUDA ≥ 12.4.
- GPU: manifest `gpu_preference_order` decides. `scripts/create_pod.sh` checks live stock and
  walks that order. Stock fluctuates and low-stock listings are racy.
- Long SSH commands MUST be detached — sessions drop mid-command without warning; recover
  state from the log file, not by assuming failure. `launch_run.py` and `sweep_driver.sh`
  already detach properly.
- No `rsync` on pod images — use `scp` to pull files back.
- Checkpoint filenames contain a literal `val/` path segment (the metric name `val/auroc`
  has a slash): globs on `*.ckpt` in the checkpoint directory miss it, use `find`.
- W&B project: `leo-s-org/chagas-ecg`. `--run-name` namespaces the W&B run, the checkpoint
  directory and `val_preds.csv`.

### Pod environment gotchas (observed 2026-10-10)

- `--env` vars reach PID 1 only; SSH sessions do not see `RUNPOD_API_KEY`, `RUNPOD_POD_ID`,
  `WANDB_API_KEY`. Start the driver after `P="$PATH"; . /etc/rp_environment; PATH="$P"`
  (has `RUNPOD_*`, not WANDB), or it cannot self-terminate and meta.json records `pod_id: null`.
- W&B: the user runs `.venv/bin/wandb login` on the pod; `~/.netrc` is on container disk, so
  redo it per pod. Do not extract secrets from `/proc/1/environ`.
- `runpodctl pod create --wait` echoes the env block including keys; pipe through `sed`.

## Guardrails

Details: `.guardrails/README.md`. In short: only runs with `"status": "planned"` in the active
manifest may launch, each once; one plan-level approval `.guardrails/approve-<manifest-stem>`
(8 h TTL) covers one pod, the launches, the driver, pod sync and pod delete; installs, kills
and off-path `scp` are one-shot; destructive commands are never allowed. You cannot edit the
manifest, launcher, hook or approvals. Direct `train.py` launches are blocked.
`scripts/sweep_driver.sh`: **always pass the manifest**.

## Budget

- Global cap and the user's `spent_before_ledger_usd`: `.guardrails/budget.json`. Check with
  `python3 .claude/hooks/guard.py --ledger`.
- The RunPod balance is shared with other uses and is not the budget. Run `runpodctl me` before
  creating a pod and report it, and run `runpodctl pod get <id>` right after create so the
  hourly rate lands in the ledger.
- Manifest cap: `cap_usd` and `max_pod_hours`. Over a cap: pull finished outputs, delete the
  pod, report.

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

1. User: `touch .guardrails/approve-<manifest-stem>` and has pushed the commit named in the manifest.
2. Agent: `runpodctl me` (balance), `runpodctl pod list` (expect `[]`), then
   `./scripts/create_pod.sh` — checks stock, picks per preference order, passes keys into the
   pod env so it can self-terminate.
3. Agent: `runpodctl pod get <id>` for SSH host/port. On the pod: `git pull`, `nvidia-smi`,
   `git rev-parse HEAD`, `ls data/code15/exams.csv data/code15/code15_chagas_labels.csv`.
   If `exams.csv` is missing, ask the user.
4. Agent: `python3 scripts/launch_run.py <run> --manifest <manifest> --dry-run` to confirm the
   checks pass.
5. Agent: start the driver detached, with the manifest:
   `setsid ./scripts/sweep_driver.sh <manifest> > sweep_driver.log 2>&1 < /dev/null &`
6. Agent: monitor `sweep_driver.log` and `train_<run>.log`; report results as they land, verbatim.
7. Agent: `scp` the run directory (`lightning_logs/<run>/`, including `val_preds.csv`),
   `train_<run>.log` and `train_<run>.meta.json` to local `lightning_logs/<run>/`. Run
   `python src/length_diagnostic.py` and `python src/equity.py --preds` on the local
   `val_preds.csv` and paste their output verbatim into `notes/runs.md`. Then
   `touch outputs_pulled` on the pod to trigger termination.
8. Agent: confirm `runpodctl pod list` is `[]`, then run `python3 .claude/hooks/guard.py --ledger`
   and paste the output verbatim (this session's cost and the running total).
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
