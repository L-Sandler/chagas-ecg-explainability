# .guardrails — user-only approval and budget files

The hook `.claude/hooks/guard.py` (v3) gates risky actions on files here. **Only the user creates
or edits these**, from a normal terminal. The agent cannot write to `.guardrails/`, `.claude/`,
`spec/runs/` or `notes/decisions/`.

## Files you own

| File | Purpose |
|------|---------|
| `active-manifest` | One line: which `spec/runs/*.json` pod create/delete/sync and launches gate on. Change it when you move to the next experiment batch. `--manifest X` on a command still overrides. |
| `budget.json` | `global_cap_usd` (50) and `spent_before_ledger_usd` (**you** fill this from RunPod billing; while it is `null` every pod create and launch is refused). |
| `approve-<manifest-stem>` | Plan-level approval, see below. |

## Plan-level approval (the normal path)

    touch .guardrails/approve-<manifest-stem>      # e.g. approve-baseline for spec/runs/baseline.json

While it exists and is inside `budget.approval_ttl_hours` it authorises **exactly what that
manifest describes**: one `runpodctl pod create`, one launch of each run whose status is
`planned` (each at most once), `scripts/sweep_driver.sh`, repo sync on the pod, `runpodctl pod
delete`, and `scp` into `lightning_logs/` or the pod repo. Anything off-manifest comes back to
you as a question. Delete the file to revoke.

## Budget (two caps, both enforced)

- **Global:** `budget.json` `global_cap_usd` against `spent_before_ledger_usd` + the ledger.
- **Per manifest:** `budget.cap_usd` in the manifest (0 = no manifest-level cap).

Pod create, run launch and sweep start are refused when either cap is reached. Spend is an
**estimate**: pod-hours × the highest GPU price listed in the manifest's `gpu_price_usd_per_hr`
(or `assumed_usd_per_hr_unlisted`). It errs high. Check it any time:

    python3 .claude/hooks/guard.py --ledger

`state/ledger.jsonl` gets one line per pod lifetime, written when the hook sees
`runpodctl pod delete`, or when `runpodctl pod list` returns empty after the pod
self-terminated (closed at "now", so the estimate can only be too high). Reconcile against the
RunPod billing page and adjust `spent_before_ledger_usd` if the drift matters.

## Design-decision gate (`notes/decisions/`)

The agent can edit `src/preprocess.py`, `dataset.py`, `model.py`, `train.py` and `explain.py`
only if a record in `notes/decisions/` has `status: approved` and lists that file on its
`applies_to:` line. The agent may **draft** records (Write/Edit on `notes/decisions/` is allowed
only while the record's status is `proposed`, before and after the change). It can never set
`approved`, `superseded` or `rejected`, and can never touch a record that already has one of
those. Approval is your one-line edit of the `status:` line; the reasoning in the record is
yours to edit before you do. Limits: this gates the Edit/Write tools and obvious shell writes
(`>`, `sed -i`, `tee`, `mv`, `cp`); it cannot see a script that rewrites a file from inside Python.

## Still one-shot (each file authorises one action, then is deleted)

| File to `touch`        | Authorises                                    |
|------------------------|-----------------------------------------------|
| `unlock-final-test`    | one command that opens the locked-final split (`--unlock-final` / `unlock_final=True`), or writing a `.py`/`.sh`/`.ipynb` that does. Consumed when the command succeeds: the locked set is read once, at the very end (notes/decisions/0004). |
| `approve-pod-install`  | one install/download command on the pod       |
| `approve-kill`         | one `kill`/`pkill`                            |
| `approve-scp`          | one `scp` outside `lightning_logs/`           |

## Local runs

`python src/train.py ... --fast` run locally (no `ssh`) is allowed: it is the CPU smoke test.
Any other direct `train.py` launch is refused; use `scripts/launch_run.py` or
`scripts/sweep_driver.sh`. Commands that merely mention these filenames (`git log`, `grep`,
`cat`) are not gated.

## Never allowed, approval or not

Recursive deletes; any delete touching `/workspace`, `data/`, `lightning_logs/` or a
checkpoint; force-push; branch delete; `git reset --hard`; `git clean`; pod stop/reset/restart;
network-volume delete; and any write to the guardrail files themselves. The root `README.md` is
user-only too: the agent may propose README text in chat, never write it into the file. (Decision
records are the one exception: the agent may draft them as `status: proposed`, see above.)

## State (written by the hook, after commands actually succeed)

`state/pod-created-at`, `state/pod-manifest`, `state/pod-id` describe the pod being billed;
`state/launched-<run>` marks a run as already launched (**re-running is your decision**: remove
the marker). Stale state: `ls .guardrails/state/` and `runpodctl pod list` to compare with reality.
