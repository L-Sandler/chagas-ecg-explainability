# .guardrails — user-only approval files

The hook `.claude/hooks/guard.py` gates risky actions on files here. **Only the user creates
these**, from a normal terminal.

## Plan-level approval (the normal path)

One file authorises a whole sweep:

    touch .guardrails/approve-lr-sweep      # named after spec/runs/lr-sweep.json

While it exists and is inside `budget.approval_ttl_hours` (8h) it authorises **exactly what
the manifest describes** and nothing more:

- one `runpodctl pod create` (a second is blocked while a pod is recorded)
- one launch of each run whose status is `planned` — each run at most **once**
- `scripts/sweep_driver.sh`
- repo sync (`git pull`) on the pod
- `runpodctl pod delete`
- `scp` into `lightning_logs/` or the pod's repo

Anything off-manifest still stops and comes back to you as a question. Delete the file to
revoke mid-sweep; the agent is blocked at its next gated action.

## Still one-shot (each file authorises one action, then is deleted)

| File to `touch`        | Authorises                                    |
|------------------------|-----------------------------------------------|
| `approve-pod-install`  | one install/download command on the pod       |
| `approve-kill`         | one `kill`/`pkill`                            |
| `approve-scp`          | one `scp` outside `lightning_logs/`           |

## Never allowed, approval or not

Recursive deletes; any delete touching `/workspace`, `data/`, `lightning_logs/` or a
checkpoint; force-push; branch delete; `git reset --hard`; `git clean`; pod stop/reset/restart;
network-volume delete; and any write to the guardrail files themselves.

## State (written by the hook, after commands actually succeed)

`state/pod-created-at` starts the pod-hours budget clock; removed on successful delete.
`state/launched-<run>` marks a run as already launched. **Re-running a run after a crash is
your decision**: remove that marker to authorise a second attempt.

If a create is blocked by another layer after the hook passed, no state is written — the
bookkeeping happens in PostToolUse, only once the command has actually run.

Stale state, if something goes wrong:

    ls .guardrails/state/            # see what is recorded
    runpodctl pod list               # check against reality
