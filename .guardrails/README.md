# .guardrails: user-only approval and budget files

Only you create or edit files here, from a normal terminal (not through the agent).

## Who enforces what

| Rule | Enforced by |
|---|---|
| Config, manifests, launcher, CLAUDE.md, root README are not agent-editable | `permissions.deny` (file tools) + sandbox `denyWrite` (shell and Python) |
| `src/{preprocess,dataset,model,train,explain}.py` | Edit/Write only against an approved record in `notes/decisions/` (hook); shell writes blocked by the sandbox |
| Decision records | Agent may write `status: proposed` only (hook); sandbox blocks shell writes |
| Manifest, launch-once, budget, pod lifecycle, one-shots | `.claude/hooks/guard.py` (stateful) |
| Destructive commands | hook blocklist, matched on command position, not on any mention |

The hook fails closed: an internal error or a crash blocks the call.

## Files you own

| File | Purpose |
|---|---|
| `active-manifest` | The `spec/runs/*.json` that pod create/delete/sync and launches gate on. |
| `budget.json` | `global_cap_usd` and `spent_before_ledger_usd`. Set the cap from the real RunPod balance, not a round number. |
| `approve-<manifest-stem>` | Plan-level approval (8 h TTL): one pod, one launch per `planned` run, sweep driver, pod sync, pod delete, `scp` into `lightning_logs/` or `/workspace/`. Delete it to revoke. |

## One-shot approvals (consumed when the command succeeds)

| `touch` | Authorises |
|---|---|
| `unlock-final-test` | one command or script that opens the locked-final split (notes/decisions/0004) |
| `approve-pod-install` | one install/download on the pod |
| `approve-kill` | one `kill`/`pkill` |
| `approve-scp` | one `scp` outside `lightning_logs/` and `/workspace/` |

## Budget and spend tracking

`global_cap_usd` is your GPU budget for this project (not the RunPod balance, which is shared
with other uses). Pod create, launch and sweep start are refused when it, or the manifest's
`cap_usd`, is reached.

`python3 .claude/hooks/guard.py --ledger` prints every pod (date, GPU, hours x $/h, the runs it
carried), subtotals per manifest, total spend and what remains of the budget. The hourly rate
is the one RunPod reports for the pod (captured from `create`/`pod get` output); failing that,
the manifest's price for that GPU; failing that, the highest listed price. A pod that
self-terminated is billed at most `max_pod_hours`. Not tracked: network volume storage and
pods created outside this hook; reconcile with the billing page and edit
`state/ledger.jsonl` or `spent_before_ledger_usd`.

## State (written by the hook after a command succeeds)

`state/pod-created-at`, `pod-manifest`, `pod-id`: the pod being billed. This is now recorded for
`scripts/create_pod.sh` as well as `runpodctl pod create`. `state/launched-<run>`: remove it to
allow a rerun. `state/hook-errors.log`: internal hook errors (post-run bookkeeping failures).

## Known limits

- One-shot approvals are not tied to a session: any session's matching command can use one.
- The locked-split check is textual. The strong fix is for `train.py` to require the approval
  file itself (a design change; needs a decision record).
- `ssh`, `scp`, `runpodctl`, `create_pod.sh` and git network commands run outside the sandbox, so
  only the hook gates them.
- `git checkout/merge/rebase` that rewrites a protected file fails inside the sandbox. Do those by hand.
