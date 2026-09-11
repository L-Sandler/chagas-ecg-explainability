# .guardrails — user-only approval files

The PreToolUse hook (`.claude/hooks/guard.py`) gates risky actions on files in this directory.
**Only the user creates these**, from a normal terminal. Each file is consumed (deleted) the
first time it authorises an action, so one file = one action.

| File to `touch`                        | Authorises                                   |
|----------------------------------------|----------------------------------------------|
| `approve-pod-create`                   | one `runpodctl pod create`                    |
| `approve-launch-<run-name>`            | one launch of that manifest run               |
| `approve-pod-delete`                   | one `runpodctl pod delete`                    |
| `approve-pod-sync`                     | one git pull/checkout/etc. on the pod         |
| `approve-pod-install`                  | one install/download command on the pod       |
| `approve-kill`                         | one kill/pkill                                |
| `approve-scp`                          | one scp to somewhere other than lightning_logs/ |

Pre-authorise a whole session in one go, e.g.:

    touch .guardrails/approve-pod-create .guardrails/approve-pod-sync \
          .guardrails/approve-launch-full-code15-lr3e-4 \
          .guardrails/approve-launch-full-code15-lr3e-3 \
          .guardrails/approve-pod-delete

`state/pod-created-at` is written by the hook when a pod is created and removed on delete;
it drives the `max_pod_hours` budget check from `spec/runs/lr-sweep.json`.

Never gated, always blocked: `rm -r`/`rm -rf`, any `rm` touching `/workspace`, `data/`,
`lightning_logs/` or `.ckpt`; force-push; branch delete; `git reset --hard`; `git clean`;
pod stop/reset/restart; volume delete; and any command that touches the hook/settings files.
