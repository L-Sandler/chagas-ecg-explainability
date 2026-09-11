# Claude Code Guidelines

## SSH into RunPod pods

When connecting to a RunPod pod via SSH, apply the following discipline:

**Auto-proceed (no confirmation needed):**
- Read-only commands: `ls`, `du`, `df`, `ps aux`, `date`, `nvidia-smi`, `cat`, `tail`, `head`, `wc`, status/progress checks

**Always confirm before running:**
- Commands that write, install, or delete: `apt-get`, `curl`, `wget`, `unzip`, `rm`, `mv`, `cp` (to new locations)
- Commands that modify the repo or environment: `git`, `uv`, `pip`
- Killing or stopping processes: `kill`, `pkill`
- Long-running training commands (`uv run python src/train.py ...`)

**Never run without explicit instruction:**
- `rm -rf` on any path
- Force-push or branch deletion
- Pod termination or reboot
- Any command that could destroy training checkpoints or downloaded data

**General principle:** On a cloud GPU pod, mistakes are expensive (data re-download, lost checkpoints, wasted GPU time). When in doubt, show the command and ask first.

## Facilitator sessions (training-run execution)

When the task is running training experiments, read `spec/fable-experiment-facilitator-prompt.md`
first. The role there is execution + faithful recording only; model/hyperparameter choices
and interpretation stay with the user. Runs come from `spec/runs/*.json`, launch via
`scripts/launch_run.py`, and the PreToolUse hook in `.claude/hooks/guard.py` gates risky
actions behind `.guardrails/approve-*` files that only the user creates.
