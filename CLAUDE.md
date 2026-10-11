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

When the task is running training experiments, read `spec/experiment-facilitator-prompt.md`
first. The role there is execution + faithful recording only; model/hyperparameter choices
and interpretation stay with the user. Runs come from `spec/runs/*.json`, launch via
`scripts/launch_run.py`, and the PreToolUse hook in `.claude/hooks/guard.py` gates risky
actions behind `.guardrails/approve-*` files that only the user creates.

## Working agreement (research authorship)

- **Paper prose is written by the user.** Claude coaches only: questions, feedback on the user's drafts, tracing every number to a log line. Claude never drafts paper text. The root `README.md` is edited only with the user's explicit permission; Claude may propose README text in chat but never writes it into the file.
- **Design decisions are the user's.** Ask before any choice that affects research design: data, preprocessing, splits, model, training config, metrics, explainability method, example/figure selection, budget allocation. A bug fix that changes model inputs or reported numbers counts as design-affecting: surface it, do not apply it silently.
- `.claude/hooks/guard.py` enforces part of this: `src/preprocess.py`, `dataset.py`, `model.py`, `train.py`, `explain.py` can be edited only against an approved record in `notes/decisions/` (see `.guardrails/README.md`).
- Existing AI-generated text (`spec/spec.md`, `spec/next-session-plan.md`, backlog entries, the old README) is not source material for the paper.
- Problem before solution: the problem statement is `notes/problem-definition.md`. When working on problem definition, ask questions; do not propose directions unless asked.
- Budget: $50 total GPU spend, tracked by the hook ledger (`python3 .claude/hooks/guard.py --ledger`) and spread across several manifests; the allocation is the user's.
