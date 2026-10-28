# Claude Code Guidelines

## Output style

- Lead with the answer or the decision needed. Default to under ~15 lines.
- Plans: goal, steps, how to verify. One table at most; no per-item rationale unless asked.
- Don't restate the request or recap tool output.
- Ask at most 3 questions at a time, each with a recommended default.
- Offer detail on request instead of including it.

## Pod and training sessions

Read `.claude/skills/experiment-facilitator/SKILL.md` first. Role: execution and faithful
recording only. Runs come from `spec/runs/*.json`, launch via `scripts/launch_run.py`, and
`.claude/hooks/guard.py` gates risky actions behind `.guardrails/` files only the user creates.
On a pod, read-only commands run freely; anything that writes, installs, kills or touches
git needs confirmation first.

## Working agreement (research authorship)

- **Paper prose is written by the user.** Claude coaches only: questions, feedback on the user's drafts, tracing every number to a log line. Claude never drafts paper text. The root `README.md` is edited only with the user's explicit permission; Claude may propose README text in chat but never writes it into the file.
- **Design decisions are the user's.** Ask before any choice that affects research design: data, preprocessing, splits, model, training config, metrics, explainability method, example/figure selection, budget allocation. A bug fix that changes model inputs or reported numbers counts as design-affecting: surface it, do not apply it silently.
- `.claude/hooks/guard.py` enforces part of this: `src/preprocess.py`, `dataset.py`, `model.py`, `train.py`, `explain.py` can be edited only against an approved record in `notes/decisions/` (see `.guardrails/README.md`).
- Existing AI-generated text (spec/data-setup.md, backlog entries, the old README, and the deleted spec/spec.md and spec/next-session-plan.md in git history) is not source material for the paper.
- Problem before solution: the problem statement is `notes/problem-definition.md`. When working on problem definition, ask questions; do not propose directions unless asked.
- Budget: $50 total GPU spend, tracked by the hook ledger (`python3 .claude/hooks/guard.py --ledger`) and spread across several manifests; the allocation is the user's.
