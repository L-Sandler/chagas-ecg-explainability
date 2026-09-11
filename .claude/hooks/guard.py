#!/usr/bin/env python3
"""PreToolUse guardrail hook for facilitator sessions.

Enforces the CLAUDE.md RunPod discipline mechanically instead of by prompt text:
  * always-blocked commands (rm -rf, force-push, deleting /workspace or data/, ...)
  * one-shot approval files under .guardrails/ that ONLY the user creates
  * training launches must go through scripts/launch_run.py with a manifest run name
  * a pod-hours budget read from the run manifest

Protocol: stdin JSON {tool_name, tool_input}; exit 2 + stderr message = block.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
GUARD = ROOT / ".guardrails"
STATE = GUARD / "state"
MANIFEST = ROOT / "spec" / "runs" / "lr-sweep.json"

# Files the agent must not edit through any tool. The user edits these by hand.
PROTECTED_PATHS = (
    ".guardrails/",
    ".claude/",
    "spec/runs/",
    "scripts/launch_run.py",
    "spec/fable-experiment-facilitator-prompt.md",
    "CLAUDE.md",
)

ALWAYS_BLOCK = [
    (r"\brm\s+(-[A-Za-z]*[rR][A-Za-z]*\s|--recursive)", "rm -r / rm -rf is never run by the agent"),
    (r"\brm\b[^\n]*(/workspace|lightning_logs|\bdata/|\.ckpt)", "rm on training data / checkpoints / volume"),
    (r"\bgit\s+push\b[^\n]*(\s--force\b|\s-f\b|--force-with-lease)", "force push"),
    (r"\bgit\s+branch\s+(-D|--delete)", "branch deletion"),
    (r"\bgit\s+reset\s+--hard", "git reset --hard"),
    (r"\bgit\s+clean\b", "git clean"),
    (r"\brunpodctl\s+pod\s+(stop|reset|restart)\b", "pod stop/reset/restart — use delete with approval instead"),
    (r"\brunpodctl\s+(network-volume|volume)\s+(delete|rm)", "network volume deletion"),
    (r"\.guardrails|\.claude/hooks|\.claude/settings", "touching the guardrail machinery itself"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b", "pod shutdown/reboot"),
]

# Writes to protected files via shell tricks (sed -i, redirects, tee, mv, cp, touch ...).
WRITE_TOKENS = r"(>|>>|\bsed\s+-i|\btee\b|\bmv\b|\bcp\b|\btouch\b|\btruncate\b|\bcat\s*<<|\bpython[3]?\s+-\s*<<)"
SHELL_PROTECTED = (r"spec/runs", r"scripts/launch_run\.py", r"fable-experiment-facilitator-prompt", r"\bCLAUDE\.md")

# Gated actions: (regex, approval-file-name, human hint)
GATED = [
    (r"\brunpodctl\s+pod\s+create\b", "approve-pod-create", "provision a pod"),
    (r"\brunpodctl\s+pod\s+delete\b", "approve-pod-delete", "delete the pod"),
    (r"\b(kill|pkill|killall)\b", "approve-kill", "kill a process"),
]
# Gated only when they run on the pod (inside an ssh command).
GATED_ON_POD = [
    (r"\bgit\s+(pull|fetch|checkout|merge|rebase|stash|reset|commit|add|switch)\b", "approve-pod-sync", "change the repo on the pod"),
    (r"\b(uv\s+pip|uv\s+sync|pip\s+install|apt(-get)?\s|wget\s|curl\s)", "approve-pod-install", "install/download on the pod"),
]


def block(msg: str) -> None:
    sys.stderr.write("GUARDRAIL BLOCKED: " + msg + "\n")
    sys.exit(2)


def approval(name: str) -> Path:
    return GUARD / name


def consume(name: str) -> None:
    try:
        approval(name).unlink()
    except FileNotFoundError:
        pass


def require(name: str, what: str) -> None:
    if not approval(name).exists():
        block(
            f"'{what}' needs a one-shot approval from the user. "
            f"Ask them to run:  touch .guardrails/{name}   (then retry the same command)."
        )


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception as e:  # noqa: BLE001
        block(f"cannot read run manifest {MANIFEST}: {e}")
    return {}


def pod_hours_elapsed() -> float:
    f = STATE / "pod-created-at"
    if not f.exists():
        return 0.0
    try:
        return (time.time() - float(f.read_text().strip())) / 3600.0
    except ValueError:
        return 0.0


def check_budget(manifest: dict, what: str) -> None:
    cap = float(manifest.get("budget", {}).get("max_pod_hours", 0) or 0)
    used = pod_hours_elapsed()
    if cap and used > cap:
        block(
            f"pod-hours budget exceeded ({used:.2f}h used, cap {cap}h from manifest). "
            f"Refusing to {what}. Report this to the user; only pod delete is allowed now."
        )


def handle_file_tool(tool_input: dict) -> None:
    p = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    try:
        rel = os.path.relpath(p, ROOT)
    except ValueError:
        rel = p
    for prot in PROTECTED_PATHS:
        if rel == prot.rstrip("/") or rel.startswith(prot):
            block(f"'{rel}' is user-owned guardrail config; the agent does not edit it.")


def handle_bash(cmd: str) -> None:
    for pat, why in ALWAYS_BLOCK:
        if re.search(pat, cmd):
            block(f"{why}. This is never run by the agent; the user runs it by hand if wanted.")

    if re.search(WRITE_TOKENS, cmd):
        for pat in SHELL_PROTECTED:
            if re.search(pat, cmd):
                block("shell write to a user-owned guardrail file (manifest, launcher, prompt, CLAUDE.md).")

    on_pod = bool(re.search(r"\bssh\b", cmd))

    # --- training launches ---
    if re.search(r"train\.py", cmd) and "launch_run.py" not in cmd:
        block("direct train.py launches are not allowed; use  scripts/launch_run.py <run-name>  (manifest-driven).")
    m = re.search(r"launch_run\.py\s+(?:--\S+\s+\S+\s+)*([A-Za-z0-9._-]+)", cmd)
    if m and "--dry-run" not in cmd:
        name = m.group(1)
        manifest = load_manifest()
        runs = {r["name"]: r for r in manifest.get("runs", [])}
        if name not in runs:
            block(f"run '{name}' is not in {MANIFEST.relative_to(ROOT)}; only pre-registered runs may launch.")
        if runs[name].get("status") != "planned":
            block(f"run '{name}' has status '{runs[name].get('status')}', not 'planned'.")
        check_budget(manifest, f"launch {name}")
        require(f"approve-launch-{name}", f"launch {name}")
        consume(f"approve-launch-{name}")
        return

    for pat, appr, what in GATED:
        if re.search(pat, cmd):
            if appr == "approve-pod-create":
                manifest = load_manifest()
                if (STATE / "pod-created-at").exists():
                    block("a pod is already recorded as created this session (max_concurrent_pods=1). Delete it first.")
                check_budget(manifest, "create another pod")
                require(appr, what)
                consume(appr)
                STATE.mkdir(parents=True, exist_ok=True)
                (STATE / "pod-created-at").write_text(str(time.time()))
                return
            if appr == "approve-pod-delete":
                require(appr, what)
                consume(appr)
                try:
                    (STATE / "pod-created-at").unlink()
                except FileNotFoundError:
                    pass
                return
            require(appr, what)
            consume(appr)
            return

    if on_pod:
        for pat, appr, what in GATED_ON_POD:
            if re.search(pat, cmd):
                require(appr, what)
                consume(appr)
                return

    # scp: only pulling run outputs back into lightning_logs/ (gitignored) is unattended.
    if re.search(r"\bscp\b", cmd):
        parts = cmd.split()
        dest = parts[-1] if parts else ""
        if ":" in dest or not (dest.startswith("lightning_logs") or "/lightning_logs" in dest or "/scratchpad" in dest):
            require("approve-scp", "scp to a destination outside lightning_logs/")
            consume("approve-scp")


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        sys.exit(0)
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        handle_file_tool(tool_input)
    elif tool == "Bash":
        handle_bash(str(tool_input.get("command", "")))
    sys.exit(0)


if __name__ == "__main__":
    main()
