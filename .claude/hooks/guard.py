#!/usr/bin/env python3
"""PreToolUse + PostToolUse guardrail hook for facilitator sessions.

Design (v2, 2026-09-11):
  * ONE plan-level approval per sweep: `.guardrails/approve-<manifest-stem>`.
    While it exists and is within its TTL it authorises exactly what the manifest
    describes: one pod, one launch of each "planned" run, repo sync on the pod,
    the sweep driver, and pod deletion. Nothing else.
  * Approvals and state are recorded in PostToolUse, AFTER the command actually ran,
    so a command blocked by another layer no longer burns the approval.
  * Each planned run can launch at most once (state/launched-<run>); re-running after a
    crash is deliberate friction and requires the user to remove that marker.
  * Genuinely destructive commands are never allowed, approval or not.

Protocol: stdin JSON {hook_event_name, tool_name, tool_input, tool_response}.
PreToolUse: exit 2 + stderr = block. PostToolUse: exit 0 always (bookkeeping only).
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
DEFAULT_MANIFEST = "spec/runs/lr-sweep.json"
DEFAULT_TTL_HOURS = 8

PROTECTED_PATHS = (
    ".guardrails/", ".claude/", "spec/runs/",
    "scripts/launch_run.py", "scripts/sweep_driver.sh",
    "spec/fable-experiment-facilitator-prompt.md", "CLAUDE.md",
)

ALWAYS_BLOCK = [
    (r"\brm\s+(-[A-Za-z]*[rR][A-Za-z]*\s|--recursive)", "recursive delete"),
    (r"\brm\b[^\n]*(/workspace|lightning_logs|\bdata/|\.ckpt)", "delete of training data, checkpoints or the volume"),
    (r"\bgit\s+push\b[^\n]*(\s--force\b|\s-f\b|--force-with-lease)", "force push"),
    (r"\bgit\s+branch\s+(-D|--delete)", "branch deletion"),
    (r"\bgit\s+reset\s+--hard", "git reset --hard"),
    (r"\bgit\s+clean\b", "git clean"),
    (r"\brunpodctl\s+pod\s+(stop|reset|restart)\b", "pod stop/reset/restart (use delete instead)"),
    (r"\brunpodctl\s+(network-volume|volume)\s+(delete|rm)", "network volume deletion"),
    (r"\bshutdown\b|\breboot\b|\bpoweroff\b", "pod shutdown/reboot"),
]

# Writes that would modify guardrail config through the shell.
WRITE_TOKENS = r"(>>?|\bsed\s+-i|\btee\b|\bmv\b|\bcp\b|\btouch\b|\btruncate\b|\bdd\b)"
SHELL_PROTECTED = (r"\.guardrails", r"\.claude/", r"spec/runs", r"launch_run\.py",
                   r"sweep_driver\.sh", r"fable-experiment-facilitator-prompt", r"\bCLAUDE\.md")

ONE_SHOT = [  # (pattern, approval file, description, pod_only)
    (r"\b(kill|pkill|killall)\b", "approve-kill", "kill a process", False),
    (r"\b(uv\s+pip|uv\s+sync|pip\s+install|apt(-get)?\s+install|\bwget\s|\bcurl\s+-[^|]*-O)",
     "approve-pod-install", "install or download on the pod", True),
]

POD_SYNC = r"\bgit\s+(pull|fetch|checkout|merge|rebase|stash|switch|reset|commit|add)\b"


def block(msg):
    sys.stderr.write("GUARDRAIL BLOCKED: " + msg + "\n")
    sys.exit(2)


def manifest_path(cmd=""):
    m = re.search(r"--manifest\s+(\S+)", cmd)
    return ROOT / (m.group(1) if m else DEFAULT_MANIFEST)


def load_manifest(cmd=""):
    p = manifest_path(cmd)
    try:
        return json.loads(p.read_text()), p
    except Exception as e:
        block(f"cannot read run manifest {p}: {e}")


def sweep_approval(mp):
    """Path to the plan-level approval file for this manifest, or None if not valid."""
    f = GUARD / f"approve-{mp.stem}"
    if not f.exists():
        return None
    try:
        manifest = json.loads(mp.read_text())
        ttl = float(manifest.get("budget", {}).get("approval_ttl_hours", DEFAULT_TTL_HOURS))
    except Exception:
        ttl = DEFAULT_TTL_HOURS
    age_h = (time.time() - f.stat().st_mtime) / 3600.0
    if ttl and age_h > ttl:
        block(f"sweep approval {f.name} is {age_h:.1f}h old, past its {ttl}h TTL. "
              f"Ask the user to re-touch it if the sweep should continue.")
    return f


def require_sweep(mp, what):
    if sweep_approval(mp) is None:
        block(f"'{what}' is covered by the sweep plan but the plan is not approved. "
              f"Ask the user to run:  touch .guardrails/approve-{mp.stem}")


def pod_hours():
    f = STATE / "pod-created-at"
    if not f.exists():
        return 0.0
    try:
        return (time.time() - float(f.read_text().strip())) / 3600.0
    except ValueError:
        return 0.0


def check_budget(manifest, what):
    cap = float(manifest.get("budget", {}).get("max_pod_hours", 0) or 0)
    used = pod_hours()
    if cap and used > cap:
        block(f"pod-hours budget exceeded ({used:.2f}h used, cap {cap}h). Refusing to {what}. "
              f"Pull outputs, delete the pod, and report to the user.")


def one_shot(name, what):
    if not (GUARD / name).exists():
        block(f"'{what}' needs a one-shot approval. Ask the user to run:  "
              f"touch .guardrails/{name}   (then retry the same command)")


def run_name_from(cmd):
    m = re.search(r"(?:launch_run\.py)\s+([A-Za-z0-9._-]+)", cmd)
    return m.group(1) if m else None


# ----------------------------- PreToolUse -----------------------------

def pre_file(tool_input):
    p = str(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    try:
        rel = os.path.relpath(p, ROOT)
    except ValueError:
        rel = p
    for prot in PROTECTED_PATHS:
        if rel == prot.rstrip("/") or rel.startswith(prot):
            block(f"'{rel}' is user-owned guardrail config; the agent does not edit it.")


def pre_bash(cmd):
    for pat, why in ALWAYS_BLOCK:
        if re.search(pat, cmd):
            block(f"{why} is never run by the agent. The user runs it by hand if they want it.")

    if re.search(WRITE_TOKENS, cmd) and any(re.search(p, cmd) for p in SHELL_PROTECTED):
        block("shell write touching guardrail config (approvals, hook, manifest, launcher, prompt).")

    on_pod = bool(re.search(r"\bssh\b", cmd))

    # --- training launches ---
    if re.search(r"\bsrc/train\.py", cmd) and "launch_run.py" not in cmd and "sweep_driver" not in cmd:
        block("direct train.py launches are not allowed; use scripts/launch_run.py <run-name> "
              "or scripts/sweep_driver.sh (both manifest-driven).")

    if "sweep_driver.sh" in cmd:
        manifest, mp = load_manifest(cmd)
        require_sweep(mp, "run the sweep driver")
        check_budget(manifest, "start the sweep driver")
        return

    run = run_name_from(cmd)
    if run and "--dry-run" not in cmd:
        manifest, mp = load_manifest(cmd)
        runs = {r["name"]: r for r in manifest.get("runs", [])}
        if run not in runs:
            block(f"run '{run}' is not in {mp.name}; only pre-registered runs may launch.")
        if runs[run].get("status") != "planned":
            block(f"run '{run}' has status '{runs[run].get('status')}', not 'planned'.")
        if (STATE / f"launched-{run}").exists():
            block(f"run '{run}' has already been launched once this sweep. Re-running is a "
                  f"deliberate decision: ask the user to remove .guardrails/state/launched-{run}.")
        check_budget(manifest, f"launch {run}")
        require_sweep(mp, f"launch {run}")
        return

    # --- pod lifecycle ---
    if re.search(r"\brunpodctl\s+pod\s+create\b", cmd):
        manifest, mp = load_manifest()
        if (STATE / "pod-created-at").exists():
            block("a pod is already recorded for this sweep (max_concurrent_pods=1). Delete it "
                  "first, or if that record is stale ask the user to remove "
                  ".guardrails/state/pod-created-at")
        check_budget(manifest, "create a pod")
        require_sweep(mp, "provision a pod")
        return

    if re.search(r"\brunpodctl\s+pod\s+delete\b|\brunpodctl\s+remove\s+pod\b", cmd):
        _, mp = load_manifest()
        require_sweep(mp, "delete the pod")
        return

    # --- one-shot gated actions ---
    for pat, appr, what, pod_only in ONE_SHOT:
        if re.search(pat, cmd) and (on_pod or not pod_only):
            one_shot(appr, what)
            return

    if on_pod and re.search(POD_SYNC, cmd):
        _, mp = load_manifest()
        require_sweep(mp, "sync the repo on the pod")
        return

    if re.search(r"\bscp\b", cmd):
        dest = cmd.split()[-1] if cmd.split() else ""
        ok = ("lightning_logs" in dest or "/scratchpad" in dest
              or re.search(r":\S*chagas-ecg-explainability", dest))
        if not ok:
            one_shot("approve-scp", "scp to a destination outside lightning_logs/ or the pod repo")


# ----------------------------- PostToolUse -----------------------------

def succeeded(resp):
    """Best-effort success check. Defaults to True, which is the safe direction for both
    recording pod state (blocks more) and consuming approvals (user re-touches)."""
    if not isinstance(resp, dict):
        return True
    for k in ("exitCode", "exit_code", "returncode", "code"):
        if k in resp and isinstance(resp[k], int):
            return resp[k] == 0
    if resp.get("interrupted"):
        return False
    blob = " ".join(str(resp.get(k, "")) for k in ("stderr", "stdout", "error"))
    if re.search(r"GUARDRAIL BLOCKED|Permission for this action was denied|was denied by", blob):
        return False
    return True


def post_bash(cmd, resp):
    STATE.mkdir(parents=True, exist_ok=True)
    ok = succeeded(resp)
    out = " ".join(str(resp.get(k, "")) for k in ("stdout", "stderr")) if isinstance(resp, dict) else ""

    if re.search(r"\brunpodctl\s+pod\s+create\b", cmd) and ok:
        if not re.search(r"error|failed|no longer any instances", out, re.I):
            (STATE / "pod-created-at").write_text(str(time.time()))
            m = re.search(r"\b([a-z0-9]{12,16})\b", out)
            if m:
                (STATE / "pod-id").write_text(m.group(1))

    if re.search(r"\brunpodctl\s+(pod\s+delete|remove\s+pod)\b", cmd) and ok:
        for f in ("pod-created-at", "pod-id"):
            try:
                (STATE / f).unlink()
            except FileNotFoundError:
                pass

    run = run_name_from(cmd)
    if run and "--dry-run" not in cmd and ok:
        (STATE / f"launched-{run}").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    if ok:
        for pat, appr, _what, pod_only in ONE_SHOT:
            if re.search(pat, cmd) and (bool(re.search(r"\bssh\b", cmd)) or not pod_only):
                try:
                    (GUARD / appr).unlink()
                except FileNotFoundError:
                    pass
        if re.search(r"\bscp\b", cmd):
            try:
                (GUARD / "approve-scp").unlink()
            except FileNotFoundError:
                pass


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    event = payload.get("hook_event_name", "PreToolUse")
    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input", {}) or {}

    if event == "PostToolUse":
        if tool == "Bash":
            try:
                post_bash(str(ti.get("command", "")), payload.get("tool_response", {}))
            except Exception:
                pass
        sys.exit(0)

    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        pre_file(ti)
    elif tool == "Bash":
        pre_bash(str(ti.get("command", "")))
    sys.exit(0)


if __name__ == "__main__":
    main()
