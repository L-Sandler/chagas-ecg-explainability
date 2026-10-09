#!/usr/bin/env python3
"""PreToolUse + PostToolUse guardrail hook (v3, draft 2026-10-09).

v2 gated GPU execution and spend for one hardwired sweep. v3 adds:
  * G1  invocation-aware matching: only real launches of train.py / launch_run.py /
        sweep_driver.sh are gated, not any command that merely mentions the filename.
        Local `--fast` CPU smoke runs of train.py are allowed (no ssh).
  * G2  active manifest: `.guardrails/active-manifest` (user-owned) names the manifest that
        pod create/delete/sync/launch gate on. `--manifest X` or a positional manifest
        argument to sweep_driver.sh still wins.
  * G4  design-decision gate: research-relevant source files can be edited by the agent only
        if an approved record in notes/decisions/ lists the file under `applies_to:`.
        notes/decisions/ is user-write-only, so the agent cannot approve its own work.
  * G5  global budget: `.guardrails/budget.json` (user-owned) holds the global USD cap and
        the spend before this ledger existed. The hook keeps `.guardrails/state/ledger.jsonl`
        (one line per pod lifetime). Pod create / run launch / sweep start are blocked if the
        global cap OR the manifest's own `budget.cap_usd` is reached. Cost is an ESTIMATE:
        pod-hours x the highest listed GPU price (errs high).

Unchanged from v2: plan-level approval file per manifest with TTL, post-success bookkeeping,
one-shot approvals, ALWAYS_BLOCK list, protected guardrail files.

Protocol: stdin JSON {hook_event_name, tool_name, tool_input, tool_response}.
PreToolUse: exit 2 + stderr = block. PostToolUse: exit 0 always (bookkeeping only).
`python3 guard.py --ledger` prints current spend (run it by hand, no stdin needed).
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
LEDGER = STATE / "ledger.jsonl"
DECISIONS = ROOT / "notes" / "decisions"
DEFAULT_MANIFEST = "spec/runs/lr-sweep.json"  # only used if .guardrails/active-manifest is absent
DEFAULT_TTL_HOURS = 8
DEFAULT_ASSUMED_USD_PER_HR = 1.0  # used when a manifest lists no usable GPU prices

PROTECTED_PATHS = (
    ".guardrails/", ".claude/", "spec/runs/",  # notes/decisions/ has its own rule (propose-only)
    "scripts/launch_run.py", "scripts/sweep_driver.sh",
    "spec/fable-experiment-facilitator-prompt.md", "CLAUDE.md",
    "README.md",  # root README only (rel == "README.md"): edited only with the user's permission
)

# Research-relevant code. Editing these is a design-affecting act (G4).
GATED_PATHS = (
    "src/preprocess.py", "src/dataset.py", "src/model.py", "src/train.py", "src/explain.py",
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
WRITE_TOKENS = r"(\bsed\s+-i|\btee\b|\bmv\b|\bcp\b|\btouch\b|\btruncate\b|\bdd\b)"
SHELL_PROTECTED = (r"\.guardrails", r"\.claude/", r"spec/runs", r"notes/decisions", r"launch_run\.py",
                   r"sweep_driver\.sh", r"fable-experiment-facilitator-prompt", r"\bCLAUDE\.md",
                   r"(?<![\w/.-])README\.md")  # root README only; sub-dir READMEs unaffected
# Redirections that are not file writes (2>&1, >/dev/null) must not trip the write check.
NOISE_REDIRECTS = r"(\d*>&\d+|\d*>>?\s*/dev/null)"

UNLOCK_PATTERN = r"--unlock-final\b|\bunlock_final\s*=\s*True"

ONE_SHOT = [  # (pattern, approval file, description, pod_only)
    # notes/decisions/0004: the locked-final split is read ONCE, at the very end. The user
    # creates this file for exactly that evaluation; it is consumed when the command succeeds.
    (UNLOCK_PATTERN, "unlock-final-test", "open the locked-final split (final evaluation only)", False),
    (r"\b(kill|pkill|killall)\b", "approve-kill", "kill a process", False),
    (r"\b(uv\s+pip|uv\s+sync|pip\s+install|apt(-get)?\s+install|\bwget\s|\bcurl\s+-[^|]*-O)",
     "approve-pod-install", "install or download on the pod", True),
]

POD_SYNC = r"\bgit\s+(pull|fetch|checkout|merge|rebase|stash|switch|reset|commit|add)\b"

# --- invocation patterns (G1): a bare mention of a filename is not an invocation ---
TRAIN_INVOKE = r"(\bpython[0-9.]*|\buv\s+run)\b[^|;&\n]*\bsrc/train\.py"
LAUNCH_INVOKE = r"(?:\bpython[0-9.]*\s+|\./)(?:\S*/)?scripts/launch_run\.py\s+([A-Za-z0-9][A-Za-z0-9._-]*)"
SWEEP_INVOKE = r"(?:\./|\b(?:bash|sh|setsid|nohup)\s+(?:-\S+\s+)*)(?:\S*/)?scripts/sweep_driver\.sh"


def block(msg):
    sys.stderr.write("GUARDRAIL BLOCKED: " + msg + "\n")
    sys.exit(2)


def read_json(p):
    return json.loads(Path(p).read_text())


# ----------------------------- manifest / approvals -----------------------------

def active_manifest_rel():
    f = GUARD / "active-manifest"
    try:
        v = f.read_text().strip()
        if v:
            return v
    except OSError:
        pass
    return DEFAULT_MANIFEST


def manifest_path(cmd=""):
    m = re.search(r"--manifest\s+(\S+)", cmd) or re.search(r"sweep_driver\.sh\s+(\S+\.json)\b", cmd)
    return ROOT / (m.group(1) if m else active_manifest_rel())


def load_manifest(cmd=""):
    p = manifest_path(cmd)
    try:
        return read_json(p), p
    except Exception as e:
        block(f"cannot read run manifest {p}: {e}")


def sweep_approval(mp):
    """Path to the plan-level approval file for this manifest, or None if not valid."""
    f = GUARD / f"approve-{mp.stem}"
    if not f.exists():
        return None
    try:
        ttl = float(read_json(mp).get("budget", {}).get("approval_ttl_hours", DEFAULT_TTL_HOURS))
    except Exception:
        ttl = DEFAULT_TTL_HOURS
    age_h = (time.time() - f.stat().st_mtime) / 3600.0
    if ttl and age_h > ttl:
        block(f"sweep approval {f.name} is {age_h:.1f}h old, past its {ttl}h TTL. "
              f"Ask the user to re-touch it if the sweep should continue.")
    return f


def require_sweep(mp, what):
    if sweep_approval(mp) is None:
        block(f"'{what}' is covered by the manifest {mp.name} but it is not approved. "
              f"Ask the user to run:  touch .guardrails/approve-{mp.stem}")


def one_shot(name, what):
    if not (GUARD / name).exists():
        block(f"'{what}' needs a one-shot approval. Ask the user to run:  "
              f"touch .guardrails/{name}   (then retry the same command)")


# ----------------------------- budget (G5) -----------------------------

def pod_hours():
    f = STATE / "pod-created-at"
    if not f.exists():
        return 0.0
    try:
        return (time.time() - float(f.read_text().strip())) / 3600.0
    except ValueError:
        return 0.0


def price_per_hr(manifest):
    """Highest listed GPU price (or the assumed default): deliberately errs high."""
    prices = manifest.get("budget", {}).get("gpu_price_usd_per_hr") or {}
    vals = [float(v) for v in prices.values() if isinstance(v, (int, float))]
    assumed = float(manifest.get("budget", {}).get("assumed_usd_per_hr_unlisted", DEFAULT_ASSUMED_USD_PER_HR))
    return max(vals + [assumed])


def ledger_entries():
    out = []
    try:
        for line in LEDGER.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return out


def running_pod():
    """(estimated USD so far, manifest stem) for the currently recorded pod, else (0, None)."""
    f = STATE / "pod-created-at"
    if not f.exists():
        return 0.0, None
    try:
        rel = (STATE / "pod-manifest").read_text().strip()
    except OSError:
        rel = active_manifest_rel()
    try:
        price = price_per_hr(read_json(ROOT / rel))
    except Exception:
        price = DEFAULT_ASSUMED_USD_PER_HR
    return pod_hours() * price, Path(rel).stem


def spend(stem=None):
    """Estimated USD spent through this ledger, optionally for one manifest stem."""
    total = sum(float(e.get("usd_est", 0)) for e in ledger_entries()
                if stem is None or e.get("manifest") == stem)
    run_usd, run_stem = running_pod()
    if stem is None or run_stem == stem:
        total += run_usd
    return total


def load_budget():
    try:
        b = read_json(GUARD / "budget.json")
        cap = float(b["global_cap_usd"])
        before = b["spent_before_ledger_usd"]
        if not isinstance(before, (int, float)):
            raise ValueError("spent_before_ledger_usd must be a number")
        return cap, float(before)
    except Exception as e:
        block(f"cannot read .guardrails/budget.json ({e}). The user must create it with "
              f"global_cap_usd and spent_before_ledger_usd (from RunPod billing).")


def check_budget(manifest, mp, what):
    cap_h = float(manifest.get("budget", {}).get("max_pod_hours", 0) or 0)
    used_h = pod_hours()
    if cap_h and used_h > cap_h:
        block(f"pod-hours budget exceeded ({used_h:.2f}h used, cap {cap_h}h). Refusing to {what}. "
              f"Pull outputs, delete the pod, and report to the user.")

    g_cap, before = load_budget()
    g_spent = before + spend()
    if g_spent >= g_cap:
        block(f"global budget reached (${g_spent:.2f} est. of ${g_cap:.2f}). Refusing to {what}. "
              f"Report to the user; only they can raise .guardrails/budget.json.")

    b = manifest.get("budget", {})
    m_cap = float(b.get("cap_usd", b.get("wallet_hard_cap_usd", 0)) or 0)
    m_spent = spend(mp.stem)
    if m_cap and m_spent >= m_cap:
        block(f"manifest budget reached for {mp.name} (${m_spent:.2f} est. of ${m_cap:.2f}). "
              f"Refusing to {what}.")


def ledger_summary():
    g_cap, before = None, None
    try:
        b = read_json(GUARD / "budget.json")
        g_cap, before = b.get("global_cap_usd"), b.get("spent_before_ledger_usd")
    except Exception:
        pass
    print(f"global cap: {g_cap}  spent before ledger: {before}")
    for e in ledger_entries():
        print(f"  {e.get('manifest')}: {e.get('hours', 0):.2f}h ~${e.get('usd_est', 0):.2f} "
              f"({e.get('closed_by')})")
    run_usd, run_stem = running_pod()
    if run_stem:
        print(f"  RUNNING pod for {run_stem}: ~${run_usd:.2f} so far")
    print(f"ledger total (est.): ${spend():.2f}")


# ----------------------------- design-decision gate (G4) -----------------------------

def approved_decision_paths():
    """{repo-relative path: record filename} for every approved decision record."""
    out = {}
    if not DECISIONS.is_dir():
        return out
    for f in sorted(DECISIONS.glob("*.md")):
        if f.name.upper().startswith("TEMPLATE"):
            continue
        try:
            text = f.read_text()
        except OSError:
            continue
        head = re.match(r"---\s*\n(.*?)\n---", text, re.S)
        if not head:
            continue
        fm = head.group(1)
        if not re.search(r"^status:\s*approved\s*$", fm, re.M):
            continue
        m = re.search(r"^applies_to:\s*(.+)$", fm, re.M)
        if m:
            for p in m.group(1).split(","):
                out[p.strip()] = f.name
    return out


def require_decision(rel):
    if rel in GATED_PATHS and rel not in approved_decision_paths():
        block(f"'{rel}' is research-relevant code; editing it is a design decision. No approved "
              f"record in notes/decisions/ lists it under applies_to. Ask the user to write/approve "
              f"one (status: approved), or ask which decision this change belongs to.")


# ----------------------------- PreToolUse -----------------------------

def rel_of(p):
    try:
        return os.path.relpath(str(p), ROOT)
    except ValueError:
        return str(p)


def record_status(text):
    head = re.match(r"---\s*\n(.*?)\n---", text or "", re.S)
    m = head and re.search(r"^status:\s*(\w+)", head.group(1), re.M)
    return m.group(1).lower() if m else None


def check_decision_write(rel, tool, ti):
    """The agent may DRAFT records (status: proposed) but never approve, supersede, reject or
    rewrite one. Approval stays a one-line edit by the user."""
    path = ROOT / rel
    current = path.read_text() if path.exists() else None
    if tool == "Write":
        result = ti.get("content", "")
    elif tool in ("Edit", "MultiEdit"):
        if current is None:
            block(f"cannot edit '{rel}': it does not exist.")
        edits = ti.get("edits") if tool == "MultiEdit" else [ti]
        result = current
        for e in edits or []:
            old, new = e.get("old_string", ""), e.get("new_string", "")
            result = result.replace(old, new) if e.get("replace_all") else result.replace(old, new, 1)
    else:
        block(f"'{rel}' is user-owned; {tool} is not allowed on decision records.")
    if current is not None and record_status(current) != "proposed":
        block(f"'{rel}' has status '{record_status(current)}'. Only the user changes a record that "
              f"is not 'proposed'; write a new record instead.")
    if record_status(result) != "proposed":
        block("agents may only write decision records with 'status: proposed'. The user flips a "
              "record to 'approved'.")


def pre_file(tool, tool_input):
    rel = rel_of(tool_input.get("file_path") or tool_input.get("notebook_path") or "")
    if rel.startswith("notes/decisions/"):
        check_decision_write(rel, tool, tool_input)
        return
    for prot in PROTECTED_PATHS:
        if rel == prot.rstrip("/") or rel.startswith(prot):
            block(f"'{rel}' is user-owned guardrail/decision config; the agent does not edit it.")
    require_decision(rel)
    # A script that opens the locked split would dodge the command-line check, so inspect
    # the content being written too. The gated source files (which define the flag) are
    # already covered by the decision gate above.
    if rel not in GATED_PATHS and rel.endswith((".py", ".sh", ".ipynb")):
        text = " ".join(str(v) for k, v in tool_input.items()
                        if k in ("content", "new_string")) + " ".join(
            str(e.get("new_string", "")) for e in tool_input.get("edits", []) or [])
        if re.search(UNLOCK_PATTERN, text):
            one_shot("unlock-final-test", f"write code that opens the locked-final split ({rel})")


def pre_bash(cmd):
    for pat, why in ALWAYS_BLOCK:
        if re.search(pat, cmd):
            block(f"{why} is never run by the agent. The user runs it by hand if they want it.")

    writeable = re.sub(NOISE_REDIRECTS, " ", cmd)
    # A redirect only counts if its TARGET is protected (v2 also flagged `> sweep_driver.log`
    # merely because the command mentioned sweep_driver.sh, which blocked the sweep start).
    redirect_hit = any(re.search(rf">>?\s*\S*{p}", writeable) for p in SHELL_PROTECTED)
    token_hit = re.search(WRITE_TOKENS, writeable) and any(re.search(p, cmd) for p in SHELL_PROTECTED)
    if redirect_hit or token_hit:
        block("shell write touching guardrail config (approvals, hook, manifest, decision records, "
              "launcher, prompt).")

    # G4 via shell: only the obvious file-writing forms aimed at a gated file.
    approved = approved_decision_paths()
    for g in GATED_PATHS:
        if g in approved:
            continue
        gp = re.escape(g)
        if re.search(rf"(>>?\s*\S*{gp}|\bsed\s+-i[^\n]*{gp}|\btee\b[^\n]*{gp}|"
                     rf"\b(mv|cp)\b[^\n]*{gp})", writeable):
            block(f"shell write to '{g}': research-relevant code needs an approved record in "
                  f"notes/decisions/ (applies_to: {g}).")

    on_pod = bool(re.search(r"\bssh\b", cmd))

    # --- training launches (G1: invocation-aware) ---
    if re.search(TRAIN_INVOKE, cmd) and not re.search(LAUNCH_INVOKE, cmd) \
            and not re.search(SWEEP_INVOKE, cmd):
        if "--fast" in cmd and not on_pod:
            pass  # local CPU smoke run: allowed
        else:
            block("direct train.py launches are not allowed (only local `--fast` smoke runs); "
                  "use scripts/launch_run.py <run-name> or scripts/sweep_driver.sh "
                  "(both manifest-driven).")

    if re.search(SWEEP_INVOKE, cmd):
        manifest, mp = load_manifest(cmd)
        require_sweep(mp, "run the sweep driver")
        check_budget(manifest, mp, "start the sweep driver")
        return

    lm = re.search(LAUNCH_INVOKE, cmd)
    if lm and "--dry-run" not in cmd:
        run = lm.group(1)
        manifest, mp = load_manifest(cmd)
        runs = {r["name"]: r for r in manifest.get("runs", [])}
        if run not in runs:
            block(f"run '{run}' is not in {mp.name}; only pre-registered runs may launch.")
        if runs[run].get("status") != "planned":
            block(f"run '{run}' has status '{runs[run].get('status')}', not 'planned'.")
        if (STATE / f"launched-{run}").exists():
            block(f"run '{run}' has already been launched once. Re-running is a deliberate "
                  f"decision: ask the user to remove .guardrails/state/launched-{run}.")
        check_budget(manifest, mp, f"launch {run}")
        require_sweep(mp, f"launch {run}")
        return

    # --- pod lifecycle ---
    if re.search(r"\brunpodctl\s+pod\s+create\b|\bcreate_pod\.sh\b", cmd):
        manifest, mp = load_manifest(cmd)
        if (STATE / "pod-created-at").exists():
            block("a pod is already recorded (max_concurrent_pods=1). Delete it first, or if that "
                  "record is stale ask the user to remove .guardrails/state/pod-created-at")
        check_budget(manifest, mp, "create a pod")
        require_sweep(mp, "provision a pod")
        return

    if re.search(r"\brunpodctl\s+pod\s+delete\b|\brunpodctl\s+remove\s+pod\b", cmd):
        _, mp = load_manifest(cmd)
        require_sweep(mp, "delete the pod")
        return

    # --- one-shot gated actions ---
    for pat, appr, what, pod_only in ONE_SHOT:
        if re.search(pat, cmd) and (on_pod or not pod_only):
            one_shot(appr, what)
            return

    if on_pod and re.search(POD_SYNC, cmd):
        _, mp = load_manifest(cmd)
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


def close_pod(closed_by):
    """Append the running pod to the ledger and clear pod state. No-op if none recorded."""
    if not (STATE / "pod-created-at").exists():
        return
    usd, stem = running_pod()
    entry = {
        "manifest": stem,
        "created_at": (STATE / "pod-created-at").read_text().strip(),
        "closed_at": time.time(),
        "hours": round(pod_hours(), 3),
        "usd_est": round(usd, 2),
        "closed_by": closed_by,
    }
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    for f in ("pod-created-at", "pod-id", "pod-manifest"):
        try:
            (STATE / f).unlink()
        except FileNotFoundError:
            pass


def post_bash(cmd, resp):
    STATE.mkdir(parents=True, exist_ok=True)
    ok = succeeded(resp)
    out = " ".join(str(resp.get(k, "")) for k in ("stdout", "stderr")) if isinstance(resp, dict) else ""

    if re.search(r"\brunpodctl\s+pod\s+create\b", cmd) and ok:
        if not re.search(r"error|failed|no longer any instances", out, re.I):
            (STATE / "pod-created-at").write_text(str(time.time()))
            (STATE / "pod-manifest").write_text(active_manifest_rel() if "--manifest" not in cmd
                                                else manifest_path(cmd).relative_to(ROOT).as_posix())
            m = re.search(r"\b([a-z0-9]{12,16})\b", out)
            if m:
                (STATE / "pod-id").write_text(m.group(1))

    if re.search(r"\brunpodctl\s+(pod\s+delete|remove\s+pod)\b", cmd) and ok:
        close_pod("delete")

    # Pod self-terminated by sweep_driver.sh: the hook never sees the delete. An empty
    # `pod list` is the evidence; close at "now", which can only over-estimate cost.
    if re.search(r"\brunpodctl\s+pod\s+list\b", cmd) and ok and re.fullmatch(r"\s*(\[\s*\])?\s*", out):
        close_pod("pod-list-empty (self-terminated; estimate errs high)")

    lm = re.search(LAUNCH_INVOKE, cmd)
    if lm and "--dry-run" not in cmd and ok:
        (STATE / f"launched-{lm.group(1)}").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

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
    if len(sys.argv) > 1 and sys.argv[1] == "--ledger":
        ledger_summary()
        sys.exit(0)
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
        pre_file(tool, ti)
    elif tool == "Bash":
        pre_bash(str(ti.get("command", "")))
    sys.exit(0)


if __name__ == "__main__":
    main()
