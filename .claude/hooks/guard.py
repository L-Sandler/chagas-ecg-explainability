#!/usr/bin/env python3
"""PreToolUse + PostToolUse guardrail hook (v4, staged draft 2026-10-10).

Division of labour (see .guardrails/README.md):
  * File protection (config, manifests, decision records, gated src files) is enforced by
    permission deny rules + the OS sandbox (settings.json). This hook keeps only a small
    file-tool check as a backstop, plus the two things the sandbox cannot do:
      - decision gate: gated src files / decision records are editable by the Edit/Write
        tools only against an approved (resp. proposed) record in notes/decisions/;
      - stateful gates: manifest, launch-once, budget ledger, pod lifecycle, one-shots.
  * Command matching works on command POSITION (not any mention): quoted strings and heredoc
    bodies are ignored unless the command runs a remote/inner shell (ssh, bash -c).
  * Fails closed: an internal error blocks the call (exit 2); settings.json adds
    "onFailure": "block" for crashes and timeouts.

Changes from v3: pod state is recorded for scripts/create_pod.sh (v3 only saw the literal
`runpodctl pod create`, so the ledger stayed empty); locked-split check runs before the
early-return gates; scp into guardrail paths is refused (scp runs outside the sandbox);
stale approvals are listed by --ledger; shell-write regexes were dropped (sandbox does it).

Protocol: stdin JSON {hook_event_name, tool_name, tool_input, tool_response}.
PreToolUse: exit 2 + stderr = block. PostToolUse: exit 0 always (bookkeeping only).
`python3 guard.py --ledger` prints current spend and approvals (no stdin needed).
"""
import json
import os
import re
import shlex
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

# Backstop for the file tools (the deny rules in settings.json are the primary control).
PROTECTED_PATHS = (
    ".guardrails/", ".claude/", "spec/runs/",  # notes/decisions/ has its own rule (propose-only)
    "scripts/launch_run.py", "scripts/sweep_driver.sh",
    "CLAUDE.md",
    "README.md",  # root README only (rel == "README.md"): edited only with the user's permission
)

# Research-relevant code. Editing these is a design-affecting act (G4).
GATED_PATHS = (
    "src/preprocess.py", "src/dataset.py", "src/model.py", "src/train.py", "src/explain.py",
)

# Where a command word may start: line start, after ; & | ( ` quotes, newline, sudo, xargs.
# `grep kill file` and `echo reset` do not match; `ssh h "kill 1"` and `a && kill 1` do.
CMD = r"(?:^|[;&|(`\"'\n]|\bsudo\b|\bxargs\b)\s*(?:\S*/)?"


def at_cmd(body):
    return CMD + "(?:" + body + ")"


ALWAYS_BLOCK = [(at_cmd(b), why) for b, why in [
    (r"rm\s+(-[A-Za-z]*[rR][A-Za-z]*\s|--recursive)", "recursive delete"),
    (r"rm\b[^\n]*(/workspace|lightning_logs|\bdata/|\.ckpt)",
     "delete of training data, checkpoints or the volume"),
    (r"git\s+push\b[^\n]*(\s--force\b|\s-f\b|--force-with-lease)", "force push"),
    (r"git\s+branch\s+(-D|--delete)", "branch deletion"),
    (r"git\s+reset\s+--hard", "git reset --hard"),
    (r"git\s+clean\b", "git clean"),
    (r"find\b[^\n]*(-delete\b|-exec\s+rm\b)", "find-based delete"),
    (r"runpodctl\s+pod\s+(stop|reset|restart)\b", "pod stop/reset/restart (use delete instead)"),
    (r"runpodctl\s+(network-volume|volume)\s+(delete|rm)", "network volume deletion"),
    (r"(shutdown|reboot|poweroff)\b", "pod shutdown/reboot"),
]]
PY_DELETE = r"\bshutil\.rmtree\b"  # checked on the raw command (python -c "..." is quoted)

# Built from pieces so this file never contains the literal flag (a content scan would hit it).
UNLOCK_PATTERN = r"--unlock" + r"-final\b|\bunlock_final" + r"\s*=\s*True"

ONE_SHOT = [  # (pattern, approval file, description, pod_only)
    (at_cmd(r"kill|pkill|killall") + r"\b", "approve-kill", "kill a process", False),
    (at_cmd(r"uv\s+pip|uv\s+sync|pip3?\s+install|apt(-get)?\s+install|wget\s|curl\s+-[^|]*-O"),
     "approve-pod-install", "install or download on the pod", True),
]

POD_SYNC = at_cmd(r"git\s+(pull|fetch|checkout|merge|rebase|stash|switch|reset|commit|add)\b")

# --- invocation patterns: a bare mention of a filename is not an invocation ---
TRAIN_INVOKE = r"(\bpython[0-9.]*|\buv\s+run)\b[^|;&\n]*\bsrc/train\.py"
LAUNCH_INVOKE = r"(?:\bpython[0-9.]*\s+|\./)(?:\S*/)?scripts/launch_run\.py\s+([A-Za-z0-9][A-Za-z0-9._-]*)"
SWEEP_INVOKE = r"(?:\./|\b(?:bash|sh|setsid|nohup)\s+(?:-\S+\s+)*)(?:\S*/)?scripts/sweep_driver\.sh"
CREATE_INVOKE = (r"(?:\./|\b(?:bash|sh|setsid|nohup)\s+(?:-\S+\s+)*)(?:\S*/)?scripts/create_pod\.sh|"
                 + at_cmd(r"runpodctl\s+pod\s+create\b"))
DELETE_INVOKE = at_cmd(r"runpodctl\s+(pod\s+delete|remove\s+pod)\b")
LIST_INVOKE = at_cmd(r"runpodctl\s+pod\s+list\b")
GET_INVOKE = at_cmd(r"runpodctl\s+pod\s+get\b")
SCP_INVOKE = at_cmd(r"scp\s")
GUARD_PATH_RE = r"\.guardrails|\.claude/|spec/runs|notes/decisions|launch_run\.py|sweep_driver\.sh"

QUOTED = re.compile(r"\"(?:\\.|[^\"\\])*\"|'[^']*'")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\b", re.S)


def block(msg):
    sys.stderr.write("GUARDRAIL BLOCKED: " + msg + "\n")
    sys.exit(2)


def read_json(p):
    return json.loads(Path(p).read_text())


def analyze(cmd):
    """(text to match patterns on, running-on-pod?). Quoted strings and heredoc bodies are
    blanked unless the command runs an inner shell (ssh, bash -c), where they are the command."""
    plain = QUOTED.sub('""', HEREDOC.sub("<<", cmd))
    on_pod = bool(re.search(CMD + r"ssh\b", plain))
    inner_shell = bool(re.search(CMD + r"(?:ba|z)?sh\s+(?:-\S+\s+)*-c\b", plain))
    return (cmd if (on_pod or inner_shell) else plain), on_pod


# ----------------------------- manifest / approvals -----------------------------

def active_manifest_rel():
    try:
        v = (GUARD / "active-manifest").read_text().strip()
        if v:
            return v
    except OSError:
        pass
    return DEFAULT_MANIFEST


def manifest_path(cmd=""):
    m = (re.search(r"--manifest\s+(\S+)", cmd)
         or re.search(r"(?:sweep_driver|create_pod)\.sh\s+(\S+\.json)\b", cmd))
    return ROOT / (m.group(1) if m else active_manifest_rel())


def load_manifest(cmd=""):
    p = manifest_path(cmd)
    try:
        return read_json(p), p
    except Exception as e:
        block(f"cannot read run manifest {p}: {e}")


def sweep_approval(mp):
    """Path to the plan-level approval file for this manifest, or None if absent."""
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


# ----------------------------- budget -----------------------------

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


def _norm_gpu(s):
    return re.sub(r"[^a-z0-9]", "", s.lower()).replace("nvidia", "")


def pod_price(manifest):
    """($/hr, source). Best source first: the rate RunPod reported for this pod, then the
    manifest's listed price for this GPU, then the highest listed price (errs high)."""
    try:
        return float((STATE / "pod-price").read_text().strip()), "reported by RunPod"
    except (OSError, ValueError):
        pass
    try:
        gpu = _norm_gpu((STATE / "pod-gpu").read_text())
    except OSError:
        gpu = ""
    prices = manifest.get("budget", {}).get("gpu_price_usd_per_hr") or {}
    for name, v in prices.items():
        if gpu and isinstance(v, (int, float)) and _norm_gpu(name) == gpu:
            return float(v), "manifest price for this GPU"
    return price_per_hr(manifest), "manifest, highest listed GPU"


def pod_cost():
    """Details for the currently recorded pod, or None."""
    if not (STATE / "pod-created-at").exists():
        return None
    try:
        rel = (STATE / "pod-manifest").read_text().strip()
    except OSError:
        rel = active_manifest_rel()
    try:
        manifest = read_json(ROOT / rel)
        price, source = pod_price(manifest)
    except Exception:
        manifest, price, source = {}, DEFAULT_ASSUMED_USD_PER_HR, "assumed default"
    try:
        gpu = (STATE / "pod-gpu").read_text().strip()
    except OSError:
        gpu = None
    hours = pod_hours()
    return {"manifest": Path(rel).stem, "gpu": gpu, "price": price, "source": source,
            "hours": hours, "usd": hours * price,
            "max_hours": float(manifest.get("budget", {}).get("max_pod_hours", 0) or 0)}


def running_pod():
    """(estimated USD so far, manifest stem) for the currently recorded pod, else (0, None)."""
    c = pod_cost()
    return (c["usd"], c["manifest"]) if c else (0.0, None)


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
    b = manifest.get("budget", {})
    cap_h = float(b.get("max_pod_hours", 0) or 0)
    used_h = pod_hours()
    if cap_h and used_h > cap_h:
        block(f"pod-hours budget exceeded ({used_h:.2f}h used, cap {cap_h}h). Refusing to {what}. "
              f"Pull outputs, delete the pod, and report to the user.")

    g_cap, before = load_budget()
    g_spent = before + spend()
    if g_spent >= g_cap:
        block(f"global budget reached (${g_spent:.2f} est. of ${g_cap:.2f}). Refusing to {what}. "
              f"Report to the user; only they can raise .guardrails/budget.json.")

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
    print("Pods (estimates):")
    per_manifest = {}
    for e in ledger_entries():
        when = time.strftime("%Y-%m-%d", time.localtime(float(e.get("closed_at") or 0)))
        usd = float(e.get("usd_est", 0))
        per_manifest[e.get("manifest")] = per_manifest.get(e.get("manifest"), 0.0) + usd
        print(f"  {when} {e.get('manifest')}: {e.get('gpu') or 'GPU unknown'}, "
              f"{float(e.get('hours', 0)):.2f}h x ${float(e.get('price_per_hr', 0)):.2f}/h = ${usd:.2f} "
              f"[{e.get('price_source', 'n/a')}] runs: {', '.join(e.get('runs') or []) or '-'} "
              f"({e.get('closed_by')})")
    c = pod_cost()
    if c:
        print(f"  RUNNING {c['manifest']}: {c['gpu'] or 'GPU unknown'}, {c['hours']:.2f}h x "
              f"${c['price']:.2f}/h = ${c['usd']:.2f} so far [{c['source']}]")
        per_manifest[c["manifest"]] = per_manifest.get(c["manifest"], 0.0) + c["usd"]
    if per_manifest:
        print("By manifest: " + ", ".join(f"{k} ${v:.2f}" for k, v in per_manifest.items()))
    tracked = spend()
    print(f"Tracked by this ledger: ${tracked:.2f}")
    if isinstance(before, (int, float)):
        print(f"Spent before the ledger (you set it): ${before:.2f}")
        total = before + tracked
        line = f"Total GPU spend: ${total:.2f}"
        if isinstance(g_cap, (int, float)):
            line += f" of ${g_cap:.2f} budget, ${g_cap - total:.2f} remaining"
        print(line)
    print("Not tracked: network volume storage and anything not created through this hook. "
          "Reconcile with the RunPod billing page and edit .guardrails/state/ledger.jsonl or "
          "spent_before_ledger_usd if the drift matters.")
    print("approvals present:")
    found = False
    for f in sorted(GUARD.glob("approve-*")) + sorted(GUARD.glob("unlock-*")):
        found = True
        age = (time.time() - f.stat().st_mtime) / 3600.0
        print(f"  {f.name}: {age:.1f}h old")
    if not found:
        print("  (none)")


# ----------------------------- design-decision gate -----------------------------

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


# ----------------------------- PreToolUse -----------------------------

def rel_of(p):
    """Repo-relative path with symlinks and '..' resolved; '../x' means outside the repo."""
    p = str(p)
    if not p:
        return ""
    ap = os.path.realpath(p if os.path.isabs(p) else str(ROOT / p))
    try:
        return os.path.relpath(ap, os.path.realpath(ROOT))
    except ValueError:
        return ap


def pre_file(tool, ti):
    rel = rel_of(ti.get("file_path") or ti.get("notebook_path") or "")
    if rel.startswith("notes/decisions/"):
        check_decision_write(rel, tool, ti)
        return
    for prot in PROTECTED_PATHS:
        if rel == prot.rstrip("/") or rel.startswith(prot):
            block(f"'{rel}' is user-owned guardrail/decision config; the agent does not edit it.")
    require_decision(rel)
    # A script that opens the locked split would dodge the command-line check, so inspect
    # the content being written too. The gated source files are covered by the gate above.
    if rel not in GATED_PATHS and rel.endswith((".py", ".sh", ".ipynb")):
        text = " ".join(str(v) for k, v in ti.items() if k in ("content", "new_string")) + " ".join(
            str(e.get("new_string", "")) for e in ti.get("edits", []) or [])
        if re.search(UNLOCK_PATTERN, text):
            one_shot("unlock-final-test", f"write code that opens the locked-final split ({rel})")


def scp_check(text):
    seg = next((s for s in re.split(r"[;&|]", text) if re.search(SCP_INVOKE, s)), "")
    try:
        toks = shlex.split(seg)
    except ValueError:
        toks = seg.split()
    args = [t for t in toks[1:] if not t.startswith("-")]
    dest = args[-1] if args else ""
    if re.search(GUARD_PATH_RE, dest):
        block(f"scp into a guardrail path ({dest}); scp runs outside the sandbox, so this is refused "
              f"outright.")
    ok = ("lightning_logs" in dest or "/scratchpad" in dest
          or (":" in dest and ("/workspace/" in dest or "chagas-ecg-explainability" in dest)))
    if not ok:
        one_shot("approve-scp", "scp to a destination outside lightning_logs/ or the pod volume")


def pre_bash(cmd):
    text, on_pod = analyze(cmd)

    for pat, why in ALWAYS_BLOCK:
        if re.search(pat, text):
            block(f"{why} is never run by the agent. The user runs it by hand if they want it.")
    if re.search(PY_DELETE, cmd):
        block("recursive delete (shutil.rmtree) is never run by the agent.")

    # --- one-shot gated actions (all matching ones must be approved, before any early return) ---
    if re.search(UNLOCK_PATTERN, cmd):  # raw: also catches a quoted flag
        one_shot("unlock-final-test", "open the locked-final split (final evaluation only)")
    for pat, appr, what, pod_only in ONE_SHOT:
        if re.search(pat, text) and (on_pod or not pod_only):
            one_shot(appr, what)

    # --- training launches ---
    if re.search(TRAIN_INVOKE, text) and not re.search(LAUNCH_INVOKE, text) \
            and not re.search(SWEEP_INVOKE, text):
        if "--fast" in text and not on_pod:
            pass  # local CPU smoke run: allowed
        else:
            block("direct train.py launches are not allowed (only local `--fast` smoke runs); "
                  "use scripts/launch_run.py <run-name> or scripts/sweep_driver.sh "
                  "(both manifest-driven).")

    if re.search(SWEEP_INVOKE, text):
        manifest, mp = load_manifest(cmd)
        require_sweep(mp, "run the sweep driver")
        check_budget(manifest, mp, "start the sweep driver")
        return

    lm = re.search(LAUNCH_INVOKE, text)
    if lm and "--dry-run" not in text:
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
    if re.search(CREATE_INVOKE, text) and not re.search(r"\s(--help|-h)\b", text):
        manifest, mp = load_manifest(cmd)
        if (STATE / "pod-created-at").exists():
            block("a pod is already recorded (max_concurrent_pods=1). Delete it first, or if that "
                  "record is stale ask the user to remove .guardrails/state/pod-created-at")
        check_budget(manifest, mp, "create a pod")
        require_sweep(mp, "provision a pod")
        return

    if re.search(DELETE_INVOKE, text):
        _, mp = load_manifest(cmd)
        require_sweep(mp, "delete the pod")
        return

    if on_pod and re.search(POD_SYNC, text):
        _, mp = load_manifest(cmd)
        require_sweep(mp, "sync the repo on the pod")
        return

    if re.search(SCP_INVOKE, text):
        scp_check(text)


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


def close_pod(closed_by, cap_at_max_hours=False):
    """Append the running pod to the ledger and clear pod state. No-op if none recorded.
    cap_at_max_hours: for a pod that self-terminated, the driver stops it at max_pod_hours, so
    the bill cannot exceed that even if we only noticed much later."""
    c = pod_cost()
    if not c:
        return
    created = (STATE / "pod-created-at").read_text().strip()
    hours = c["hours"]
    if cap_at_max_hours and c["max_hours"]:
        hours = min(hours, c["max_hours"])
    try:
        runs = sorted(p.name[len("launched-"):] for p in STATE.glob("launched-*")
                      if p.stat().st_mtime >= float(created))
    except ValueError:
        runs = []
    entry = {
        "manifest": c["manifest"],
        "gpu": c["gpu"],
        "created_at": created,
        "closed_at": time.time(),
        "hours": round(hours, 3),
        "price_per_hr": round(c["price"], 4),
        "price_source": c["source"],
        "usd_est": round(hours * c["price"], 2),
        "runs": runs,
        "closed_by": closed_by,
    }
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    for f in ("pod-created-at", "pod-id", "pod-manifest", "pod-price", "pod-gpu"):
        try:
            (STATE / f).unlink()
        except FileNotFoundError:
            pass


def capture_price(out):
    """Best-effort: pull the hourly rate and GPU name out of pod create/get output."""
    m = re.search(r'"costPerHr"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)', out)
    if m:
        (STATE / "pod-price").write_text(m.group(1))
    g = (re.search(r"->\s+(.+?)\s+has stock", out)
         or re.search(r'"(?:gpuTypeId|gpuId)"\s*:\s*"([^"]+)"', out))
    if g:
        (STATE / "pod-gpu").write_text(g.group(1))


def post_bash(cmd, resp):
    STATE.mkdir(parents=True, exist_ok=True)
    text, on_pod = analyze(cmd)
    ok = succeeded(resp)
    out = " ".join(str(resp.get(k, "")) for k in ("stdout", "stderr")) if isinstance(resp, dict) else ""

    if re.search(CREATE_INVOKE, text) and not re.search(r"\s(--help|-h)\b", text) and ok:
        if not re.search(r"error|failed|no longer any instances", out, re.I):
            (STATE / "pod-created-at").write_text(str(time.time()))
            (STATE / "pod-manifest").write_text(rel_of(manifest_path(cmd)))
            m = (re.search(r'"id"\s*:\s*"([a-z0-9]+)"', out)
                 or re.search(r"\b([a-z0-9]{12,16})\b", out))
            if m:
                (STATE / "pod-id").write_text(m.group(1))
            capture_price(out)

    # `runpodctl pod get` (the facilitator runs it right after create) reports the hourly rate.
    if re.search(GET_INVOKE, text) and ok and (STATE / "pod-created-at").exists():
        capture_price(out)

    if re.search(DELETE_INVOKE, text) and ok:
        close_pod("delete")

    # Pod self-terminated by sweep_driver.sh: the hook never sees the delete. An empty
    # `pod list` is the evidence; closed at "now" but never past the manifest's max_pod_hours.
    if re.search(LIST_INVOKE, text) and ok and re.fullmatch(r"\s*(\[\s*\])?\s*", out):
        close_pod("pod-list-empty (self-terminated)", cap_at_max_hours=True)

    lm = re.search(LAUNCH_INVOKE, text)
    if lm and "--dry-run" not in text and ok:
        (STATE / f"launched-{lm.group(1)}").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    if ok:
        consumed = []
        if re.search(UNLOCK_PATTERN, cmd):
            consumed.append("unlock-final-test")
        consumed += [appr for pat, appr, _w, pod_only in ONE_SHOT
                     if re.search(pat, text) and (on_pod or not pod_only)]
        if re.search(SCP_INVOKE, text):
            consumed.append("approve-scp")
        for name in consumed:
            try:
                (GUARD / name).unlink()
            except FileNotFoundError:
                pass


def log_error(msg):
    try:
        STATE.mkdir(parents=True, exist_ok=True)
        with (STATE / "hook-errors.log").open("a") as fh:
            fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")
    except OSError:
        pass


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ledger":
        ledger_summary()
        sys.exit(0)
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.stderr.write("GUARDRAIL BLOCKED: hook could not parse its input; failing closed.\n")
        sys.exit(2)
    event = payload.get("hook_event_name", "PreToolUse")
    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input", {}) or {}

    if event == "PostToolUse":
        if tool == "Bash":
            try:
                post_bash(str(ti.get("command", "")), payload.get("tool_response", {}))
            except Exception as e:
                log_error(f"post_bash failed: {type(e).__name__}: {e}")
        sys.exit(0)

    try:
        if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            pre_file(tool, ti)
        elif tool == "Bash":
            pre_bash(str(ti.get("command", "")))
    except Exception as e:  # SystemExit from block() is not an Exception, so it passes through
        log_error(f"pre-hook failed: {type(e).__name__}: {e}")
        block(f"hook internal error ({type(e).__name__}: {e}); failing closed. Tell the user.")
    sys.exit(0)


if __name__ == "__main__":
    main()
